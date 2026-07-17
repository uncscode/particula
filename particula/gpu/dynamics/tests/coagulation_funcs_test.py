"""Parity tests for GPU coagulation composite functions."""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

wp: Any = None
try:
    import warp as wp
except ImportError:
    pass

pytestmark = (
    [pytest.mark.warp, pytest.mark.skip(reason="Warp not installed")]
    if wp is None
    else pytest.mark.warp
)

if wp is not None:
    from particula.dynamics.coagulation.brownian_kernel import (  # noqa: E402
        _brownian_diffusivity,
        _g_collection_term,
        _mean_free_path_l,
        get_brownian_kernel,
        get_brownian_kernel_via_system_state,
    )
    from particula.gpu.dynamics.coagulation_funcs import (  # noqa: E402
        brownian_diffusivity_wp,
        brownian_kernel_pair_wp,
        coulomb_continuum_limit_wp,
        coulomb_kinetic_limit_wp,
        coulomb_potential_ratio_wp,
        diffusive_knudsen_number_wp,
        g_collection_term_wp,
        particle_mean_free_path_wp,
        reduced_value_wp,
    )
    from particula.gpu.properties.gas_properties import (  # noqa: E402
        dynamic_viscosity_wp,
        molecule_mean_free_path_wp,
    )
    from particula.gpu.properties.particle_properties import (  # noqa: E402
        aerodynamic_mobility_wp,
        cunningham_slip_correction_wp,
        knudsen_number_wp,
        mean_thermal_speed_wp,
    )
    from particula.gpu.tests.cuda_availability import warp_devices  # noqa: E402
    from particula.util.constants import (  # noqa: E402
        BOLTZMANN_CONSTANT,
        ELECTRIC_PERMITTIVITY,
        ELEMENTARY_CHARGE_VALUE,
        GAS_CONSTANT,
        MOLECULAR_WEIGHT_AIR,
        REF_TEMPERATURE_STP,
        REF_VISCOSITY_AIR_STP,
        SUTHERLAND_CONSTANT,
    )


def _warp_kernel(function):
    """Decorate kernels only when Warp is available."""
    if wp is None:
        return function
    return wp.kernel(function)


def _available_warp_devices() -> list[str]:
    """Return collection-safe Warp device params."""
    if wp is None:
        return ["cpu"]
    return warp_devices(wp)


@_warp_kernel
def _brownian_diffusivity_kernel(
    temperatures: Any,
    aerodynamic_mobilities: Any,
    boltzmann_constant: Any,
    result: Any,
) -> None:
    """Compute Brownian diffusivity for each sample.

    Args:
        temperatures: Gas temperatures [K].
        aerodynamic_mobilities: Aerodynamic mobilities [m²/s].
        boltzmann_constant: Boltzmann constant [J/K].
        result: Output array for Brownian diffusivities [m²/s].
    """
    tid = wp.tid()
    result[tid] = brownian_diffusivity_wp(
        temperatures[tid],
        aerodynamic_mobilities[tid],
        boltzmann_constant,
    )


@_warp_kernel
def _particle_mean_free_path_kernel(
    diffusivities: Any,
    mean_thermal_speeds: Any,
    result: Any,
) -> None:
    """Compute particle mean free path for each sample.

    Args:
        diffusivities: Particle diffusivities [m²/s].
        mean_thermal_speeds: Mean thermal speeds [m/s].
        result: Output array for mean free paths [m].
    """
    tid = wp.tid()
    result[tid] = particle_mean_free_path_wp(
        diffusivities[tid],
        mean_thermal_speeds[tid],
    )


@_warp_kernel
def _g_collection_term_kernel(
    mean_free_paths: Any,
    particle_radii: Any,
    result: Any,
) -> None:
    """Compute Brownian collection term for each sample.

    Args:
        mean_free_paths: Particle mean free paths [m].
        particle_radii: Particle radii [m].
        result: Output array for collection terms (dimensionless).
    """
    tid = wp.tid()
    result[tid] = g_collection_term_wp(
        mean_free_paths[tid],
        particle_radii[tid],
    )


@_warp_kernel
def _brownian_kernel_pair_kernel(
    radii_i: Any,  # wp.array(dtype=wp.float64)
    radii_j: Any,  # wp.array(dtype=wp.float64)
    diffusivities_i: Any,  # wp.array(dtype=wp.float64)
    diffusivities_j: Any,  # wp.array(dtype=wp.float64)
    g_terms_i: Any,  # wp.array(dtype=wp.float64)
    g_terms_j: Any,  # wp.array(dtype=wp.float64)
    speeds_i: Any,  # wp.array(dtype=wp.float64)
    speeds_j: Any,  # wp.array(dtype=wp.float64)
    alpha_values: Any,  # wp.array(dtype=wp.float64)
    result: Any,  # wp.array(dtype=wp.float64)
) -> None:
    """Compute scalar Brownian kernel for each pair sample.

    Args:
        radii_i: Particle radii for sample i [m].
        radii_j: Particle radii for sample j [m].
        diffusivities_i: Diffusivities for particle i [m²/s].
        diffusivities_j: Diffusivities for particle j [m²/s].
        g_terms_i: Collection terms for particle i (dimensionless).
        g_terms_j: Collection terms for particle j (dimensionless).
        speeds_i: Mean thermal speeds for particle i [m/s].
        speeds_j: Mean thermal speeds for particle j [m/s].
        alpha_values: Collision efficiency values (dimensionless).
        result: Output array for Brownian kernel values [m³/s].
    """
    tid = wp.tid()
    result[tid] = brownian_kernel_pair_wp(
        radii_i[tid],
        radii_j[tid],
        diffusivities_i[tid],
        diffusivities_j[tid],
        g_terms_i[tid],
        g_terms_j[tid],
        speeds_i[tid],
        speeds_j[tid],
        alpha_values[tid],
    )


@_warp_kernel
def _brownian_chain_kernel(
    temperatures: Any,  # wp.array(dtype=wp.float64)
    pressures: Any,  # wp.array(dtype=wp.float64)
    particle_radii_i: Any,  # wp.array(dtype=wp.float64)
    particle_radii_j: Any,  # wp.array(dtype=wp.float64)
    particle_masses_i: Any,  # wp.array(dtype=wp.float64)
    particle_masses_j: Any,  # wp.array(dtype=wp.float64)
    alpha_values: Any,  # wp.array(dtype=wp.float64)
    boltzmann_constant: Any,  # wp.float64
    gas_constant: Any,  # wp.float64
    molecular_weight_air: Any,  # wp.float64
    ref_viscosity: Any,  # wp.float64
    ref_temperature: Any,  # wp.float64
    sutherland_constant: Any,  # wp.float64
    result: Any,  # wp.array(dtype=wp.float64)
) -> None:
    """Compute Brownian kernel from chained GPU property functions.

    Args:
        temperatures: Gas temperatures [K].
        pressures: Gas pressures [Pa].
        particle_radii_i: Radii for particle i [m].
        particle_radii_j: Radii for particle j [m].
        particle_masses_i: Masses for particle i [kg].
        particle_masses_j: Masses for particle j [kg].
        alpha_values: Collision efficiency values (dimensionless).
        boltzmann_constant: Boltzmann constant [J/K].
        gas_constant: Universal gas constant [J/(mol·K)].
        molecular_weight_air: Molecular weight of air [kg/mol].
        ref_viscosity: Reference viscosity at STP [Pa·s].
        ref_temperature: Reference temperature at STP [K].
        sutherland_constant: Sutherland constant [K].
        result: Output array for Brownian kernel values [m³/s].
    """
    tid = wp.tid()
    dynamic_viscosity = dynamic_viscosity_wp(
        temperatures[tid],
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperatures[tid],
        pressures[tid],
        dynamic_viscosity,
        gas_constant,
    )
    knudsen_i = knudsen_number_wp(mean_free_path, particle_radii_i[tid])
    knudsen_j = knudsen_number_wp(mean_free_path, particle_radii_j[tid])
    slip_i = cunningham_slip_correction_wp(knudsen_i)
    slip_j = cunningham_slip_correction_wp(knudsen_j)
    mobility_i = aerodynamic_mobility_wp(
        particle_radii_i[tid],
        slip_i,
        dynamic_viscosity,
    )
    mobility_j = aerodynamic_mobility_wp(
        particle_radii_j[tid],
        slip_j,
        dynamic_viscosity,
    )
    diffusivity_i = brownian_diffusivity_wp(
        temperatures[tid],
        mobility_i,
        boltzmann_constant,
    )
    diffusivity_j = brownian_diffusivity_wp(
        temperatures[tid],
        mobility_j,
        boltzmann_constant,
    )
    speed_i = mean_thermal_speed_wp(
        particle_masses_i[tid],
        temperatures[tid],
        boltzmann_constant,
    )
    speed_j = mean_thermal_speed_wp(
        particle_masses_j[tid],
        temperatures[tid],
        boltzmann_constant,
    )
    mean_free_path_i = particle_mean_free_path_wp(diffusivity_i, speed_i)
    mean_free_path_j = particle_mean_free_path_wp(diffusivity_j, speed_j)
    g_term_i = g_collection_term_wp(mean_free_path_i, particle_radii_i[tid])
    g_term_j = g_collection_term_wp(mean_free_path_j, particle_radii_j[tid])
    result[tid] = brownian_kernel_pair_wp(
        particle_radii_i[tid],
        particle_radii_j[tid],
        diffusivity_i,
        diffusivity_j,
        g_term_i,
        g_term_j,
        speed_i,
        speed_j,
        alpha_values[tid],
    )


@_warp_kernel
def _coulomb_potential_ratio_kernel(
    radii_i: Any,
    radii_j: Any,
    charges_i: Any,
    charges_j: Any,
    temperatures: Any,
    boltzmann_constant: Any,
    elementary_charge_value: Any,
    electric_permittivity: Any,
    result: Any,
) -> None:
    """Compute the Coulomb potential ratio for each scalar pair."""
    tid = wp.tid()
    result[tid] = coulomb_potential_ratio_wp(
        radii_i[tid],
        radii_j[tid],
        charges_i[tid],
        charges_j[tid],
        temperatures[tid],
        boltzmann_constant,
        elementary_charge_value,
        electric_permittivity,
    )


@_warp_kernel
def _reduced_value_kernel(left: Any, right: Any, result: Any) -> None:
    """Compute reduced scalar values."""
    tid = wp.tid()
    result[tid] = reduced_value_wp(left[tid], right[tid])


@_warp_kernel
def _coulomb_kinetic_limit_kernel(potential: Any, result: Any) -> None:
    """Compute kinetic Coulomb enhancement factors."""
    tid = wp.tid()
    result[tid] = coulomb_kinetic_limit_wp(potential[tid])


@_warp_kernel
def _coulomb_continuum_limit_kernel(potential: Any, result: Any) -> None:
    """Compute continuum Coulomb enhancement factors."""
    tid = wp.tid()
    result[tid] = coulomb_continuum_limit_wp(potential[tid])


@_warp_kernel
def _diffusive_knudsen_number_kernel(
    radii_i: Any,
    radii_j: Any,
    masses_i: Any,
    masses_j: Any,
    frictions_i: Any,
    frictions_j: Any,
    potentials: Any,
    temperatures: Any,
    boltzmann_constant: Any,
    result: Any,
) -> None:
    """Compute pair diffusive Knudsen numbers."""
    tid = wp.tid()
    result[tid] = diffusive_knudsen_number_wp(
        radii_i[tid],
        radii_j[tid],
        masses_i[tid],
        masses_j[tid],
        frictions_i[tid],
        frictions_j[tid],
        potentials[tid],
        temperatures[tid],
        boltzmann_constant,
    )


@pytest.fixture(params=_available_warp_devices())
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def _coulomb_potential_ratio_oracle(
    radius_i: float,
    radius_j: float,
    charge_i: float,
    charge_j: float,
    temperature: float,
) -> float:
    """Model the scalar GPU Coulomb-potential contract independently."""
    sum_radius = radius_i + radius_j
    if sum_radius <= 0.0 or temperature <= 0.0:
        return 0.0
    value = -(
        charge_i * charge_j * ELEMENTARY_CHARGE_VALUE * ELEMENTARY_CHARGE_VALUE
    ) / (
        4.0
        * np.pi
        * ELECTRIC_PERMITTIVITY
        * sum_radius
        * BOLTZMANN_CONSTANT
        * temperature
    )
    return max(value, -200.0)


def _reduced_value_oracle(left: float, right: float) -> float:
    """Model the scalar GPU reduced-value contract independently."""
    if left + right <= 0.0:
        return 0.0
    return left * right / (left + right)


def _kinetic_limit_oracle(potential: float) -> float:
    """Model the scalar GPU kinetic-limit equation independently."""
    return 1.0 + potential if potential >= 0.0 else float(np.exp(potential))


def _continuum_limit_oracle(potential: float) -> float:
    """Model the scalar GPU continuum-limit equation independently."""
    if potential == 0.0:
        return 1.0
    return potential / -float(np.expm1(-potential))


def _diffusive_knudsen_number_oracle(
    radius_i: float,
    radius_j: float,
    mass_i: float,
    mass_j: float,
    friction_i: float,
    friction_j: float,
    potential: float,
    temperature: float,
) -> float:
    """Model the scalar GPU diffusive-Knudsen contract independently."""
    sum_radius = radius_i + radius_j
    if sum_radius <= 0.0 or temperature <= 0.0:
        return 0.0
    reduced_mass = _reduced_value_oracle(mass_i, mass_j)
    reduced_friction = _reduced_value_oracle(friction_i, friction_j)
    if reduced_mass <= 0.0 or reduced_friction <= 0.0:
        return 0.0
    kinetic_limit = _kinetic_limit_oracle(potential)
    if kinetic_limit < 1.0e-80:
        return 0.0
    continuum_limit = _continuum_limit_oracle(potential)
    numerator = np.sqrt(BOLTZMANN_CONSTANT * temperature * reduced_mass)
    denominator = (
        reduced_friction * sum_radius * continuum_limit / kinetic_limit
    )
    return float(numerator / denominator)


def _warp_array(values: np.ndarray, device: str) -> Any:
    """Create a float64 Warp array for a scalar parity probe."""
    return wp.array(values, dtype=wp.float64, device=device)


def _assert_casewise_allclose(
    observed: np.ndarray,
    expected: np.ndarray,
    case_names: tuple[str, ...],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare one launched probe batch while retaining case diagnostics."""
    for case_index, (case_name, observed_value, expected_value) in enumerate(
        zip(case_names, observed, expected, strict=True)
    ):
        npt.assert_allclose(
            observed_value,
            expected_value,
            rtol=rtol,
            atol=atol,
            err_msg=f"case {case_index}: {case_name}",
        )


@pytest.mark.gpu_parity
def test_coulomb_potential_ratio_wp_matches_independent_oracle(
    device: str,
) -> None:
    """Validate neutral, attractive, repulsive, and safe Coulomb branches."""
    radii_i = np.array(
        [1e-8, 1e-8, 1e-8, 1e-8, 0.0, -1e-8, 1e-8, 1e-8],
        dtype=np.float64,
    )
    radii_j = np.array(
        [2e-8, 2e-8, 2e-8, 1e-8, 0.0, 0.0, 2e-8, 2e-8],
        dtype=np.float64,
    )
    charges_i = np.array(
        [0.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
    )
    charges_j = np.array(
        [0.0, -1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
    )
    temperatures = np.array(
        [298.15, 298.15, 298.15, 298.15, 298.15, 298.15, -1.0, 0.0]
    )
    expected = np.array(
        [
            _coulomb_potential_ratio_oracle(*values)
            for values in zip(
                radii_i,
                radii_j,
                charges_i,
                charges_j,
                temperatures,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    result = wp.zeros(len(expected), dtype=wp.float64, device=device)
    wp.launch(
        _coulomb_potential_ratio_kernel,
        dim=len(expected),
        inputs=[
            _warp_array(radii_i, device),
            _warp_array(radii_j, device),
            _warp_array(charges_i, device),
            _warp_array(charges_j, device),
            _warp_array(temperatures, device),
            wp.float64(BOLTZMANN_CONSTANT),
            wp.float64(ELEMENTARY_CHARGE_VALUE),
            wp.float64(ELECTRIC_PERMITTIVITY),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    observed = result.numpy()
    assert observed.dtype == np.float64
    _assert_casewise_allclose(
        observed,
        expected,
        (
            "neutral",
            "opposite_sign_attraction",
            "same_sign_repulsion",
            "repulsion_lower_clipped",
            "zero_radius_sum",
            "negative_radius_sum",
            "negative_temperature",
            "zero_temperature",
        ),
        rtol=1e-12,
        atol=0.0,
    )
    assert observed[0] == 0.0
    assert observed[1] > 0.0
    assert observed[2] < 0.0
    assert observed[3] == -200.0
    assert observed[4] == 0.0
    assert observed[5] == 0.0
    assert observed[6] == 0.0
    assert observed[7] == 0.0


@pytest.mark.gpu_parity
def test_reduced_value_wp_matches_independent_oracle(device: str) -> None:
    """Validate equal and invalid-denominator reduced values."""
    left = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float64)
    right = np.array([2.0, -1.0, 0.0, -2.0], dtype=np.float64)
    expected = np.array(
        [
            _reduced_value_oracle(*values)
            for values in zip(left, right, strict=True)
        ],
        dtype=np.float64,
    )
    result = wp.zeros(len(expected), dtype=wp.float64, device=device)
    wp.launch(
        _reduced_value_kernel,
        dim=len(expected),
        inputs=[_warp_array(left, device), _warp_array(right, device)],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    observed = result.numpy()
    assert observed.dtype == np.float64
    _assert_casewise_allclose(
        observed,
        expected,
        (
            "equal_values",
            "zero_denominator",
            "zero_inputs",
            "negative_denominator",
        ),
        rtol=1e-12,
        atol=0.0,
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed >= 0.0)
    npt.assert_array_equal(observed[1:], np.zeros(3, dtype=np.float64))


@pytest.mark.gpu_parity
def test_coulomb_limit_helpers_wp_match_independent_oracle(device: str) -> None:
    """Validate neutral, attractive, and repulsive Coulomb enhancements."""
    potential = np.array([0.0, 2.0, -2.0, -200.0], dtype=np.float64)
    kinetic_expected = np.array(
        [_kinetic_limit_oracle(value) for value in potential], dtype=np.float64
    )
    continuum_expected = np.array(
        [_continuum_limit_oracle(value) for value in potential],
        dtype=np.float64,
    )
    kinetic = wp.zeros(len(potential), dtype=wp.float64, device=device)
    continuum = wp.zeros(len(potential), dtype=wp.float64, device=device)
    potential_wp = _warp_array(potential, device)
    wp.launch(
        _coulomb_kinetic_limit_kernel,
        dim=len(potential),
        inputs=[potential_wp],
        outputs=[kinetic],
        device=device,
    )
    wp.launch(
        _coulomb_continuum_limit_kernel,
        dim=len(potential),
        inputs=[potential_wp],
        outputs=[continuum],
        device=device,
    )
    wp.synchronize()
    kinetic_observed = kinetic.numpy()
    continuum_observed = continuum.numpy()
    assert kinetic_observed.dtype == np.float64
    assert continuum_observed.dtype == np.float64
    case_names = (
        "neutral",
        "attraction",
        "repulsion",
        "extreme_repulsion",
    )
    _assert_casewise_allclose(
        kinetic_observed,
        kinetic_expected,
        case_names,
        rtol=1e-12,
        atol=0.0,
    )
    _assert_casewise_allclose(
        continuum_observed,
        continuum_expected,
        case_names,
        rtol=1e-12,
        atol=1e-100,
    )
    assert kinetic_observed[0] == 1.0
    assert continuum_observed[0] == 1.0
    assert np.all(np.isfinite(kinetic_observed))
    assert np.all(np.isfinite(continuum_observed))
    assert np.all(kinetic_observed >= 0.0)
    assert np.all(continuum_observed >= 0.0)


@pytest.mark.gpu_parity
def test_diffusive_knudsen_number_wp_matches_independent_oracle(
    device: str,
) -> None:
    """Validate equal, mixed, safe, and kinetic-threshold pair branches."""
    radii_i = np.array(
        [1e-8, 1e-9, 0.0, -1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    )
    radii_j = np.array(
        [1e-8, 1e-6, 0.0, 0.0, 2e-8, 2e-8, 2e-8, 2e-8, 2e-8, 2e-8]
    )
    masses_i = np.array(
        [1e-20, 1e-24, 1e-20, 1e-20, 0.0, -1.0, 1e-20, 1e-20, 1e-20, 1e-20]
    )
    masses_j = np.array(
        [1e-20, 1e-18, 1e-20, 1e-20, 1e-20, -2.0, 1e-20, 1e-20, 1e-20, 1e-20]
    )
    frictions_i = np.array(
        [1e-12, 1e-14, 1e-12, 1e-12, 1e-12, 1e-12, 0.0, 1e-12, -1.0, 1e-12]
    )
    frictions_j = np.array(
        [1e-12, 1e-10, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, -2.0, 1e-12]
    )
    potentials = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -200.0, 0.0, 0.0])
    temperatures = np.array(
        [
            298.15,
            320.0,
            298.15,
            298.15,
            -1.0,
            298.15,
            298.15,
            298.15,
            298.15,
            0.0,
        ]
    )
    values = zip(
        radii_i,
        radii_j,
        masses_i,
        masses_j,
        frictions_i,
        frictions_j,
        potentials,
        temperatures,
        strict=True,
    )
    expected = np.array(
        [_diffusive_knudsen_number_oracle(*value) for value in values],
        dtype=np.float64,
    )
    result = wp.zeros(len(expected), dtype=wp.float64, device=device)
    wp.launch(
        _diffusive_knudsen_number_kernel,
        dim=len(expected),
        inputs=[
            _warp_array(radii_i, device),
            _warp_array(radii_j, device),
            _warp_array(masses_i, device),
            _warp_array(masses_j, device),
            _warp_array(frictions_i, device),
            _warp_array(frictions_j, device),
            _warp_array(potentials, device),
            _warp_array(temperatures, device),
            wp.float64(BOLTZMANN_CONSTANT),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    observed = result.numpy()
    assert observed.dtype == np.float64
    _assert_casewise_allclose(
        observed,
        expected,
        (
            "equal_particle_values",
            "mixed_radius_mass_friction_scales",
            "zero_radius_sum",
            "negative_radius_sum",
            "negative_temperature",
            "negative_reduced_mass",
            "zero_reduced_friction",
            "kinetic_threshold",
            "negative_reduced_friction",
            "zero_temperature",
        ),
        rtol=1e-12,
        atol=1e-100,
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed >= 0.0)
    assert observed[0] > 0.0
    assert observed[1] > 0.0
    npt.assert_array_equal(observed[2:], np.zeros(8, dtype=np.float64))


def test_brownian_diffusivity_wp_matches_numpy(device: str) -> None:
    """Ensure brownian_diffusivity_wp matches NumPy reference values."""
    temperatures = np.array([250.0, 298.15, 320.0], dtype=np.float64)
    mobilities = np.array([1.0e-8, 2.5e-8, 4.0e-8], dtype=np.float64)
    expected = np.array(
        [
            _brownian_diffusivity(temp, mobility)
            for temp, mobility in zip(temperatures, mobilities, strict=True)
        ],
        dtype=np.float64,
    )

    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    mobilities_wp = wp.array(mobilities, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(temperatures), dtype=wp.float64, device=device)

    wp.launch(
        _brownian_diffusivity_kernel,
        dim=len(temperatures),
        inputs=[
            temperatures_wp,
            mobilities_wp,
            wp.float64(BOLTZMANN_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-10, atol=1e-20)


def test_particle_mean_free_path_wp_matches_numpy(device: str) -> None:
    """Ensure particle_mean_free_path_wp matches NumPy reference values."""
    diffusivities = np.array([1.0e-9, 2.0e-9, 5.0e-9], dtype=np.float64)
    mean_speeds = np.array([100.0, 250.0, 400.0], dtype=np.float64)
    expected = np.array(
        [
            _mean_free_path_l(diffusivity, speed)
            for diffusivity, speed in zip(
                diffusivities, mean_speeds, strict=True
            )
        ],
        dtype=np.float64,
    )

    diffusivities_wp = wp.array(diffusivities, dtype=wp.float64, device=device)
    speeds_wp = wp.array(mean_speeds, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(diffusivities), dtype=wp.float64, device=device)

    wp.launch(
        _particle_mean_free_path_kernel,
        dim=len(diffusivities),
        inputs=[diffusivities_wp, speeds_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-10, atol=1e-18)


def test_g_collection_term_wp_matches_numpy(device: str) -> None:
    """Ensure g_collection_term_wp matches NumPy reference values."""
    mean_free_paths = np.array([1.0e-9, 2.0e-8, 5.0e-8], dtype=np.float64)
    particle_radii = np.array([1.0e-8, 1.0e-7, 1.0e-6], dtype=np.float64)
    expected = np.array(
        [
            _g_collection_term(mean_path, radius)
            for mean_path, radius in zip(
                mean_free_paths,
                particle_radii,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    mean_paths_wp = wp.array(mean_free_paths, dtype=wp.float64, device=device)
    particle_radii_wp = wp.array(
        particle_radii, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(mean_free_paths), dtype=wp.float64, device=device)

    wp.launch(
        _g_collection_term_kernel,
        dim=len(mean_free_paths),
        inputs=[mean_paths_wp, particle_radii_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-10, atol=1e-20)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_brownian_kernel_pair_wp_matches_numpy(
    device: str, alpha: float
) -> None:
    """Ensure brownian_kernel_pair_wp matches NumPy reference values."""
    radii_i = np.array([1.0e-8, 1.0e-7, 1.0e-6], dtype=np.float64)
    radii_j = np.array([2.0e-8, 2.0e-7, 2.0e-6], dtype=np.float64)
    diffusivities_i = np.array([1.0e-9, 2.0e-9, 3.0e-9], dtype=np.float64)
    diffusivities_j = np.array([1.5e-9, 2.5e-9, 3.5e-9], dtype=np.float64)
    g_terms_i = np.array([1.0e-9, 2.0e-8, 4.0e-8], dtype=np.float64)
    g_terms_j = np.array([1.2e-9, 2.2e-8, 4.2e-8], dtype=np.float64)
    speeds_i = np.array([80.0, 150.0, 250.0], dtype=np.float64)
    speeds_j = np.array([90.0, 175.0, 275.0], dtype=np.float64)
    expected = []
    for (
        radius_i,
        radius_j,
        diff_i,
        diff_j,
        g_i,
        g_j,
        speed_i,
        speed_j,
    ) in zip(
        radii_i,
        radii_j,
        diffusivities_i,
        diffusivities_j,
        g_terms_i,
        g_terms_j,
        speeds_i,
        speeds_j,
        strict=True,
    ):
        kernel_matrix = np.asarray(
            get_brownian_kernel(
                particle_radius=np.array(
                    [radius_i, radius_j], dtype=np.float64
                ),
                diffusivity_particle=np.array(
                    [diff_i, diff_j], dtype=np.float64
                ),
                g_collection_term_particle=np.array(
                    [g_i, g_j], dtype=np.float64
                ),
                mean_thermal_speed_particle=np.array(
                    [speed_i, speed_j], dtype=np.float64
                ),
                alpha_collision_efficiency=alpha,
            )
        )
        expected.append(kernel_matrix[0, 1])
    expected_array = np.array(expected, dtype=np.float64)

    radii_i_wp = wp.array(radii_i, dtype=wp.float64, device=device)
    radii_j_wp = wp.array(radii_j, dtype=wp.float64, device=device)
    diffusivities_i_wp = wp.array(
        diffusivities_i, dtype=wp.float64, device=device
    )
    diffusivities_j_wp = wp.array(
        diffusivities_j, dtype=wp.float64, device=device
    )
    g_terms_i_wp = wp.array(g_terms_i, dtype=wp.float64, device=device)
    g_terms_j_wp = wp.array(g_terms_j, dtype=wp.float64, device=device)
    speeds_i_wp = wp.array(speeds_i, dtype=wp.float64, device=device)
    speeds_j_wp = wp.array(speeds_j, dtype=wp.float64, device=device)
    alpha_wp = wp.array(
        np.full(len(radii_i), alpha, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    result_wp = wp.zeros(len(radii_i), dtype=wp.float64, device=device)

    wp.launch(
        _brownian_kernel_pair_kernel,
        dim=len(radii_i),
        inputs=[
            radii_i_wp,
            radii_j_wp,
            diffusivities_i_wp,
            diffusivities_j_wp,
            g_terms_i_wp,
            g_terms_j_wp,
            speeds_i_wp,
            speeds_j_wp,
            alpha_wp,
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(), expected_array, rtol=1e-10, atol=1e-20
    )


def test_brownian_kernel_chain_matches_numpy(device: str) -> None:
    """Ensure chained Brownian kernel calculation matches NumPy reference."""
    temperatures = np.array([298.15], dtype=np.float64)
    pressures = np.array([101325.0], dtype=np.float64)
    particle_radii_i = np.array([1.0e-8], dtype=np.float64)
    particle_radii_j = np.array([1.0e-7], dtype=np.float64)
    particle_masses_i = np.array([1.0e-21], dtype=np.float64)
    particle_masses_j = np.array([1.0e-18], dtype=np.float64)
    alpha_values = np.array([1.0], dtype=np.float64)

    expected_matrix = np.asarray(
        get_brownian_kernel_via_system_state(
            particle_radius=np.array(
                [particle_radii_i[0], particle_radii_j[0]]
            ),
            particle_mass=np.array(
                [particle_masses_i[0], particle_masses_j[0]]
            ),
            temperature=temperatures[0],
            pressure=pressures[0],
            alpha_collision_efficiency=alpha_values[0],
        )
    )
    expected_value = expected_matrix[0, 1]

    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    pressures_wp = wp.array(pressures, dtype=wp.float64, device=device)
    radii_i_wp = wp.array(particle_radii_i, dtype=wp.float64, device=device)
    radii_j_wp = wp.array(particle_radii_j, dtype=wp.float64, device=device)
    masses_i_wp = wp.array(particle_masses_i, dtype=wp.float64, device=device)
    masses_j_wp = wp.array(particle_masses_j, dtype=wp.float64, device=device)
    alpha_wp = wp.array(alpha_values, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(alpha_values), dtype=wp.float64, device=device)

    wp.launch(
        _brownian_chain_kernel,
        dim=len(alpha_values),
        inputs=[
            temperatures_wp,
            pressures_wp,
            radii_i_wp,
            radii_j_wp,
            masses_i_wp,
            masses_j_wp,
            alpha_wp,
            wp.float64(BOLTZMANN_CONSTANT),
            wp.float64(GAS_CONSTANT),
            wp.float64(MOLECULAR_WEIGHT_AIR),
            wp.float64(REF_VISCOSITY_AIR_STP),
            wp.float64(REF_TEMPERATURE_STP),
            wp.float64(SUTHERLAND_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy()[0], expected_value, rtol=1e-10, atol=1e-20
    )
