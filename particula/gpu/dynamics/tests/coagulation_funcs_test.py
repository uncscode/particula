"""Parity tests for GPU coagulation composite functions."""

import ast
from pathlib import Path
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
        charged_hard_sphere_wp,
        coulomb_continuum_limit_wp,
        coulomb_kinetic_limit_wp,
        coulomb_potential_ratio_wp,
        diffusive_knudsen_number_wp,
        effective_density_wp,
        g_collection_term_wp,
        particle_mean_free_path_wp,
        reduced_value_wp,
        sedimentation_sp2016_pair_rate_wp,
        settling_velocity_stokes_wp,
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
        STANDARD_GRAVITY,
        SUTHERLAND_CONSTANT,
    )


def _warp_kernel(function):
    """Decorate kernels only when Warp is available."""
    if wp is None:
        return function
    return wp.kernel(function)


def _available_warp_devices() -> list[Any]:
    """Return collection-safe Warp device params."""
    if wp is None:
        return ["cpu"]
    return [
        pytest.param(device, marks=pytest.mark.cuda)
        if device == "cuda"
        else device
        for device in warp_devices(wp)
    ]


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


@_warp_kernel
def _charged_hard_sphere_kernel(
    radii_i: Any,
    radii_j: Any,
    masses_i: Any,
    masses_j: Any,
    charges_i: Any,
    charges_j: Any,
    temperatures: Any,
    pressures: Any,
    boltzmann_constant: Any,
    elementary_charge_value: Any,
    electric_permittivity: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    result: Any,
) -> None:
    """Compute charged hard-sphere pair rates for a scalar probe batch."""
    tid = wp.tid()
    result[tid] = charged_hard_sphere_wp(
        radii_i[tid],
        radii_j[tid],
        masses_i[tid],
        masses_j[tid],
        charges_i[tid],
        charges_j[tid],
        temperatures[tid],
        pressures[tid],
        boltzmann_constant,
        elementary_charge_value,
        electric_permittivity,
        gas_constant,
        molecular_weight_air,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )


@_warp_kernel
def _effective_density_kernel(
    total_masses: Any,
    total_volumes: Any,
    result: Any,
) -> None:
    """Compute effective densities for lane-indexed scalar totals."""
    tid = wp.tid()
    result[tid] = effective_density_wp(total_masses[tid], total_volumes[tid])


@_warp_kernel
def _settling_velocity_stokes_kernel(
    radii: Any,
    densities: Any,
    temperatures: Any,
    pressures: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    result: Any,
) -> None:
    """Compute Stokes/Cunningham settling velocities for scalar lanes."""
    tid = wp.tid()
    result[tid] = settling_velocity_stokes_wp(
        radii[tid],
        densities[tid],
        temperatures[tid],
        pressures[tid],
        gas_constant,
        molecular_weight_air,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )


@_warp_kernel
def _sedimentation_sp2016_pair_rate_kernel(
    radii_i: Any,
    radii_j: Any,
    velocities_i: Any,
    velocities_j: Any,
    result: Any,
) -> None:
    """Compute SP2016 unit-efficiency pair rates for scalar lanes."""
    tid = wp.tid()
    result[tid] = sedimentation_sp2016_pair_rate_wp(
        radii_i[tid],
        radii_j[tid],
        velocities_i[tid],
        velocities_j[tid],
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


def _charged_hard_sphere_oracle(  # noqa: C901
    radius_i: float,
    radius_j: float,
    mass_i: float,
    mass_j: float,
    charge_i: float,
    charge_j: float,
    temperature: float,
    pressure: float,
    boltzmann_constant: float = BOLTZMANN_CONSTANT,
    elementary_charge_value: float = ELEMENTARY_CHARGE_VALUE,
    electric_permittivity: float = ELECTRIC_PERMITTIVITY,
    gas_constant: float = GAS_CONSTANT,
    molecular_weight_air: float = MOLECULAR_WEIGHT_AIR,
    ref_viscosity: float = REF_VISCOSITY_AIR_STP,
    ref_temperature: float = REF_TEMPERATURE_STP,
    sutherland_constant: float = SUTHERLAND_CONSTANT,
) -> float:
    """Independently compute the charged hard-sphere safe-zero contract."""
    positive_values = (
        radius_i,
        radius_j,
        mass_i,
        mass_j,
        temperature,
        pressure,
        boltzmann_constant,
        elementary_charge_value,
        electric_permittivity,
        gas_constant,
        molecular_weight_air,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    if (
        not all(np.isfinite(value) and value > 0.0 for value in positive_values)
        or not np.isfinite(charge_i)
        or not np.isfinite(charge_j)
    ):
        return 0.0
    with np.errstate(all="ignore"):
        viscosity = (
            ref_viscosity
            * (temperature / ref_temperature) ** 1.5
            * (ref_temperature + sutherland_constant)
            / (temperature + sutherland_constant)
        )
        mean_free_path = (2.0 * viscosity / pressure) / np.sqrt(
            8.0 * molecular_weight_air / (np.pi * gas_constant * temperature)
        )
        if not np.isfinite(viscosity) or viscosity <= 0.0:
            return 0.0
        if not np.isfinite(mean_free_path) or mean_free_path <= 0.0:
            return 0.0
        knudsen_i = mean_free_path / radius_i
        knudsen_j = mean_free_path / radius_j
        if (
            not np.isfinite(knudsen_i)
            or knudsen_i <= 0.0
            or not np.isfinite(knudsen_j)
            or knudsen_j <= 0.0
        ):
            return 0.0
        slip_i = 1.0 + knudsen_i * (1.257 + 0.4 * np.exp(-1.1 / knudsen_i))
        slip_j = 1.0 + knudsen_j * (1.257 + 0.4 * np.exp(-1.1 / knudsen_j))
        friction_i = 6.0 * np.pi * viscosity * radius_i / slip_i
        friction_j = 6.0 * np.pi * viscosity * radius_j / slip_j
        if (
            not np.isfinite(slip_i)
            or slip_i <= 0.0
            or not np.isfinite(slip_j)
            or slip_j <= 0.0
            or not np.isfinite(friction_i)
            or friction_i <= 0.0
            or not np.isfinite(friction_j)
            or friction_j <= 0.0
        ):
            return 0.0
        sum_radius = radius_i + radius_j
        potential = -(
            charge_i
            * charge_j
            * elementary_charge_value**2
            / (
                4.0
                * np.pi
                * electric_permittivity
                * sum_radius
                * boltzmann_constant
                * temperature
            )
        )
        potential = max(potential, -200.0)
        if not np.isfinite(potential):
            return 0.0
        reduced_mass = mass_i * mass_j / (mass_i + mass_j)
        reduced_friction = friction_i * friction_j / (friction_i + friction_j)
        if (
            not np.isfinite(reduced_mass)
            or reduced_mass <= 0.0
            or not np.isfinite(reduced_friction)
            or reduced_friction <= 0.0
        ):
            return 0.0
        kinetic = 1.0 + potential if potential >= 0.0 else np.exp(potential)
        if not np.isfinite(kinetic) or kinetic < 1.0e-80:
            return 0.0
        continuum = (
            1.0 if potential == 0.0 else potential / -np.expm1(-potential)
        )
        if not np.isfinite(continuum) or continuum <= 0.0:
            return 0.0
        diffusive_knudsen = (
            np.sqrt(boltzmann_constant * temperature * reduced_mass)
            / reduced_friction
            / (sum_radius * continuum / kinetic)
        )
        if not np.isfinite(diffusive_knudsen) or diffusive_knudsen < 0.0:
            return 0.0
        numerator = (
            4.0 * np.pi * diffusive_knudsen**2
            + 25.836 * diffusive_knudsen**3
            + np.sqrt(8.0 * np.pi) * 11.211 * diffusive_knudsen**4
        )
        denominator = (
            1.0
            + 3.502 * diffusive_knudsen
            + 7.211 * diffusive_knudsen**2
            + 11.211 * diffusive_knudsen**3
        )
        dimensionless = numerator / denominator
        result = (
            dimensionless
            * reduced_friction
            * sum_radius**3
            * kinetic**2
            / (reduced_mass * continuum)
        )
    if not np.isfinite(result) or result <= 0.0:
        return 0.0
    return float(result)


def _effective_density_oracle(
    masses: np.ndarray,
    species_densities: np.ndarray,
) -> float:
    """Independently calculate mixture density with the safe-zero contract."""
    with np.errstate(all="ignore"):
        total_mass = np.sum(masses)
        total_volume = np.sum(masses / species_densities)
        density = total_mass / total_volume
    if (
        not np.isfinite(total_mass)
        or total_mass <= 0.0
        or not np.isfinite(total_volume)
        or total_volume <= 0.0
        or not np.isfinite(density)
        or density <= 0.0
    ):
        return 0.0
    return float(density)


def _settling_velocity_stokes_oracle(
    radius: float,
    density: float,
    temperature: float,
    pressure: float,
    gas_constant: float = GAS_CONSTANT,
    molecular_weight_air: float = MOLECULAR_WEIGHT_AIR,
    ref_viscosity: float = REF_VISCOSITY_AIR_STP,
    ref_temperature: float = REF_TEMPERATURE_STP,
    sutherland_constant: float = SUTHERLAND_CONSTANT,
) -> float:
    """Independently model Stokes/Cunningham settling safe-zero behavior."""
    values = (
        radius,
        density,
        temperature,
        pressure,
        gas_constant,
        molecular_weight_air,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        return 0.0
    with np.errstate(all="ignore"):
        viscosity = (
            ref_viscosity
            * (temperature / ref_temperature) ** 1.5
            * (ref_temperature + sutherland_constant)
            / (temperature + sutherland_constant)
        )
        mean_free_path = (2.0 * viscosity / pressure) / np.sqrt(
            8.0 * molecular_weight_air / (np.pi * gas_constant * temperature)
        )
        knudsen = mean_free_path / radius
        slip = 1.0 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))
        result = (
            2.0
            * radius**2
            * density
            * slip
            * STANDARD_GRAVITY
            / (9.0 * viscosity)
        )
    if (
        not np.isfinite(viscosity)
        or viscosity <= 0.0
        or not np.isfinite(mean_free_path)
        or mean_free_path <= 0.0
        or not np.isfinite(knudsen)
        or knudsen <= 0.0
        or not np.isfinite(slip)
        or slip <= 0.0
        or not np.isfinite(result)
        or result <= 0.0
    ):
        return 0.0
    return float(result)


def _sedimentation_sp2016_pair_rate_oracle(
    radius_i: float,
    radius_j: float,
    velocity_i: float,
    velocity_j: float,
) -> float:
    """Independently model the SP2016 Eq. 13A.4 safe-zero contract."""
    values = (radius_i, radius_j, velocity_i, velocity_j)
    if (
        not all(np.isfinite(value) for value in values)
        or radius_i <= 0.0
        or radius_j <= 0.0
        or velocity_i < 0.0
        or velocity_j < 0.0
    ):
        return 0.0
    with np.errstate(all="ignore"):
        rate = np.pi * (radius_i + radius_j) ** 2 * abs(velocity_i - velocity_j)
    if not np.isfinite(rate) or rate <= 0.0:
        return 0.0
    return float(rate)


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


def _launch_charged_hard_sphere(
    device: str,
    radii_i: np.ndarray,
    radii_j: np.ndarray,
    masses_i: np.ndarray,
    masses_j: np.ndarray,
    charges_i: np.ndarray,
    charges_j: np.ndarray,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    constants: tuple[float, ...],
) -> np.ndarray:
    """Launch the charged scalar probe and return its fp64 host result."""
    result = wp.zeros(len(radii_i), dtype=wp.float64, device=device)
    wp.launch(
        _charged_hard_sphere_kernel,
        dim=len(radii_i),
        inputs=[
            _warp_array(radii_i, device),
            _warp_array(radii_j, device),
            _warp_array(masses_i, device),
            _warp_array(masses_j, device),
            _warp_array(charges_i, device),
            _warp_array(charges_j, device),
            _warp_array(temperatures, device),
            _warp_array(pressures, device),
            *(wp.float64(value) for value in constants),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


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
    potential = np.array(
        [0.0, 1.0e-8, -1.0e-8, 2.0, -2.0, -200.0],
        dtype=np.float64,
    )
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
        "near_neutral_attraction",
        "near_neutral_repulsion",
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
    assert continuum_observed[1] > 1.0
    assert continuum_observed[2] < 1.0


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


@pytest.mark.gpu_parity
@pytest.mark.warp
def test_charged_hard_sphere_wp_matches_independent_oracle_and_symmetry(
    device: str,
) -> None:
    """Validate charged fp64 parity, neutral behavior, and pair symmetry."""
    radii_i = np.array([1e-9, 8e-9, 3e-8, 2e-7, 1e-8], dtype=np.float64)
    radii_j = np.array([2e-9, 2e-8, 8e-9, 8e-7, 2e-8], dtype=np.float64)
    masses_i = np.array([1e-24, 3e-21, 2e-19, 8e-15, 1e-20], dtype=np.float64)
    masses_j = np.array([8e-24, 2e-20, 7e-21, 3e-13, 2e-20], dtype=np.float64)
    charges_i = np.array([0.0, 1.0, 2.0, -3.0, 100.0], dtype=np.float64)
    charges_j = np.array([0.0, 1.0, -1.0, 4.0, 100.0], dtype=np.float64)
    temperatures = np.array(
        [250.0, 298.15, 320.0, 285.0, 298.15], dtype=np.float64
    )
    pressures = np.array(
        [80000.0, 101325.0, 95000.0, 110000.0, 101325.0], dtype=np.float64
    )
    constants: tuple[float, float, float, float, float, float, float, float] = (
        np.float64(BOLTZMANN_CONSTANT),
        np.float64(ELEMENTARY_CHARGE_VALUE),
        np.float64(ELECTRIC_PERMITTIVITY),
        np.float64(GAS_CONSTANT),
        np.float64(MOLECULAR_WEIGHT_AIR),
        np.float64(REF_VISCOSITY_AIR_STP),
        np.float64(REF_TEMPERATURE_STP),
        np.float64(SUTHERLAND_CONSTANT),
    )
    expected = np.array(
        [
            _charged_hard_sphere_oracle(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                boltzmann_constant=constants[0],
                elementary_charge_value=constants[1],
                electric_permittivity=constants[2],
                gas_constant=constants[3],
                molecular_weight_air=constants[4],
                ref_viscosity=constants[5],
                ref_temperature=constants[6],
                sutherland_constant=constants[7],
            )
            for values in zip(
                radii_i,
                radii_j,
                masses_i,
                masses_j,
                charges_i,
                charges_j,
                temperatures,
                pressures,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    observed = _launch_charged_hard_sphere(
        device,
        radii_i,
        radii_j,
        masses_i,
        masses_j,
        charges_i,
        charges_j,
        temperatures,
        pressures,
        constants,
    )
    assert observed.dtype == np.float64
    _assert_casewise_allclose(
        observed,
        expected,
        (
            "neutral_nanometer",
            "same_sign_repulsion",
            "opposite_sign_attraction",
            "mixed_scale_attraction",
            "repulsion_floor",
        ),
        rtol=1e-6,
        atol=0.0,
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed >= 0.0)
    assert observed[4] == 0.0
    assert expected[4] == 0.0
    assert observed[0] > 0.0

    swapped = _launch_charged_hard_sphere(
        device,
        radii_j,
        radii_i,
        masses_j,
        masses_i,
        charges_j,
        charges_i,
        temperatures,
        pressures,
        constants,
    )
    npt.assert_allclose(swapped, observed, rtol=1e-6, atol=0.0)


@pytest.mark.gpu_parity
@pytest.mark.warp
def test_charged_hard_sphere_wp_handles_extreme_charge_lanes(
    device: str,
) -> None:
    """Bound extreme finite charge lanes without losing attraction rates."""
    radii_i = np.full(3, 1e-8, dtype=np.float64)
    radii_j = np.full(3, 2e-8, dtype=np.float64)
    masses_i = np.full(3, 1e-20, dtype=np.float64)
    masses_j = np.full(3, 2e-20, dtype=np.float64)
    charges_i = np.array([1.0, 1e200, -1e200], dtype=np.float64)
    charges_j = np.array([-1.0, 1e200, 1e200], dtype=np.float64)
    temperatures = np.full(3, 298.15, dtype=np.float64)
    pressures = np.full(3, 101325.0, dtype=np.float64)
    constants: tuple[float, float, float, float, float, float, float, float] = (
        np.float64(BOLTZMANN_CONSTANT),
        np.float64(ELEMENTARY_CHARGE_VALUE),
        np.float64(ELECTRIC_PERMITTIVITY),
        np.float64(GAS_CONSTANT),
        np.float64(MOLECULAR_WEIGHT_AIR),
        np.float64(REF_VISCOSITY_AIR_STP),
        np.float64(REF_TEMPERATURE_STP),
        np.float64(SUTHERLAND_CONSTANT),
    )
    expected = np.array(
        [
            _charged_hard_sphere_oracle(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                boltzmann_constant=constants[0],
                elementary_charge_value=constants[1],
                electric_permittivity=constants[2],
                gas_constant=constants[3],
                molecular_weight_air=constants[4],
                ref_viscosity=constants[5],
                ref_temperature=constants[6],
                sutherland_constant=constants[7],
            )
            for values in zip(
                radii_i,
                radii_j,
                masses_i,
                masses_j,
                charges_i,
                charges_j,
                temperatures,
                pressures,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    observed = _launch_charged_hard_sphere(
        device,
        radii_i,
        radii_j,
        masses_i,
        masses_j,
        charges_i,
        charges_j,
        temperatures,
        pressures,
        constants,
    )
    npt.assert_allclose(observed[0], expected[0], rtol=1e-12, atol=0.0)
    assert observed[0] > 0.0
    assert observed[1] == 0.0
    assert np.isfinite(observed[2])
    assert observed[2] > 0.0


_SAFE_ZERO_CASES = (
    *(
        (field, value)
        for field in (
            "radius_i",
            "radius_j",
            "mass_i",
            "mass_j",
            "temperature",
            "pressure",
        )
        for value in (0.0, -1.0)
    ),
    *(
        (field, value)
        for field in (
            "radius_i",
            "radius_j",
            "mass_i",
            "mass_j",
            "charge_i",
            "charge_j",
            "temperature",
            "pressure",
        )
        for value in (np.nan, np.inf, -np.inf)
    ),
    *(
        (field, value)
        for field in (
            "boltzmann_constant",
            "elementary_charge_value",
            "electric_permittivity",
            "gas_constant",
            "molecular_weight_air",
            "ref_viscosity",
            "ref_temperature",
            "sutherland_constant",
        )
        for value in (np.nan, np.inf, -np.inf, 0.0, -1.0)
    ),
)


@pytest.mark.gpu_parity
@pytest.mark.warp
@pytest.mark.parametrize(("invalid_field", "invalid_value"), _SAFE_ZERO_CASES)
def test_charged_hard_sphere_wp_returns_safe_zero_for_invalid_inputs(
    device: str,
    invalid_field: str,
    invalid_value: float,
) -> None:
    """Verify every documented invalid scalar domain returns exact zero."""
    state = {
        "radius_i": np.float64(1e-8),
        "radius_j": np.float64(2e-8),
        "mass_i": np.float64(1e-20),
        "mass_j": np.float64(2e-20),
        "charge_i": np.float64(1.0),
        "charge_j": np.float64(-1.0),
        "temperature": np.float64(298.15),
        "pressure": np.float64(101325.0),
    }
    constants = {
        "boltzmann_constant": np.float64(BOLTZMANN_CONSTANT),
        "elementary_charge_value": np.float64(ELEMENTARY_CHARGE_VALUE),
        "electric_permittivity": np.float64(ELECTRIC_PERMITTIVITY),
        "gas_constant": np.float64(GAS_CONSTANT),
        "molecular_weight_air": np.float64(MOLECULAR_WEIGHT_AIR),
        "ref_viscosity": np.float64(REF_VISCOSITY_AIR_STP),
        "ref_temperature": np.float64(REF_TEMPERATURE_STP),
        "sutherland_constant": np.float64(SUTHERLAND_CONSTANT),
    }
    if invalid_field in state:
        state[invalid_field] = np.float64(invalid_value)
    else:
        constants[invalid_field] = np.float64(invalid_value)
    constant_values: tuple[
        float, float, float, float, float, float, float, float
    ] = (
        constants["boltzmann_constant"],
        constants["elementary_charge_value"],
        constants["electric_permittivity"],
        constants["gas_constant"],
        constants["molecular_weight_air"],
        constants["ref_viscosity"],
        constants["ref_temperature"],
        constants["sutherland_constant"],
    )
    observed = _launch_charged_hard_sphere(
        device,
        np.array([state["radius_i"]], dtype=np.float64),
        np.array([state["radius_j"]], dtype=np.float64),
        np.array([state["mass_i"]], dtype=np.float64),
        np.array([state["mass_j"]], dtype=np.float64),
        np.array([state["charge_i"]], dtype=np.float64),
        np.array([state["charge_j"]], dtype=np.float64),
        np.array([state["temperature"]], dtype=np.float64),
        np.array([state["pressure"]], dtype=np.float64),
        constant_values,
    )
    expected = _charged_hard_sphere_oracle(
        state["radius_i"],
        state["radius_j"],
        state["mass_i"],
        state["mass_j"],
        state["charge_i"],
        state["charge_j"],
        state["temperature"],
        state["pressure"],
        boltzmann_constant=constant_values[0],
        elementary_charge_value=constant_values[1],
        electric_permittivity=constant_values[2],
        gas_constant=constant_values[3],
        molecular_weight_air=constant_values[4],
        ref_viscosity=constant_values[5],
        ref_temperature=constant_values[6],
        sutherland_constant=constant_values[7],
    )
    assert observed[0] == 0.0
    assert expected == 0.0


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


def _launch_effective_density(
    device: str,
    total_masses: np.ndarray,
    total_volumes: np.ndarray,
) -> np.ndarray:
    """Launch a lane-indexed effective-density probe."""
    result = wp.zeros(len(total_masses), dtype=wp.float64, device=device)
    wp.launch(
        _effective_density_kernel,
        dim=len(total_masses),
        inputs=[
            _warp_array(total_masses, device),
            _warp_array(total_volumes, device),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


def _launch_settling_velocity_stokes(
    device: str,
    radii: np.ndarray,
    densities: np.ndarray,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    constants: tuple[float, float, float, float, float],
) -> np.ndarray:
    """Launch a lane-indexed Stokes/Cunningham settling probe."""
    result = wp.zeros(len(radii), dtype=wp.float64, device=device)
    wp.launch(
        _settling_velocity_stokes_kernel,
        dim=len(radii),
        inputs=[
            _warp_array(radii, device),
            _warp_array(densities, device),
            _warp_array(temperatures, device),
            _warp_array(pressures, device),
            *(wp.float64(value) for value in constants),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


def _launch_sedimentation_pair_rate(
    device: str,
    radii_i: np.ndarray,
    radii_j: np.ndarray,
    velocities_i: np.ndarray,
    velocities_j: np.ndarray,
) -> np.ndarray:
    """Launch a lane-indexed SP2016 sedimentation-rate probe."""
    result = wp.zeros(len(radii_i), dtype=wp.float64, device=device)
    wp.launch(
        _sedimentation_sp2016_pair_rate_kernel,
        dim=len(radii_i),
        inputs=[
            _warp_array(radii_i, device),
            _warp_array(radii_j, device),
            _warp_array(velocities_i, device),
            _warp_array(velocities_j, device),
        ],
        outputs=[result],
        device=device,
    )
    wp.synchronize()
    return result.numpy()


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_effective_density_wp_matches_mixture_oracle(device: str) -> None:
    """Match one- and multi-species mixture-density calculations."""
    compositions = (
        (
            np.array([2.0e-18], dtype=np.float64),
            np.array([1000.0], dtype=np.float64),
        ),
        (
            np.array([1.0e-18, 3.0e-18], dtype=np.float64),
            np.array([800.0, 1600.0], dtype=np.float64),
        ),
        (
            np.array([4.0e-19, 2.0e-18, 1.0e-18], dtype=np.float64),
            np.array([900.0, 1200.0, 1800.0], dtype=np.float64),
        ),
    )
    total_masses = np.array(
        [np.sum(masses) for masses, _ in compositions], dtype=np.float64
    )
    total_volumes = np.array(
        [np.sum(masses / densities) for masses, densities in compositions],
        dtype=np.float64,
    )
    expected = np.array(
        [
            _effective_density_oracle(masses, densities)
            for masses, densities in compositions
        ],
        dtype=np.float64,
    )
    observed = _launch_effective_density(device, total_masses, total_volumes)
    # fp64 operation ordering differs between NumPy and Warp.
    npt.assert_allclose(observed, expected, rtol=1e-12, atol=0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_effective_density_wp_invalid_or_overflow_returns_exact_zero(
    device: str,
) -> None:
    """Return finite exact zero for invalid or extreme density totals."""
    total_masses = np.array(
        [
            0.0,
            -1.0,
            np.nan,
            np.inf,
            -np.inf,
            np.finfo(float).max,
            np.nextafter(0.0, 1.0),
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    total_volumes = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            np.nextafter(0.0, 1.0),
            np.finfo(float).max,
            0.0,
            -1.0,
            np.nan,
            np.inf,
            -np.inf,
        ],
        dtype=np.float64,
    )
    observed = _launch_effective_density(device, total_masses, total_volumes)
    assert np.all(np.isfinite(observed))
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_effective_density_wp_subnormal_result_returns_exact_zero(
    device: str,
) -> None:
    """Return exact zero when a positive density result is subnormal."""
    observed = _launch_effective_density(
        device,
        np.array([1.0e-320], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_settling_velocity_stokes_wp_matches_numpy_oracle(device: str) -> None:
    """Match independent Stokes/Cunningham values from nano to droplet scale."""
    radii = np.array([2.0e-9, 5.0e-8, 1.0e-6, 5.0e-5], dtype=np.float64)
    densities = np.array([800.0, 1200.0, 1600.0, 1000.0], dtype=np.float64)
    temperatures = np.array([260.0, 298.15, 320.0, 285.0], dtype=np.float64)
    pressures = np.array(
        [80000.0, 101325.0, 95000.0, 110000.0], dtype=np.float64
    )
    constants = (
        GAS_CONSTANT,
        MOLECULAR_WEIGHT_AIR,
        REF_VISCOSITY_AIR_STP,
        REF_TEMPERATURE_STP,
        SUTHERLAND_CONSTANT,
    )
    expected = np.array(
        [
            _settling_velocity_stokes_oracle(
                radius, density, temperature, pressure
            )
            for radius, density, temperature, pressure in zip(
                radii, densities, temperatures, pressures, strict=True
            )
        ],
        dtype=np.float64,
    )
    observed = _launch_settling_velocity_stokes(
        device, radii, densities, temperatures, pressures, constants
    )
    npt.assert_allclose(observed, expected, rtol=1e-12, atol=0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_settling_velocity_stokes_wp_invalid_overflow_or_underflow_returns_exact_zero(
    device: str,
) -> None:
    """Return finite exact zero for every invalid Stokes input category."""
    base = (1.0e-6, 1000.0, 298.15, 101325.0)
    constants = (
        GAS_CONSTANT,
        MOLECULAR_WEIGHT_AIR,
        REF_VISCOSITY_AIR_STP,
        REF_TEMPERATURE_STP,
        SUTHERLAND_CONSTANT,
    )
    cases = []
    for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
        for index in range(4):
            case = list(base)
            case[index] = value
            cases.append((*case, *constants))
    cases.extend(
        [
            (1.0e200, *base[1:], *constants),
            (base[0], np.nextafter(0.0, 1.0), *base[2:], *constants),
        ]
    )
    values = np.asarray(cases, dtype=np.float64)
    observed = _launch_settling_velocity_stokes(
        device,
        values[:, 0],
        values[:, 1],
        values[:, 2],
        values[:, 3],
        constants,
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("invalid_constant", "constant_name"),
    [
        (0.0, "gas_constant"),
        (-1.0, "gas_constant"),
        (np.nan, "gas_constant"),
        (np.inf, "gas_constant"),
        (-np.inf, "gas_constant"),
        (0.0, "molecular_weight_air"),
        (-1.0, "molecular_weight_air"),
        (np.nan, "molecular_weight_air"),
        (np.inf, "molecular_weight_air"),
        (-np.inf, "molecular_weight_air"),
        (0.0, "ref_viscosity"),
        (-1.0, "ref_viscosity"),
        (np.nan, "ref_viscosity"),
        (np.inf, "ref_viscosity"),
        (-np.inf, "ref_viscosity"),
        (0.0, "ref_temperature"),
        (-1.0, "ref_temperature"),
        (np.nan, "ref_temperature"),
        (np.inf, "ref_temperature"),
        (-np.inf, "ref_temperature"),
        (0.0, "sutherland_constant"),
        (-1.0, "sutherland_constant"),
        (np.nan, "sutherland_constant"),
        (np.inf, "sutherland_constant"),
        (-np.inf, "sutherland_constant"),
    ],
)
def test_settling_velocity_stokes_wp_invalid_constants_return_exact_zero(
    device: str,
    invalid_constant: float,
    constant_name: str,
) -> None:
    """Return exact zero when any scalar gas-property constant is invalid."""
    constants = {
        "gas_constant": GAS_CONSTANT,
        "molecular_weight_air": MOLECULAR_WEIGHT_AIR,
        "ref_viscosity": REF_VISCOSITY_AIR_STP,
        "ref_temperature": REF_TEMPERATURE_STP,
        "sutherland_constant": SUTHERLAND_CONSTANT,
    }
    constants[constant_name] = invalid_constant
    observed = _launch_settling_velocity_stokes(
        device,
        np.array([1.0e-6], dtype=np.float64),
        np.array([1000.0], dtype=np.float64),
        np.array([298.15], dtype=np.float64),
        np.array([101325.0], dtype=np.float64),
        (
            constants["gas_constant"],
            constants["molecular_weight_air"],
            constants["ref_viscosity"],
            constants["ref_temperature"],
            constants["sutherland_constant"],
        ),
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_settling_velocity_stokes_wp_subnormal_result_returns_exact_zero(
    device: str,
) -> None:
    """Return exact zero when a positive Stokes result is subnormal."""
    constants = np.array(
        [
            GAS_CONSTANT,
            MOLECULAR_WEIGHT_AIR,
            REF_VISCOSITY_AIR_STP,
            REF_TEMPERATURE_STP,
            SUTHERLAND_CONSTANT,
        ],
        dtype=np.float64,
    )
    observed = _launch_settling_velocity_stokes(
        device,
        np.array([1.0e-6], dtype=np.float64),
        np.array([1.0e-310], dtype=np.float64),
        np.array([298.15], dtype=np.float64),
        np.array([101325.0], dtype=np.float64),
        tuple(constants.tolist()),
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_sedimentation_sp2016_pair_rate_wp_matches_oracle_and_is_symmetric(
    device: str,
) -> None:
    """Match SP2016 pair physics and its reversal symmetry."""
    radii_i = np.array([1.0e-8, 1.0e-6, 5.0e-5], dtype=np.float64)
    radii_j = np.array([2.0e-8, 3.0e-6, 1.0e-5], dtype=np.float64)
    velocities_i = np.array([1.0e-6, 2.0e-4, 1.0e-2], dtype=np.float64)
    velocities_j = np.array([3.0e-6, 5.0e-5, 1.0e-3], dtype=np.float64)
    expected = np.array(
        [
            _sedimentation_sp2016_pair_rate_oracle(*values)
            for values in zip(
                radii_i, radii_j, velocities_i, velocities_j, strict=True
            )
        ],
        dtype=np.float64,
    )
    observed = _launch_sedimentation_pair_rate(
        device, radii_i, radii_j, velocities_i, velocities_j
    )
    reversed_observed = _launch_sedimentation_pair_rate(
        device, radii_j, radii_i, velocities_j, velocities_i
    )
    npt.assert_allclose(observed, expected, rtol=1e-12, atol=0.0)
    npt.assert_allclose(reversed_observed, expected, rtol=1e-12, atol=0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_sedimentation_sp2016_pair_rate_wp_equal_velocity_returns_exact_zero(
    device: str,
) -> None:
    """Return exact zero for valid pairs with equal settling velocities."""
    observed = _launch_sedimentation_pair_rate(
        device,
        np.array([1.0e-8, 1.0e-6], dtype=np.float64),
        np.array([2.0e-8, 3.0e-6], dtype=np.float64),
        np.array([1.0e-5, 0.0], dtype=np.float64),
        np.array([1.0e-5, 0.0], dtype=np.float64),
    )
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_sedimentation_sp2016_pair_rate_wp_subnormal_result_returns_exact_zero(
    device: str,
) -> None:
    """Return exact zero when a positive SP2016 result is subnormal."""
    observed = _launch_sedimentation_pair_rate(
        device,
        np.array([1.0e-154], dtype=np.float64),
        np.array([1.0e-154], dtype=np.float64),
        np.array([1.0e-10], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_sedimentation_sp2016_pair_rate_wp_invalid_overflow_or_underflow_returns_exact_zero(
    device: str,
) -> None:
    """Return finite exact zero for invalid and extreme SP2016 lanes."""
    invalid = (0.0, -1.0, np.nan, np.inf, -np.inf)
    cases = []
    for value in invalid:
        cases.extend(
            [(value, 1.0e-6, 1.0e-4, 2.0e-4), (1.0e-6, value, 1.0e-4, 2.0e-4)]
        )
        paired_velocity = 0.0 if value == 0.0 else 2.0e-4
        cases.extend(
            [
                (1.0e-6, 2.0e-6, value, paired_velocity),
                (1.0e-6, 2.0e-6, 0.0 if value == 0.0 else 1.0e-4, value),
            ]
        )
    cases.extend(
        [
            (np.finfo(float).max, np.finfo(float).max, 1.0e-4, 2.0e-4),
            (1.0e154, 1.0e154, np.finfo(float).max, 0.0),
            (1.0e-200, 1.0e-200, 1.0, 0.0),
        ]
    )
    values = np.asarray(cases, dtype=np.float64)
    observed = _launch_sedimentation_pair_rate(
        device, values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    )
    assert np.all(np.isfinite(observed))
    assert np.all(observed == 0.0)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_sedimentation_sp2016_pair_rate_wp_has_no_efficiency_argument() -> None:
    """Keep the internal SP2016 primitive limited to its four physics inputs."""
    source_path = Path(__file__).parents[1] / "coagulation_funcs.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "sedimentation_sp2016_pair_rate_wp"
    )
    assert tuple(argument.arg for argument in function.args.posonlyargs) == ()
    assert tuple(argument.arg for argument in function.args.args) == (
        "radius_i",
        "radius_j",
        "velocity_i",
        "velocity_j",
    )
    assert function.args.vararg is None
    assert function.args.kwarg is None
    assert function.args.kwonlyargs == []
