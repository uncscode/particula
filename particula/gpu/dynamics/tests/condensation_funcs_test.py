"""Parity tests for GPU condensation composite functions."""

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
    from particula.dynamics.condensation.mass_transfer import (  # noqa: E402
        get_first_order_mass_transport_k,
        get_mass_transfer_rate,
    )
    from particula.gas.properties.dynamic_viscosity import (  # noqa: E402
        get_dynamic_viscosity,
    )
    from particula.gas.properties.mean_free_path import (  # noqa: E402
        get_molecule_mean_free_path,
    )
    from particula.gas.properties.pressure_function import (  # noqa: E402
        get_partial_pressure,
    )
    from particula.gpu.dynamics.condensation_funcs import (  # noqa: E402
        diffusion_coefficient_wp,
        first_order_mass_transport_k_wp,
        mass_transfer_rate_wp,
        water_activity_ideal_wp,
        water_activity_kappa_wp,
    )
    from particula.gpu.properties.gas_properties import (  # noqa: E402
        dynamic_viscosity_wp,
        molecule_mean_free_path_wp,
        partial_pressure_wp,
    )
    from particula.gpu.properties.particle_properties import (  # noqa: E402
        aerodynamic_mobility_wp,
        cunningham_slip_correction_wp,
        kelvin_radius_wp,
        kelvin_term_wp,
        knudsen_number_wp,
        partial_pressure_delta_wp,
        vapor_transition_correction_wp,
    )
    from particula.gpu.tests.cuda_availability import warp_devices  # noqa: E402
    from particula.particles.properties.aerodynamic_mobility_module import (  # noqa: E402
        get_aerodynamic_mobility,
    )
    from particula.particles.properties.diffusion_coefficient import (  # noqa: E402
        get_diffusion_coefficient,
    )
    from particula.particles.properties.kelvin_effect_module import (  # noqa: E402
        get_kelvin_radius,
        get_kelvin_term,
    )
    from particula.particles.properties.knudsen_number_module import (  # noqa: E402
        get_knudsen_number,
    )
    from particula.particles.properties.partial_pressure_module import (  # noqa: E402
        get_partial_pressure_delta,
    )
    from particula.particles.properties.slip_correction_module import (  # noqa: E402
        get_cunningham_slip_correction,
    )
    from particula.particles.properties.vapor_correction_module import (  # noqa: E402
        get_vapor_transition_correction,
    )
    from particula.util.constants import (  # noqa: E402
        BOLTZMANN_CONSTANT,
        GAS_CONSTANT,
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
def _diffusion_coefficient_kernel(
    temperatures: Any,
    aerodynamic_mobilities: Any,
    boltzmann_constant: Any,
    result: Any,
) -> None:
    """Compute diffusion coefficient for each sample.

    Args:
        temperatures: Gas temperatures [K].
        aerodynamic_mobilities: Aerodynamic mobilities [m²/s].
        boltzmann_constant: Boltzmann constant [J/K].
        result: Output array for diffusion coefficients [m²/s].
    """
    tid = wp.tid()
    result[tid] = diffusion_coefficient_wp(
        temperatures[tid],
        aerodynamic_mobilities[tid],
        boltzmann_constant,
    )


@_warp_kernel
def _first_order_mass_transport_kernel(
    particle_radii: Any,
    vapor_transitions: Any,
    diffusion_coefficients: Any,
    result: Any,
) -> None:
    """Compute mass transport coefficients for each sample.

    Args:
        particle_radii: Particle radii [m].
        vapor_transitions: Vapor transition correction factors (dimensionless).
        diffusion_coefficients: Diffusion coefficients [m²/s].
        result: Output array for mass transport coefficients [m³/s].
    """
    tid = wp.tid()
    result[tid] = first_order_mass_transport_k_wp(
        particle_radii[tid],
        vapor_transitions[tid],
        diffusion_coefficients[tid],
    )


@_warp_kernel
def _mass_transfer_rate_kernel(
    pressure_deltas: Any,
    first_order_mass_transports: Any,
    temperatures: Any,
    molar_masses: Any,
    gas_constant: Any,
    result: Any,
) -> None:
    """Compute mass transfer rate for each sample.

    Args:
        pressure_deltas: Partial pressure differences [Pa].
        first_order_mass_transports: Mass transport coefficients [m³/s].
        temperatures: Temperatures [K].
        molar_masses: Molar masses [kg/mol].
        gas_constant: Universal gas constant [J/(mol·K)].
        result: Output array for mass transfer rates [kg/s].
    """
    tid = wp.tid()
    result[tid] = mass_transfer_rate_wp(
        pressure_deltas[tid],
        first_order_mass_transports[tid],
        temperatures[tid],
        molar_masses[tid],
        gas_constant,
    )


@_warp_kernel
def _condensation_chain_kernel(
    temperatures: Any,
    pressures: Any,
    particle_radii: Any,
    molar_masses: Any,
    concentrations: Any,
    partial_pressures_particle: Any,
    surface_tensions: Any,
    densities: Any,
    mass_accommodations: Any,
    boltzmann_constant: Any,
    gas_constant: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    result: Any,
) -> None:
    """Compute condensation mass transfer rate from raw inputs.

    Args:
        temperatures: Gas temperatures [K].
        pressures: Gas pressures [Pa].
        particle_radii: Particle radii [m].
        molar_masses: Molar masses [kg/mol].
        concentrations: Gas concentrations [kg/m³].
        partial_pressures_particle: Particle partial pressures [Pa].
        surface_tensions: Effective surface tensions [N/m].
        densities: Effective densities [kg/m³].
        mass_accommodations: Mass accommodation coefficients (dimensionless).
        boltzmann_constant: Boltzmann constant [J/K].
        gas_constant: Universal gas constant [J/(mol·K)].
        ref_viscosity: Reference viscosity at STP [Pa·s].
        ref_temperature: Reference temperature at STP [K].
        sutherland_constant: Sutherland constant [K].
        result: Output array for mass transfer rates [kg/s].
    """
    tid = wp.tid()
    dynamic_viscosity = dynamic_viscosity_wp(
        temperatures[tid],
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molar_masses[tid],
        temperatures[tid],
        pressures[tid],
        dynamic_viscosity,
        gas_constant,
    )
    knudsen_number = knudsen_number_wp(mean_free_path, particle_radii[tid])
    slip_correction = cunningham_slip_correction_wp(knudsen_number)
    mobility = aerodynamic_mobility_wp(
        particle_radii[tid],
        slip_correction,
        dynamic_viscosity,
    )
    diffusion_coefficient = diffusion_coefficient_wp(
        temperatures[tid],
        mobility,
        boltzmann_constant,
    )
    vapor_transition = vapor_transition_correction_wp(
        knudsen_number,
        mass_accommodations[tid],
    )
    mass_transport = first_order_mass_transport_k_wp(
        particle_radii[tid],
        vapor_transition,
        diffusion_coefficient,
    )
    kelvin_radius = kelvin_radius_wp(
        surface_tensions[tid],
        densities[tid],
        molar_masses[tid],
        temperatures[tid],
        gas_constant,
    )
    kelvin_term = kelvin_term_wp(particle_radii[tid], kelvin_radius)
    partial_pressure_gas = partial_pressure_wp(
        concentrations[tid],
        molar_masses[tid],
        temperatures[tid],
        gas_constant,
    )
    pressure_delta = partial_pressure_delta_wp(
        partial_pressure_gas,
        partial_pressures_particle[tid],
        kelvin_term,
    )
    result[tid] = mass_transfer_rate_wp(
        pressure_delta,
        mass_transport,
        temperatures[tid],
        molar_masses[tid],
        gas_constant,
    )


@_warp_kernel
def _water_activity_ideal_kernel(
    masses: Any,
    molar_masses: Any,
    water_index: int,
    result: Any,
) -> None:
    """Compute ideal water activity for each particle box."""
    case_idx = wp.tid()
    result[case_idx] = water_activity_ideal_wp(
        masses,
        molar_masses,
        case_idx,
        0,
        water_index,
    )


@_warp_kernel
def _water_activity_kappa_kernel(
    masses: Any,
    densities: Any,
    kappas: Any,
    water_index: int,
    result: Any,
) -> None:
    """Compute κ-model water activity for each particle box."""
    case_idx = wp.tid()
    result[case_idx] = water_activity_kappa_wp(
        masses,
        densities,
        kappas,
        case_idx,
        0,
        water_index,
    )


def _ideal_water_activity_reference(
    masses: np.ndarray,
    molar_masses: np.ndarray,
    water_index: int,
) -> np.ndarray:
    """Calculate direct mole-fraction water activity references."""
    expected = []
    for case_masses in masses:
        total_moles = np.sum(case_masses / molar_masses)
        if total_moles == np.float64(0.0):
            expected.append(np.float64(0.0))
        else:
            expected.append(
                case_masses[water_index]
                / molar_masses[water_index]
                / total_moles
            )
    return np.asarray(expected, dtype=np.float64)


def _kappa_water_activity_reference(
    masses: np.ndarray,
    densities: np.ndarray,
    kappas: np.ndarray,
    water_index: int,
) -> np.ndarray:
    """Calculate direct volume-based κ-model water activity references."""
    expected = []
    for case_masses in masses:
        volumes = case_masses / densities
        water_volume = volumes[water_index]
        solute_volume = np.sum(volumes) - water_volume
        if water_volume == np.float64(0.0):
            expected.append(np.float64(0.0))
        elif solute_volume == np.float64(0.0):
            expected.append(np.float64(1.0))
        else:
            kappa_volume = np.float64(0.0)
            for species_idx, species_volume in enumerate(volumes):
                if species_idx != water_index:
                    kappa_volume += kappas[species_idx] * species_volume
            expected.append(
                np.float64(1.0)
                / (np.float64(1.0) + kappa_volume / water_volume)
            )
    return np.asarray(expected, dtype=np.float64)


@pytest.fixture(params=_available_warp_devices())
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def test_diffusion_coefficient_matches_numpy(device: str) -> None:
    """Ensure diffusion_coefficient_wp matches NumPy reference values."""
    temperatures = np.array([250.0, 298.15, 320.0], dtype=np.float64)
    mobilities = np.array([1.0e-8, 2.5e-8, 4.0e-8], dtype=np.float64)
    expected = np.array(
        [
            get_diffusion_coefficient(temp, mobility)
            for temp, mobility in zip(temperatures, mobilities, strict=True)
        ],
        dtype=np.float64,
    )

    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    mobilities_wp = wp.array(mobilities, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(temperatures), dtype=wp.float64, device=device)

    wp.launch(
        _diffusion_coefficient_kernel,
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


def test_first_order_mass_transport_k_matches_numpy(device: str) -> None:
    """Ensure first_order_mass_transport_k_wp matches NumPy reference values."""
    particle_radii = np.array([5.0e-8, 1.0e-7, 2.0e-7], dtype=np.float64)
    vapor_transitions = np.array([0.5, 0.7, 1.0], dtype=np.float64)
    diffusion_coefficients = np.array(
        [1.0e-9, 2.0e-9, 5.0e-9],
        dtype=np.float64,
    )
    expected = np.array(
        [
            get_first_order_mass_transport_k(radius, transition, coefficient)
            for radius, transition, coefficient in zip(
                particle_radii,
                vapor_transitions,
                diffusion_coefficients,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    particle_radii_wp = wp.array(
        particle_radii, dtype=wp.float64, device=device
    )
    vapor_transitions_wp = wp.array(
        vapor_transitions, dtype=wp.float64, device=device
    )
    diffusion_wp = wp.array(
        diffusion_coefficients, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)

    wp.launch(
        _first_order_mass_transport_kernel,
        dim=len(particle_radii),
        inputs=[particle_radii_wp, vapor_transitions_wp, diffusion_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-10, atol=1e-20)


def test_mass_transfer_rate_matches_numpy(device: str) -> None:
    """Ensure mass_transfer_rate_wp matches NumPy reference values."""
    pressure_deltas = np.array([5.0, -2.0, 12.0], dtype=np.float64)
    mass_transports = np.array([1.0e-17, 2.0e-17, 3.5e-17], dtype=np.float64)
    temperatures = np.array([290.0, 298.15, 310.0], dtype=np.float64)
    molar_masses = np.array([0.018, 0.029, 0.044], dtype=np.float64)
    expected = np.array(
        [
            get_mass_transfer_rate(delta, transport, temp, molar_mass)
            for delta, transport, temp, molar_mass in zip(
                pressure_deltas,
                mass_transports,
                temperatures,
                molar_masses,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    pressure_wp = wp.array(pressure_deltas, dtype=wp.float64, device=device)
    transport_wp = wp.array(mass_transports, dtype=wp.float64, device=device)
    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    molar_masses_wp = wp.array(molar_masses, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(pressure_deltas), dtype=wp.float64, device=device)

    wp.launch(
        _mass_transfer_rate_kernel,
        dim=len(pressure_deltas),
        inputs=[
            pressure_wp,
            transport_wp,
            temperatures_wp,
            molar_masses_wp,
            wp.float64(GAS_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-10, atol=1e-20)


def test_condensation_chain_matches_numpy(device: str) -> None:
    """Ensure chained condensation calculation matches NumPy reference."""
    temperatures = np.array([298.15, 310.0], dtype=np.float64)
    pressures = np.array([101325.0, 90000.0], dtype=np.float64)
    particle_radii = np.array([1.0e-7, 2.0e-7], dtype=np.float64)
    molar_masses = np.array([0.018, 0.044], dtype=np.float64)
    concentrations = np.array([1.2, 0.8], dtype=np.float64)
    partial_pressures_particle = np.array([600.0, 400.0], dtype=np.float64)
    surface_tensions = np.array([0.072, 0.072], dtype=np.float64)
    densities = np.array([1000.0, 1200.0], dtype=np.float64)
    mass_accommodations = np.array([1.0, 0.9], dtype=np.float64)

    expected = []
    for (
        temperature,
        pressure,
        radius,
        molar_mass,
        concentration,
        partial_pressure,
        surface_tension,
        density,
        accommodation,
    ) in zip(
        temperatures,
        pressures,
        particle_radii,
        molar_masses,
        concentrations,
        partial_pressures_particle,
        surface_tensions,
        densities,
        mass_accommodations,
        strict=True,
    ):
        dynamic_viscosity = get_dynamic_viscosity(
            temperature,
            reference_viscosity=REF_VISCOSITY_AIR_STP,
            reference_temperature=REF_TEMPERATURE_STP,
        )
        mean_free_path = get_molecule_mean_free_path(
            molar_mass=molar_mass,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        knudsen_number = get_knudsen_number(mean_free_path, radius)
        slip_correction = get_cunningham_slip_correction(knudsen_number)
        mobility = get_aerodynamic_mobility(
            particle_radius=radius,
            slip_correction_factor=slip_correction,
            dynamic_viscosity=dynamic_viscosity,
        )
        diffusion_coefficient = get_diffusion_coefficient(
            temperature=temperature,
            aerodynamic_mobility=mobility,
            boltzmann_constant=BOLTZMANN_CONSTANT,
        )
        vapor_transition = get_vapor_transition_correction(
            knudsen_number=knudsen_number,
            mass_accommodation=accommodation,
        )
        mass_transport = get_first_order_mass_transport_k(
            particle_radius=radius,
            vapor_transition=vapor_transition,
            diffusion_coefficient=diffusion_coefficient,
        )
        kelvin_radius = get_kelvin_radius(
            effective_surface_tension=surface_tension,
            effective_density=density,
            molar_mass=molar_mass,
            temperature=temperature,
        )
        kelvin_term = get_kelvin_term(radius, kelvin_radius)
        partial_pressure_gas = get_partial_pressure(
            concentration=concentration,
            molar_mass=molar_mass,
            temperature=temperature,
        )
        pressure_delta = get_partial_pressure_delta(
            partial_pressure_gas=partial_pressure_gas,
            partial_pressure_particle=partial_pressure,
            kelvin_term=kelvin_term,
        )
        expected.append(
            get_mass_transfer_rate(
                pressure_delta=pressure_delta,
                first_order_mass_transport=mass_transport,
                temperature=temperature,
                molar_mass=molar_mass,
            )
        )

    expected_array = np.array(expected, dtype=np.float64)

    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    pressures_wp = wp.array(pressures, dtype=wp.float64, device=device)
    radii_wp = wp.array(particle_radii, dtype=wp.float64, device=device)
    molar_masses_wp = wp.array(molar_masses, dtype=wp.float64, device=device)
    concentrations_wp = wp.array(
        concentrations, dtype=wp.float64, device=device
    )
    partial_particle_wp = wp.array(
        partial_pressures_particle, dtype=wp.float64, device=device
    )
    surface_tensions_wp = wp.array(
        surface_tensions, dtype=wp.float64, device=device
    )
    densities_wp = wp.array(densities, dtype=wp.float64, device=device)
    accommodations_wp = wp.array(
        mass_accommodations, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(temperatures), dtype=wp.float64, device=device)

    wp.launch(
        _condensation_chain_kernel,
        dim=len(temperatures),
        inputs=[
            temperatures_wp,
            pressures_wp,
            radii_wp,
            molar_masses_wp,
            concentrations_wp,
            partial_particle_wp,
            surface_tensions_wp,
            densities_wp,
            accommodations_wp,
            wp.float64(BOLTZMANN_CONSTANT),
            wp.float64(GAS_CONSTANT),
            wp.float64(REF_VISCOSITY_AIR_STP),
            wp.float64(REF_TEMPERATURE_STP),
            wp.float64(SUTHERLAND_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected_array,
        rtol=1e-10,
        atol=1e-20,
    )


@pytest.mark.parametrize(
    ("case_masses", "water_index"),
    [
        (
            np.array(
                [
                    [18.0, 0.0, 0.0],
                    [18.0, 44.0, 58.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 44.0, 58.0],
                ],
                dtype=np.float64,
            ),
            0,
        ),
        (np.array([[44.0, 18.0, 58.0]], dtype=np.float64), 1),
    ],
    ids=[
        "water-first-species",
        "water-nonzero-species-index",
    ],
)
def test_water_activity_ideal_matches_independent_reference(
    device: str,
    case_masses: np.ndarray,
    water_index: int,
) -> None:
    """Ensure ideal water activity matches a direct NumPy mole sum."""
    molar_masses = np.array([18.0, 44.0, 58.0], dtype=np.float64)
    masses = case_masses[:, np.newaxis, :]
    expected = _ideal_water_activity_reference(
        masses[:, 0, :], molar_masses, water_index
    )

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    molar_masses_wp = wp.array(molar_masses, dtype=wp.float64, device=device)
    result_wp = wp.zeros(masses.shape[0], dtype=wp.float64, device=device)
    wp.launch(
        _water_activity_ideal_kernel,
        dim=masses.shape[0],
        inputs=[masses_wp, molar_masses_wp, water_index],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    result = result_wp.numpy()
    assert result.shape == (masses.shape[0],)
    assert np.all(np.isfinite(result))
    npt.assert_allclose(result, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize(
    ("case_masses", "kappas", "water_index"),
    [
        (
            np.array(
                [
                    [18.0, 55.0, 110.0],
                    [18.0, 0.0, 0.0],
                    [0.0, 55.0, 87.0],
                    [18.0, 55.0, 87.0],
                ],
                dtype=np.float64,
            ),
            np.array([99.0, 0.35, 0.75], dtype=np.float64),
            0,
        ),
        (
            np.array([[18.0, 0.0, 0.0]], dtype=np.float64),
            np.array([99.0, 0.0, 0.0], dtype=np.float64),
            0,
        ),
        (
            np.array([[0.0, 55.0, 87.0]], dtype=np.float64),
            np.array([99.0, 0.35, 0.75], dtype=np.float64),
            0,
        ),
        (
            np.array([[18.0, 55.0, 87.0]], dtype=np.float64),
            np.array([99.0, 0.0, 0.0], dtype=np.float64),
            0,
        ),
        (
            np.array([[55.0, 18.0, 87.0]], dtype=np.float64),
            np.array([0.35, 99.0, 0.75], dtype=np.float64),
            1,
        ),
    ],
    ids=[
        "wet-multi-solute-distinct-kappas",
        "pure-water-zero-solute",
        "dry-no-water",
        "wet-zero-kappa-solutes",
        "water-nonzero-species-index",
    ],
)
def test_water_activity_kappa_matches_independent_reference(
    device: str,
    case_masses: np.ndarray,
    kappas: np.ndarray,
    water_index: int,
) -> None:
    """Ensure κ water activity matches a direct NumPy volume sum."""
    densities = np.array([1000.0, 1100.0, 1450.0], dtype=np.float64)
    masses = case_masses[:, np.newaxis, :]
    expected = _kappa_water_activity_reference(
        masses[:, 0, :], densities, kappas, water_index
    )

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    densities_wp = wp.array(densities, dtype=wp.float64, device=device)
    kappas_wp = wp.array(kappas, dtype=wp.float64, device=device)
    result_wp = wp.zeros(masses.shape[0], dtype=wp.float64, device=device)
    wp.launch(
        _water_activity_kappa_kernel,
        dim=masses.shape[0],
        inputs=[masses_wp, densities_wp, kappas_wp, water_index],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    result = result_wp.numpy()
    assert result.shape == (masses.shape[0],)
    assert np.all(np.isfinite(result))
    npt.assert_allclose(result, expected, rtol=1e-10, atol=0.0)
