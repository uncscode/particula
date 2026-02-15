"""GPU condensation kernels and orchestration utilities.

This module composes the condensation ``@wp.func`` building blocks into
end-to-end kernels and provides a high-level ``condensation_step_gpu``
orchestration API. Kernel launches operate on GPU-resident Warp
arrays and update particle masses in-place.
"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false

from typing import Any

import particula.util.constants as constants

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled via import guards
    raise ImportError(
        "Warp is required for GPU condensation kernels. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.mean_free_path import (
    get_molecule_mean_free_path,
)
from particula.gpu.dynamics.condensation_funcs import (
    diffusion_coefficient_wp,
    first_order_mass_transport_k_wp,
    mass_transfer_rate_wp,
    particle_radius_from_volume_wp,
)
from particula.gpu.properties.gas_properties import partial_pressure_wp
from particula.gpu.properties.particle_properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    kelvin_radius_wp,
    kelvin_term_wp,
    knudsen_number_wp,
    partial_pressure_delta_wp,
    vapor_transition_correction_wp,
)

# type: ignore
_DEFAULT_SURFACE_TENSION = 0.072
_DEFAULT_MASS_ACCOMMODATION = 1.0
_DEFAULT_DIFFUSION_COEFFICIENT = 2.0e-5


@wp.kernel
# type: ignore[misc]
def condensation_mass_transfer_kernel(
    masses: Any,
    concentration: Any,
    density: Any,
    gas_concentration: Any,
    vapor_pressure: Any,
    molar_mass: Any,
    surface_tension: Any,
    mass_accommodation: Any,
    diffusion_coefficient_vapor: Any,
    dynamic_viscosity: Any,
    mean_free_path: Any,
    gas_constant: Any,
    boltzmann_constant: Any,
    temperature: Any,
    time_step: Any,
    mass_transfer: Any,
) -> None:
    """Compute per-species condensation mass transfer for each particle."""  # type: ignore
    box_idx, particle_idx = wp.tid()

    if concentration[box_idx, particle_idx] == wp.float64(0.0):
        for species_idx in range(masses.shape[2]):
            mass_transfer[box_idx, particle_idx, species_idx] = wp.float64(0.0)
        return

    n_species = masses.shape[2]
    total_volume = wp.float64(0.0)
    total_mass = wp.float64(0.0)
    for species_idx in range(n_species):
        species_mass = masses[box_idx, particle_idx, species_idx]
        total_mass += species_mass
        total_volume += species_mass / density[species_idx]

    if total_volume <= wp.float64(0.0):
        for species_idx in range(n_species):
            mass_transfer[box_idx, particle_idx, species_idx] = wp.float64(0.0)
        return

    radius = particle_radius_from_volume_wp(total_volume)
    effective_density = total_mass / total_volume
    if effective_density <= wp.float64(0.0):
        effective_density = density[0]

    knudsen_number = knudsen_number_wp(mean_free_path, radius)
    slip_correction = cunningham_slip_correction_wp(knudsen_number)
    mobility = aerodynamic_mobility_wp(
        radius, slip_correction, dynamic_viscosity
    )
    diffusion_coefficient_particle = diffusion_coefficient_wp(
        temperature,
        mobility,
        boltzmann_constant,
    )

    for species_idx in range(n_species):
        transition = vapor_transition_correction_wp(
            knudsen_number,
            mass_accommodation[species_idx],
        )
        diffusion_value = diffusion_coefficient_vapor[species_idx]
        if diffusion_value <= wp.float64(0.0):
            diffusion_value = diffusion_coefficient_particle
        mass_transport = first_order_mass_transport_k_wp(
            radius,
            transition,
            diffusion_value,
        )
        kelvin_radius = kelvin_radius_wp(
            surface_tension[species_idx],
            effective_density,
            molar_mass[species_idx],
            temperature,
            gas_constant,
        )
        kelvin_term = kelvin_term_wp(radius, kelvin_radius)
        partial_pressure_gas = partial_pressure_wp(
            gas_concentration[box_idx, species_idx],
            molar_mass[species_idx],
            temperature,
            gas_constant,
        )
        pressure_delta = partial_pressure_delta_wp(
            partial_pressure_gas,
            vapor_pressure[box_idx, species_idx],
            kelvin_term,
        )
        mass_rate = mass_transfer_rate_wp(
            pressure_delta,
            mass_transport,
            temperature,
            molar_mass[species_idx],
            gas_constant,
        )
        mass_transfer[box_idx, particle_idx, species_idx] = (
            mass_rate * time_step
        )


@wp.kernel
# type: ignore[misc]
def apply_mass_transfer_kernel(
    masses: Any,
    mass_transfer: Any,
) -> None:
    """Apply mass transfer with non-negative clamping."""
    box_idx, particle_idx = wp.tid()
    n_species = masses.shape[2]
    for species_idx in range(n_species):
        updated_mass = (
            masses[box_idx, particle_idx, species_idx]
            + mass_transfer[box_idx, particle_idx, species_idx]
        )
        if updated_mass < wp.float64(0.0):
            updated_mass = wp.float64(0.0)
        masses[box_idx, particle_idx, species_idx] = updated_mass


def _validate_species_array(
    name: str,
    array: Any,
    n_species: int,
    expected_device: Any,
) -> None:
    """Validate per-species array length and device.

    Args:
        name: Array name for error messages.
        array: Array-like object with ``shape`` attribute.
        n_species: Expected length.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the array length or device mismatches expectations.
    """
    if len(array.shape) != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.shape[0] != n_species:
        raise ValueError(
            f"{name} length {array.shape[0]} does not match n_species "
            f"{n_species}"
        )
    device = getattr(array, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(f"{name} device does not match particle device")


def _validate_mass_transfer_buffer(
    mass_transfer: Any,
    expected_shape: tuple[int, int, int],
    expected_device: str,
) -> None:
    """Validate mass transfer buffer shape and device."""
    if mass_transfer.shape != expected_shape:
        raise ValueError(
            f"mass_transfer shape {mass_transfer.shape} does not match "
            f"expected {expected_shape}"
        )
    device = getattr(mass_transfer, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(
            "mass_transfer buffer device does not match particle device"
        )


def _validate_particle_arrays(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Validate particle array shapes and lengths."""
    if particles.density.shape[0] != n_species:
        raise ValueError("particle density length does not match n_species")
    if particles.concentration.shape != (n_boxes, n_particles):
        raise ValueError(
            "particle concentration shape does not match (n_boxes, n_particles)"
        )


def _validate_gas_arrays(
    gas: Any,
    n_boxes: int,
    n_species: int,
) -> None:
    """Validate gas array shapes and lengths."""
    if gas.molar_mass.shape[0] != n_species:
        raise ValueError(
            "n_species mismatch between particle masses and gas molar mass"
        )
    if gas.concentration.shape != (n_boxes, n_species):
        raise ValueError(
            "gas concentration shape does not match (n_boxes, n_species)"
        )
    if gas.vapor_pressure.shape != (n_boxes, n_species):
        raise ValueError(
            "vapor pressure shape does not match (n_boxes, n_species)"
        )


def _validate_device_match(name: str, array: Any, expected_device: Any) -> None:
    """Validate that a Warp array is on the expected device."""
    device = getattr(array, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(f"{name} device mismatch")


def _validate_device_arrays(particles: Any, gas: Any, device: Any) -> None:
    """Validate particle and gas arrays share the same Warp device."""
    _validate_device_match(
        "particle concentration", particles.concentration, device
    )
    _validate_device_match("particle density", particles.density, device)
    _validate_device_match("gas molar mass", gas.molar_mass, device)
    _validate_device_match("gas concentration", gas.concentration, device)
    _validate_device_match("gas vapor pressure", gas.vapor_pressure, device)


def condensation_step_gpu(
    particles: Any,
    gas: Any,
    temperature: float,
    pressure: float,
    time_step: float,
    surface_tension: Any | None = None,
    mass_accommodation: Any | None = None,
    diffusion_coefficient_vapor: Any | None = None,
    mass_transfer: Any | None = None,
) -> tuple[Any, Any]:
    """Execute one condensation timestep on the GPU.

    Args:
        particles: GPU-resident particle data.
        gas: GPU-resident gas data.
        temperature: Gas temperature in kelvin.
        pressure: Gas pressure in pascals.
        time_step: Condensation time step in seconds.
        surface_tension: Optional per-species surface tension [N/m].
        mass_accommodation: Optional per-species accommodation coefficient.
        diffusion_coefficient_vapor: Optional per-species vapor diffusion
            coefficient [m^2/s].
        mass_transfer: Optional preallocated mass transfer buffer with shape
            ``(n_boxes, n_particles, n_species)``.

    Returns:
        Tuple of updated particle data and the mass transfer buffer.

    Raises:
        ValueError: If species counts, array lengths, or devices mismatch.

    Notes:
        Particle masses are updated in-place on the GPU. Callers that require
        rollback should copy masses before invoking this function.
    """
    n_boxes, n_particles, n_species = particles.masses.shape
    _validate_gas_arrays(gas, n_boxes, n_species)
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, gas, device)

    if surface_tension is None:
        surface_tension = wp.full(
            n_species,
            wp.float64(_DEFAULT_SURFACE_TENSION),
            dtype=wp.float64,
            device=device,
        )
    else:
        _validate_species_array(
            "surface_tension",
            surface_tension,
            n_species,
            device,
        )

    if mass_accommodation is None:
        mass_accommodation = wp.full(
            n_species,
            wp.float64(_DEFAULT_MASS_ACCOMMODATION),
            dtype=wp.float64,
            device=device,
        )
    else:
        _validate_species_array(
            "mass_accommodation",
            mass_accommodation,
            n_species,
            device,
        )

    if diffusion_coefficient_vapor is None:
        diffusion_coefficient_vapor = wp.full(
            n_species,
            wp.float64(_DEFAULT_DIFFUSION_COEFFICIENT),
            dtype=wp.float64,
            device=device,
        )
    else:
        _validate_species_array(
            "diffusion_coefficient_vapor",
            diffusion_coefficient_vapor,
            n_species,
            device,
        )

    expected_shape = (n_boxes, n_particles, n_species)
    if mass_transfer is None:
        mass_transfer = wp.zeros(
            expected_shape,
            dtype=wp.float64,
            device=device,
        )
    else:
        _validate_mass_transfer_buffer(mass_transfer, expected_shape, device)

    dynamic_viscosity = get_dynamic_viscosity(
        temperature,
        reference_viscosity=constants.REF_VISCOSITY_AIR_STP,
        reference_temperature=constants.REF_TEMPERATURE_STP,
    )
    mean_free_path = get_molecule_mean_free_path(
        molar_mass=constants.MOLECULAR_WEIGHT_AIR,
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )

    wp.launch(
        condensation_mass_transfer_kernel,
        dim=(n_boxes, n_particles),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            gas.concentration,
            gas.vapor_pressure,
            gas.molar_mass,
            surface_tension,
            mass_accommodation,
            diffusion_coefficient_vapor,
            wp.float64(dynamic_viscosity),
            wp.float64(mean_free_path),
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(temperature),
            wp.float64(time_step),
            mass_transfer,
        ],
        device=device,
    )

    wp.launch(
        apply_mass_transfer_kernel,
        dim=(n_boxes, n_particles),
        inputs=[particles.masses, mass_transfer],
        device=device,
    )

    return particles, mass_transfer
