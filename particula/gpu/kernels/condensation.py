"""GPU condensation kernels and orchestration utilities.

This module composes the condensation ``@wp.func`` building blocks into
end-to-end kernels and provides a high-level ``condensation_step_gpu``
orchestration API. Entry-point validation accepts scalar direct inputs,
explicit ``(n_boxes,)`` Warp arrays, or a ``WarpEnvironmentData`` container,
then normalizes those sources into per-box Warp arrays before launch-time
work. Kernel launches operate on GPU-resident Warp arrays and update particle
masses in-place.
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

from particula.gpu.dynamics.condensation_funcs import (
    diffusion_coefficient_wp,
    first_order_mass_transport_k_wp,
    mass_transfer_rate_wp,
    particle_radius_from_volume_wp,
)
from particula.gpu.kernels.environment import _ensure_environment_arrays
from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
    partial_pressure_wp,
)
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
    """Compute condensation mass transfer for each particle species.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        concentration: Particle number concentration array.
        density: Per-species particle density array.
        gas_concentration: Gas concentrations array.
        vapor_pressure: Gas-phase vapor pressure array.
        molar_mass: Gas-phase molar mass array.
        surface_tension: Per-species surface tension array.
        mass_accommodation: Per-species mass accommodation coefficients.
        diffusion_coefficient_vapor: Per-species vapor diffusion coefficients.
        dynamic_viscosity: Per-box gas dynamic viscosity [Pa·s].
        mean_free_path: Per-box gas mean free path [m].
        gas_constant: Universal gas constant [J/(mol·K)].
        boltzmann_constant: Boltzmann constant [J/K].
        temperature: Per-box gas temperature [K].
        time_step: Condensation time step [s].
        mass_transfer: Output mass transfer array.
    """  # type: ignore
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    dynamic_viscosity_value = dynamic_viscosity[box_idx]
    mean_free_path_value = mean_free_path[box_idx]
    temperature_value = temperature[box_idx]

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

    if total_volume <= wp.float64(0.0):  # type: ignore[operator]
        for species_idx in range(n_species):
            mass_transfer[box_idx, particle_idx, species_idx] = wp.float64(0.0)
        return

    radius = particle_radius_from_volume_wp(total_volume)
    effective_density = total_mass / total_volume
    if effective_density <= wp.float64(0.0):
        effective_density = density[0]

    knudsen_number = knudsen_number_wp(mean_free_path_value, radius)
    slip_correction = cunningham_slip_correction_wp(knudsen_number)
    mobility = aerodynamic_mobility_wp(
        radius, slip_correction, dynamic_viscosity_value
    )
    diffusion_coefficient_particle = diffusion_coefficient_wp(
        temperature_value,
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
            temperature_value,
            gas_constant,
        )
        kelvin_term = kelvin_term_wp(radius, kelvin_radius)
        partial_pressure_gas = partial_pressure_wp(
            gas_concentration[box_idx, species_idx],
            molar_mass[species_idx],
            temperature_value,
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
            temperature_value,
            molar_mass[species_idx],
            gas_constant,
        )
        mass_transfer[box_idx, particle_idx, species_idx] = (
            mass_rate * time_step
        )


@wp.kernel
# type: ignore[misc]
def _prepare_environment_properties_kernel(
    temperature: Any,
    pressure: Any,
    dynamic_viscosity: Any,
    mean_free_path: Any,
) -> None:
    """Precompute box-level gas properties once per entry-point call."""
    box_idx = wp.tid()  # type: ignore[misc]
    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]
    viscosity_value = dynamic_viscosity_wp(
        temperature_value,
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )
    dynamic_viscosity[box_idx] = viscosity_value
    mean_free_path[box_idx] = molecule_mean_free_path_wp(
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        temperature_value,
        pressure_value,
        viscosity_value,
        wp.float64(constants.GAS_CONSTANT),
    )


@wp.kernel
# type: ignore[misc]
def apply_mass_transfer_kernel(
    masses: Any,
    mass_transfer: Any,
) -> None:
    """Apply condensation mass transfer and clamp masses to non-negative.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        mass_transfer: Mass transfer array matching ``masses`` shape.
    """
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
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
    """Validate mass transfer buffer shape and device.

    Args:
        mass_transfer: Mass transfer array allocated on the GPU.
        expected_shape: Expected ``(n_boxes, n_particles, n_species)`` shape.
        expected_device: Expected Warp device name.

    Raises:
        ValueError: If the shape or device does not match expectations.
    """
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
    """Validate particle array shapes and lengths.

    Args:
        particles: Particle data containing density and concentration arrays.
        n_boxes: Expected number of spatial boxes.
        n_particles: Expected number of particles per box.
        n_species: Expected number of particle species.

    Raises:
        ValueError: If particle array shapes do not match expectations.
    """
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
    """Validate gas array shapes and lengths.

    Args:
        gas: Gas data containing molar mass, concentration, and vapor pressure.
        n_boxes: Expected number of spatial boxes.
        n_species: Expected number of gas species.

    Raises:
        ValueError: If gas array shapes do not match expectations.
    """
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
    """Validate that a Warp array is on the expected device.

    Args:
        name: Array label for error messages.
        array: Warp array to validate.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the array is not on the expected device.
    """
    device = getattr(array, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(f"{name} device mismatch")


def _validate_device_arrays(particles: Any, gas: Any, device: Any) -> None:
    """Validate particle and gas arrays share the same Warp device.

    Args:
        particles: Particle data with GPU-backed arrays.
        gas: Gas data with GPU-backed arrays.
        device: Expected Warp device.

    Raises:
        ValueError: If any particle or gas array is on a different device.
    """
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
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    surface_tension: Any | None = None,
    mass_accommodation: Any | None = None,
    diffusion_coefficient_vapor: Any | None = None,
    mass_transfer: Any | None = None,
    *,
    environment: Any | None = None,
) -> tuple[Any, Any]:
    """Execute one condensation timestep on the GPU.

    Args:
        particles: GPU-resident particle data.
        gas: GPU-resident gas data.
        temperature: Direct gas temperature as either a scalar or a Warp array
            with shape ``(n_boxes,)``. Use ``None`` only with
            ``environment=...``.
        pressure: Direct gas pressure as either a scalar or a Warp array with
            shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
        time_step: Condensation time step in seconds.
        surface_tension: Optional per-species surface tension [N/m].
        mass_accommodation: Optional per-species accommodation coefficient.
        diffusion_coefficient_vapor: Optional per-species vapor diffusion
            coefficient [m^2/s].
        mass_transfer: Optional preallocated mass transfer buffer with shape
            ``(n_boxes, n_particles, n_species)``.
        environment: Optional ``WarpEnvironmentData`` with ``(n_boxes,)``
            temperature and pressure arrays on the same device as ``particles``
            and ``gas``.

    Returns:
        Tuple of updated particle data and the mass transfer buffer.

    Raises:
        ValueError: If species counts, array lengths, or devices mismatch.
        ValueError: If direct ``temperature`` or ``pressure`` inputs are mixed
            with ``environment``.
        ValueError: If direct inputs are missing when ``environment`` is
            omitted.
        ValueError: If environment arrays do not match ``(n_boxes,)`` or the
            caller device.

    Notes:
        Particle masses are updated in-place on the GPU. Callers that require
        rollback should copy masses before invoking this function.

        ``environment`` remains keyword-only so existing positional scalar
        callers stay source-compatible.

        Validation runs before optional buffer setup or Warp launches so
        invalid shape or device combinations fail without mutating particle
        state. Box-level gas properties are prepared once per call and reused
        during the per-particle kernel launch.
    """
    n_boxes, n_particles, n_species = particles.masses.shape
    _validate_gas_arrays(gas, n_boxes, n_species)
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, gas, device)
    temperature_array, pressure_array = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=environment,
        n_boxes=n_boxes,
        device=device,
        caller_name="condensation_step_gpu",
    )

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

    dynamic_viscosity = wp.zeros((n_boxes,), dtype=wp.float64, device=device)
    mean_free_path = wp.zeros((n_boxes,), dtype=wp.float64, device=device)

    wp.launch(
        _prepare_environment_properties_kernel,
        dim=n_boxes,
        inputs=[
            temperature_array,
            pressure_array,
            dynamic_viscosity,
            mean_free_path,
        ],
        device=device,
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
            dynamic_viscosity,
            mean_free_path,
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            temperature_array,
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
