"""Apply bounded direct GPU wall loss to fixed particle slots.

This concrete direct-kernel boundary supports particle-resolved neutral and
charged wall loss. In charged mode, only nonzero charge composes the private P2
image-charge and P3 field-drift coefficient helpers; zero-charge slots retain
the exact neutral coefficient and random-number path. A charged rectangular
field is caller-owned same-device ``wp.float64`` storage with shape ``(3,)``.
Only the charged rectangular removal kernel reads it, and it is never mutated.
The boundary retains frozen read-only preflight and sequential per-box random
number generation for eligible fixed slots.
An omitted RNG sidecar is private and seeded for each successful positive-time
call. A supplied ``uint32`` Warp sidecar remains caller-owned, advances in
place only for eligible slots, and resets only when explicitly requested.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, no_type_check

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled by test guards
    raise ImportError(
        "Warp is required for GPU wall-loss helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.dynamics.wall_loss_funcs import (
    _combine_charged_wall_loss_coefficient_wp,
    _electric_field_drift_from_viscosity_wp,
    _geometry_scale_wp,
    _image_charge_enhancement_wp,
    _resolve_rectangular_electric_field_wp,
    _resolve_spherical_electric_field_wp,
    rectangle_wall_loss_coefficient_from_transport_wp,
    spherical_wall_loss_coefficient_from_transport_wp,
)
from particula.gpu.kernels.environment import (
    _ensure_environment_arrays,
    _is_warp_array_like,
    validate_environment_inputs,
)
from particula.gpu.properties import (
    dynamic_viscosity_wp,
    effective_density_wp,
    molecule_mean_free_path_wp,
    particle_radius_from_volume_wp,
)
from particula.util.constants import (
    BOLTZMANN_CONSTANT,
    ELECTRIC_PERMITTIVITY,
    ELEMENTARY_CHARGE_VALUE,
    GAS_CONSTANT,
    MOLECULAR_WEIGHT_AIR,
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)

_BOLTZMANN_CONSTANT = wp.constant(wp.float64(BOLTZMANN_CONSTANT))
_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))
_MOLECULAR_WEIGHT_AIR = wp.constant(wp.float64(MOLECULAR_WEIGHT_AIR))
_REF_VISCOSITY_AIR_STP = wp.constant(wp.float64(REF_VISCOSITY_AIR_STP))
_REF_TEMPERATURE_STP = wp.constant(wp.float64(REF_TEMPERATURE_STP))
_SUTHERLAND_CONSTANT = wp.constant(wp.float64(SUTHERLAND_CONSTANT))
_ELEMENTARY_CHARGE_VALUE = wp.constant(wp.float64(ELEMENTARY_CHARGE_VALUE))
_ELECTRIC_PERMITTIVITY = wp.constant(wp.float64(ELECTRIC_PERMITTIVITY))


@dataclass(frozen=True)
class NeutralWallLossConfig:
    """Define immutable direct wall-loss geometry and charged inputs.

    This concrete-module-only configuration is accepted by the direct P4
    boundary, which retains frozen P3 preflight. ``geometry`` selects spherical
    or rectangular SI dimensions, while ``distribution_type`` must remain
    ``"particle_resolved"``. ``mode`` is ``"neutral"`` or ``"charged"``.
    Charged execution composes private P2 image and P3 drift helpers only for
    nonzero charges. Import this concrete configuration
    from ``particula.gpu.kernels.wall_loss``. A charged rectangular electric
    field is caller-owned device storage and is never replaced or mutated.

    Attributes:
        geometry: Exact chamber geometry, ``"spherical"`` or ``"rectangular"``.
        wall_eddy_diffusivity: Positive wall eddy diffusivity [m^2 s^-1].
        chamber_radius: Positive spherical chamber radius [m]; required only
            for spherical geometry.
        chamber_dimensions: Tuple of three positive rectangular chamber
            dimensions [m]; required only for rectangular geometry.
        distribution_type: Required ``"particle_resolved"`` representation
            selector.
        mode: ``"neutral"`` or ``"charged"`` execution mode. Zero-charge
            charged slots retain the neutral coefficient and RNG path.
        wall_potential: Finite signed wall potential [V].
        wall_electric_field: Finite signed spherical scalar field [V m^-1], or
            caller-owned same-device charged rectangular ``wp.float64`` vector
            with shape ``(3,)`` [V m^-1]. A rectangular vector is invalid in
            neutral mode and is neither replaced nor mutated in charged mode.
    """

    geometry: str
    wall_eddy_diffusivity: float
    chamber_radius: float | None = None
    chamber_dimensions: tuple[float, float, float] | None = None
    distribution_type: str = "particle_resolved"
    mode: str = "neutral"
    wall_potential: float = 0.0
    wall_electric_field: Any = 0.0


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _scan_nonnegative_finite_1d(
    values: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid entries in one-dimensional nonnegative data."""
    index = wp.tid()
    if not wp.isfinite(values[index]) or values[index] < 0.0:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _scan_nonnegative_finite_2d(
    values: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid entries in two-dimensional nonnegative data."""
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]) or values[row, column] < 0.0:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _scan_finite_2d(
    values: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record non-finite entries in two-dimensional metadata."""
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _scan_finite_1d(
    values: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record non-finite entries in one-dimensional signed metadata."""
    index = wp.tid()
    if not wp.isfinite(values[index]):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _scan_nonnegative_finite_3d(
    values: wp.array3d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid entries in three-dimensional nonnegative data."""
    box, particle, species = wp.tid()
    if (
        not wp.isfinite(values[box, particle, species])
        or values[box, particle, species] < 0.0
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _scan_positive_finite_1d(
    values: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid entries in one-dimensional positive data."""
    index = wp.tid()
    if not wp.isfinite(values[index]) or values[index] <= 0.0:
        wp.atomic_add(invalid, 0, 1)


def _require_positive_real(value: Any, name: str) -> float:
    """Validate an exact scalar positive finite physical configuration value."""
    if isinstance(value, (bool, np.bool_, np.ndarray)) or not isinstance(
        value, Real
    ):
        raise TypeError(f"{name} must be a positive real scalar.")
    try:
        scalar = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and > 0.") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return scalar


def _require_finite_real(value: Any, name: str) -> float:
    """Validate an exact scalar finite signed configuration value."""
    if isinstance(value, (bool, np.bool_, np.ndarray)) or not isinstance(
        value, Real
    ):
        raise TypeError(f"{name} must be a finite real scalar.")
    try:
        scalar = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _validate_geometry(config: NeutralWallLossConfig) -> None:
    """Validate geometry and its scalar payload without particle access."""
    if type(config.geometry) is not str or config.geometry not in (
        "spherical",
        "rectangular",
    ):
        raise ValueError(
            "config.geometry must be 'spherical' or 'rectangular'."
        )
    if config.geometry == "spherical":
        if config.chamber_dimensions is not None:
            raise ValueError(
                "spherical config.chamber_dimensions must be None."
            )
        if config.chamber_radius is None:
            raise ValueError("spherical config.chamber_radius is required.")
        _require_positive_real(config.chamber_radius, "config.chamber_radius")
    elif config.chamber_radius is not None:
        raise ValueError("rectangular config.chamber_radius must be None.")
    else:
        dimensions = config.chamber_dimensions
        if type(dimensions) is not tuple or len(dimensions) != 3:
            raise ValueError(
                "rectangular config.chamber_dimensions must be a tuple of "
                "three "
                "positive real scalars."
            )
        for index, dimension in enumerate(dimensions):
            _require_positive_real(
                dimension, f"config.chamber_dimensions[{index}]"
            )


def _validate_charged_field_form(config: NeutralWallLossConfig) -> None:
    """Validate scalar-versus-vector field form without reading Warp fields."""
    _require_finite_real(config.wall_potential, "config.wall_potential")
    field = config.wall_electric_field
    if config.geometry == "spherical" or config.mode == "neutral":
        _require_finite_real(field, "config.wall_electric_field")
    elif not _is_warp_array_like(field):
        raise ValueError(
            "charged rectangular config.wall_electric_field must be a Warp "
            "array."
        )


def _validate_config(config: Any) -> NeutralWallLossConfig:
    """Validate configuration without reading Warp fields or particles."""
    if type(config) is not NeutralWallLossConfig:
        raise TypeError("config must be a NeutralWallLossConfig.")
    if (
        type(config.distribution_type) is not str
        or config.distribution_type != "particle_resolved"
    ):
        raise ValueError(
            "config.distribution_type must be 'particle_resolved'."
        )
    if type(config.mode) is not str or config.mode not in (
        "neutral",
        "charged",
    ):
        raise ValueError("config.mode must be 'neutral' or 'charged'.")
    _require_positive_real(
        config.wall_eddy_diffusivity,
        "config.wall_eddy_diffusivity",
    )
    _validate_geometry(config)
    _validate_charged_field_form(config)
    return config


def _get_required_field(particles: Any, field: str) -> Any:
    """Return one required particle field with a stable schema error."""
    try:
        return getattr(particles, field)
    except AttributeError as exc:
        raise ValueError(f"particles.{field} must be a Warp array.") from exc


def _validate_array_schema(
    values: Any,
    name: str,
    rank: int,
    shape: tuple[int, ...] | None = None,
    device: Any | None = None,
) -> Any:
    """Validate exact Warp metadata without reading or replacing values."""
    if not _is_warp_array_like(values):
        raise ValueError(f"{name} must be a Warp array.")
    if values.dtype != wp.float64:
        raise ValueError(f"{name} must use dtype float64.")
    if values.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} shape must match particle masses.")
    if device is not None and str(values.device) != str(device):
        raise ValueError(f"{name} device must match particle device.")
    return values


def _validate_values(
    values: Any,
    name: str,
    positive: bool = False,
    finite_only: bool = False,
) -> None:
    """Run a device-resident finite-domain scan without host materialization."""
    invalid = wp.zeros(1, dtype=wp.int32, device=values.device)
    if finite_only:
        kernel = _scan_finite_1d if values.ndim == 1 else _scan_finite_2d
    elif positive:
        kernel = _scan_positive_finite_1d
    elif values.ndim == 1:
        kernel = _scan_nonnegative_finite_1d
    elif values.ndim == 2:
        kernel = _scan_nonnegative_finite_2d
    else:
        kernel = _scan_nonnegative_finite_3d
    wp.launch(
        kernel,
        dim=values.shape,
        inputs=[values, invalid],
        device=values.device,
    )
    domain = (
        "finite"
        if finite_only
        else "finite and positive"
        if positive
        else "finite and nonnegative"
    )
    if invalid.numpy()[0] != 0:
        raise ValueError(f"{name} must be {domain}.")


def _validate_particle_schema(particles: Any) -> tuple[int, Any]:
    """Validate fixed particle schema and return its box count and device."""
    masses = _validate_array_schema(
        _get_required_field(particles, "masses"), "particles.masses", 3
    )
    n_boxes, n_particles, n_species = masses.shape
    device = masses.device
    _validate_array_schema(
        _get_required_field(particles, "concentration"),
        "particles.concentration",
        2,
        (n_boxes, n_particles),
        device,
    )
    _validate_array_schema(
        _get_required_field(particles, "charge"),
        "particles.charge",
        2,
        (n_boxes, n_particles),
        device,
    )
    _validate_array_schema(
        _get_required_field(particles, "density"),
        "particles.density",
        1,
        (n_species,),
        device,
    )
    _validate_array_schema(
        _get_required_field(particles, "volume"),
        "particles.volume",
        1,
        (n_boxes,),
        device,
    )
    return n_boxes, device


def _validate_charged_rectangular_field(
    config: NeutralWallLossConfig, device: Any
) -> None:
    """Validate and scan the caller-owned charged rectangular field."""
    if config.mode != "charged" or config.geometry != "rectangular":
        return
    field = _validate_array_schema(
        config.wall_electric_field,
        "config.wall_electric_field",
        1,
        (3,),
        device,
    )
    _validate_values(field, "config.wall_electric_field", finite_only=True)


def _validate_particle_values(particles: Any) -> None:
    """Validate particle values after configuration-dependent schema checks."""
    for values, name, positive, finite_only in (
        (particles.masses, "particles.masses", False, False),
        (particles.concentration, "particles.concentration", False, False),
        (particles.charge, "particles.charge", False, True),
        (particles.density, "particles.density", True, False),
        (particles.volume, "particles.volume", True, False),
    ):
        _validate_values(values, name, positive, finite_only)


def _validate_particles(particles: Any) -> tuple[int, Any]:
    """Validate fixed particle metadata and values in legacy P3 order."""
    n_boxes, device = _validate_particle_schema(particles)
    _validate_particle_values(particles)
    return n_boxes, device


def _validate_time_step(time_step: Any) -> float:
    """Validate a finite nonnegative time step in seconds."""
    if isinstance(time_step, np.ndarray):
        if time_step.ndim != 0 or time_step.dtype.kind not in "iuf":
            raise TypeError("time_step must be a real scalar.")
        time_step = time_step.item()
    if isinstance(time_step, (bool, np.bool_)) or not isinstance(
        time_step, Real
    ):
        raise TypeError("time_step must be a real scalar.")
    try:
        scalar = float(time_step)
    except OverflowError as exc:
        raise ValueError("time_step must be finite and nonnegative.") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError("time_step must be finite and nonnegative.")
    return scalar


def _validate_rng(
    rng_seed: Any,
    rng_states: Any,
    initialize_rng: Any,
    n_boxes: int,
    device: Any,
) -> None:
    """Validate deferred RNG metadata without scanning or consuming state."""
    if isinstance(rng_seed, (bool, np.bool_)) or not isinstance(
        rng_seed, Integral
    ):
        raise TypeError("rng_seed must be an integer.")
    if rng_seed < 0 or rng_seed > 2**32 - 1:
        raise ValueError("rng_seed must be in [0, 2**32 - 1].")
    if not isinstance(initialize_rng, (bool, np.bool_)):
        raise TypeError("initialize_rng must be a boolean.")
    if rng_states is None:
        return
    if not _is_warp_array_like(rng_states):
        raise ValueError("rng_states must be a Warp array.")
    if rng_states.dtype != wp.uint32:
        raise ValueError("rng_states must use dtype uint32.")
    if rng_states.ndim != 1 or rng_states.shape != (n_boxes,):
        raise ValueError("rng_states shape must match expected (n_boxes,).")
    if str(rng_states.device) != str(device):
        raise ValueError("rng_states device must match particle device.")


@no_type_check
@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _cast_float32_to_float64(
    source: wp.array(dtype=wp.float32),
    destination: wp.array(dtype=wp.float64),
) -> None:
    """Copy a private execution environment buffer into float64."""
    index = wp.tid()
    destination[index] = wp.float64(source[index])


def _normalize_execution_environment_array(values: Any, n_boxes: int) -> Any:
    """Return a float64 execution buffer without replacing caller input."""
    if values.dtype == wp.float64:
        return values
    normalized = wp.empty(n_boxes, dtype=wp.float64, device=values.device)
    wp.launch(
        _cast_float32_to_float64,
        dim=n_boxes,
        inputs=[values, normalized],
        device=values.device,
    )
    return normalized


@wp.func
def _should_remove_for_survival_draw(
    draw: Any,
    survival_probability: wp.float64,
) -> bool:
    """Return whether one eligible slot's survival draw selects removal."""
    return wp.float64(draw) >= survival_probability


@no_type_check
@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _initialize_rng_states(seed: Any, rng_states: Any) -> None:
    """Initialize or reset one execution RNG state per particle box."""
    box = wp.tid()
    rng_states[box] = wp.rand_init(wp.int32(seed), wp.int32(box))


@no_type_check
@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _wall_loss_remove(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    time_step: wp.float64,
    wall_eddy_diffusivity: wp.float64,
    chamber_radius: wp.float64,
    chamber_length: wp.float64,
    chamber_width: wp.float64,
    chamber_height: wp.float64,
    geometry_mode: wp.int32,
    n_particles: wp.int32,
    n_species: wp.int32,
    rng_states: wp.array(dtype=wp.uint32),
    rng_seed: wp.int32,
    initialize_rng: wp.int32,
) -> None:
    """Select and clear neutral wall-loss slots in one device launch."""
    box = wp.tid()
    state = rng_states[box]
    has_usable_slot = wp.int32(0)
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature[box],
        _REF_VISCOSITY_AIR_STP,
        _REF_TEMPERATURE_STP,
        _SUTHERLAND_CONSTANT,
    )
    mean_free_path = molecule_mean_free_path_wp(
        _MOLECULAR_WEIGHT_AIR,
        temperature[box],
        pressure[box],
        dynamic_viscosity,
        _GAS_CONSTANT,
    )
    for particle in range(n_particles):
        if concentration[box, particle] <= wp.float64(0.0):
            continue

        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species in range(n_species):
            mass = masses[box, particle, species]
            total_mass += mass
            total_volume += mass / density[species]
        if total_mass <= wp.float64(0.0) or total_volume <= wp.float64(0.0):
            continue

        particle_radius = particle_radius_from_volume_wp(total_volume)
        particle_density = effective_density_wp(total_mass, total_volume)
        if (
            not wp.isfinite(particle_radius)
            or particle_radius <= wp.float64(0.0)
            or not wp.isfinite(particle_density)
            or particle_density <= wp.float64(0.0)
        ):
            continue
        if has_usable_slot == wp.int32(0):
            has_usable_slot = wp.int32(1)
            if initialize_rng != wp.int32(0):
                state = wp.rand_init(rng_seed, box)
        coefficient = wp.float64(0.0)
        if geometry_mode == wp.int32(0):
            coefficient = spherical_wall_loss_coefficient_from_transport_wp(
                wall_eddy_diffusivity,
                particle_radius,
                particle_density,
                temperature[box],
                chamber_radius,
                dynamic_viscosity,
                mean_free_path,
                _BOLTZMANN_CONSTANT,
            )
        else:
            coefficient = rectangle_wall_loss_coefficient_from_transport_wp(
                wall_eddy_diffusivity,
                particle_radius,
                particle_density,
                temperature[box],
                chamber_length,
                chamber_width,
                chamber_height,
                dynamic_viscosity,
                mean_free_path,
                _BOLTZMANN_CONSTANT,
            )
        if wp.isnan(coefficient) or coefficient <= wp.float64(0.0):
            continue
        if wp.isinf(coefficient):
            for species in range(n_species):
                masses[box, particle, species] = wp.float64(0.0)
            concentration[box, particle] = wp.float64(0.0)
            charge[box, particle] = wp.float64(0.0)
            continue

        if _should_remove_for_survival_draw(
            wp.randf(state), wp.exp(-coefficient * time_step)
        ):
            for species in range(n_species):
                masses[box, particle, species] = wp.float64(0.0)
            concentration[box, particle] = wp.float64(0.0)
            charge[box, particle] = wp.float64(0.0)
    if has_usable_slot != wp.int32(0):
        rng_states[box] = state


@no_type_check
@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _charged_spherical_wall_loss_remove(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    time_step: wp.float64,
    wall_eddy_diffusivity: wp.float64,
    chamber_radius: wp.float64,
    wall_potential: wp.float64,
    wall_electric_field: wp.float64,
    n_particles: wp.int32,
    n_species: wp.int32,
    rng_states: wp.array(dtype=wp.uint32),
    rng_seed: wp.int32,
    initialize_rng: wp.int32,
) -> None:
    """Select and clear charged spherical wall-loss slots in one launch."""
    box = wp.tid()
    state = rng_states[box]
    has_draw_eligible_slot = wp.int32(0)
    geometry_scale = _geometry_scale_wp(
        wp.int32(0),
        chamber_radius,
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )
    resolved_field = _resolve_spherical_electric_field_wp(
        wall_electric_field, wall_potential, geometry_scale
    )
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature[box],
        _REF_VISCOSITY_AIR_STP,
        _REF_TEMPERATURE_STP,
        _SUTHERLAND_CONSTANT,
    )
    mean_free_path = molecule_mean_free_path_wp(
        _MOLECULAR_WEIGHT_AIR,
        temperature[box],
        pressure[box],
        dynamic_viscosity,
        _GAS_CONSTANT,
    )
    for particle in range(n_particles):
        if concentration[box, particle] <= wp.float64(0.0):
            continue
        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species in range(n_species):
            mass = masses[box, particle, species]
            total_mass += mass
            total_volume += mass / density[species]
        if total_mass <= wp.float64(0.0) or total_volume <= wp.float64(0.0):
            continue
        particle_radius = particle_radius_from_volume_wp(total_volume)
        particle_density = effective_density_wp(total_mass, total_volume)
        if (
            not wp.isfinite(particle_radius)
            or particle_radius <= wp.float64(0.0)
            or not wp.isfinite(particle_density)
            or particle_density <= wp.float64(0.0)
        ):
            continue
        particle_charge = charge[box, particle]
        if particle_charge == wp.float64(
            0.0
        ) and has_draw_eligible_slot == wp.int32(0):
            has_draw_eligible_slot = wp.int32(1)
            if initialize_rng != wp.int32(0):
                state = wp.rand_init(rng_seed, box)
        neutral_coefficient = spherical_wall_loss_coefficient_from_transport_wp(
            wall_eddy_diffusivity,
            particle_radius,
            particle_density,
            temperature[box],
            chamber_radius,
            dynamic_viscosity,
            mean_free_path,
            _BOLTZMANN_CONSTANT,
        )
        coefficient = neutral_coefficient
        if particle_charge != wp.float64(0.0):
            enhancement = _image_charge_enhancement_wp(
                particle_radius,
                particle_charge,
                temperature[box],
                _ELEMENTARY_CHARGE_VALUE,
                _ELECTRIC_PERMITTIVITY,
                _BOLTZMANN_CONSTANT,
            )
            drift = _electric_field_drift_from_viscosity_wp(
                particle_radius,
                particle_charge,
                resolved_field,
                geometry_scale,
                _ELEMENTARY_CHARGE_VALUE,
                dynamic_viscosity,
            )
            coefficient = _combine_charged_wall_loss_coefficient_wp(
                neutral_coefficient, enhancement, drift
            )
        if wp.isnan(coefficient) or coefficient <= wp.float64(0.0):
            continue
        if has_draw_eligible_slot == wp.int32(0):
            has_draw_eligible_slot = wp.int32(1)
            if initialize_rng != wp.int32(0):
                state = wp.rand_init(rng_seed, box)
        if wp.isinf(coefficient):
            for species in range(n_species):
                masses[box, particle, species] = wp.float64(0.0)
            concentration[box, particle] = wp.float64(0.0)
            charge[box, particle] = wp.float64(0.0)
            continue
        if _should_remove_for_survival_draw(
            wp.randf(state), wp.exp(-coefficient * time_step)
        ):
            for species in range(n_species):
                masses[box, particle, species] = wp.float64(0.0)
            concentration[box, particle] = wp.float64(0.0)
            charge[box, particle] = wp.float64(0.0)
    if has_draw_eligible_slot != wp.int32(0):
        rng_states[box] = state


@no_type_check
@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _charged_rectangular_wall_loss_remove(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    time_step: wp.float64,
    wall_eddy_diffusivity: wp.float64,
    chamber_length: wp.float64,
    chamber_width: wp.float64,
    chamber_height: wp.float64,
    wall_potential: wp.float64,
    wall_electric_field: wp.array(dtype=wp.float64),
    n_particles: wp.int32,
    n_species: wp.int32,
    rng_states: wp.array(dtype=wp.uint32),
    rng_seed: wp.int32,
    initialize_rng: wp.int32,
) -> None:
    """Select and clear charged rectangular wall-loss slots in one launch."""
    box = wp.tid()
    state = rng_states[box]
    has_draw_eligible_slot = wp.int32(0)
    geometry_scale = _geometry_scale_wp(
        wp.int32(1),
        wp.float64(0.0),
        chamber_length,
        chamber_width,
        chamber_height,
    )
    resolved_field = _resolve_rectangular_electric_field_wp(
        wall_electric_field[0],
        wall_electric_field[1],
        wall_electric_field[2],
        wall_potential,
        geometry_scale,
    )
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature[box],
        _REF_VISCOSITY_AIR_STP,
        _REF_TEMPERATURE_STP,
        _SUTHERLAND_CONSTANT,
    )
    mean_free_path = molecule_mean_free_path_wp(
        _MOLECULAR_WEIGHT_AIR,
        temperature[box],
        pressure[box],
        dynamic_viscosity,
        _GAS_CONSTANT,
    )
    for particle in range(n_particles):
        if concentration[box, particle] <= wp.float64(0.0):
            continue
        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species in range(n_species):
            mass = masses[box, particle, species]
            total_mass += mass
            total_volume += mass / density[species]
        if total_mass <= wp.float64(0.0) or total_volume <= wp.float64(0.0):
            continue
        particle_radius = particle_radius_from_volume_wp(total_volume)
        particle_density = effective_density_wp(total_mass, total_volume)
        if (
            not wp.isfinite(particle_radius)
            or particle_radius <= wp.float64(0.0)
            or not wp.isfinite(particle_density)
            or particle_density <= wp.float64(0.0)
        ):
            continue
        particle_charge = charge[box, particle]
        if particle_charge == wp.float64(
            0.0
        ) and has_draw_eligible_slot == wp.int32(0):
            has_draw_eligible_slot = wp.int32(1)
            if initialize_rng != wp.int32(0):
                state = wp.rand_init(rng_seed, box)
        neutral_coefficient = rectangle_wall_loss_coefficient_from_transport_wp(
            wall_eddy_diffusivity,
            particle_radius,
            particle_density,
            temperature[box],
            chamber_length,
            chamber_width,
            chamber_height,
            dynamic_viscosity,
            mean_free_path,
            _BOLTZMANN_CONSTANT,
        )
        coefficient = neutral_coefficient
        if particle_charge != wp.float64(0.0):
            enhancement = _image_charge_enhancement_wp(
                particle_radius,
                particle_charge,
                temperature[box],
                _ELEMENTARY_CHARGE_VALUE,
                _ELECTRIC_PERMITTIVITY,
                _BOLTZMANN_CONSTANT,
            )
            drift = _electric_field_drift_from_viscosity_wp(
                particle_radius,
                particle_charge,
                resolved_field,
                geometry_scale,
                _ELEMENTARY_CHARGE_VALUE,
                dynamic_viscosity,
            )
            coefficient = _combine_charged_wall_loss_coefficient_wp(
                neutral_coefficient, enhancement, drift
            )
        if wp.isnan(coefficient) or coefficient <= wp.float64(0.0):
            continue
        if has_draw_eligible_slot == wp.int32(0):
            has_draw_eligible_slot = wp.int32(1)
            if initialize_rng != wp.int32(0):
                state = wp.rand_init(rng_seed, box)
        if wp.isinf(coefficient):
            for species in range(n_species):
                masses[box, particle, species] = wp.float64(0.0)
            concentration[box, particle] = wp.float64(0.0)
            charge[box, particle] = wp.float64(0.0)
            continue
        if _should_remove_for_survival_draw(
            wp.randf(state), wp.exp(-coefficient * time_step)
        ):
            for species in range(n_species):
                masses[box, particle, species] = wp.float64(0.0)
            concentration[box, particle] = wp.float64(0.0)
            charge[box, particle] = wp.float64(0.0)
    if has_draw_eligible_slot != wp.int32(0):
        rng_states[box] = state


def wall_loss_step_gpu(
    particles: Any,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float | Any,
    *,
    config: NeutralWallLossConfig,
    rng_seed: int = 0,
    rng_states: Any | None = None,
    initialize_rng: bool = False,
    environment: Any | None = None,
) -> Any:
    """Apply direct neutral or charged wall loss to eligible fixed slots.

    The configuration supports particle-resolved neutral and charged wall loss.
    Charged nonzero slots compose private P2 image enhancement and P3 signed
    field drift with their neutral coefficient; zero-charge charged slots use
    the exact neutral coefficient and RNG path. SI inputs are wall eddy
    diffusivity [m^2 s^-1],
    geometry dimensions [m], potential [V], electric field [V m^-1],
    temperature [K], pressure [Pa], and ``time_step`` [s]. A charged rectangular
    electric-field vector is caller-owned read-only storage; only charged
    rectangular execution reads its three lanes.
    After frozen P3 preflight, positive-time calls evaluate neutral coefficients
    for usable slots, apply survival probability ``exp(-k * time_step)``, and
    clear every mass lane, concentration, and charge for removed slots. Zero
    time is a post-preflight write-free no-op.

    Omitted ``rng_states`` uses a private per-call sidecar initialized from
    ``rng_seed`` after successful positive-time preflight. A supplied
    same-device ``uint32`` sidecar with shape ``(n_boxes,)`` remains
    caller-owned, advances in place only for eligible slots, and resets only
    when ``initialize_rng=True``. Eligible slots have positive concentration,
    usable positive mass and derived transport properties, and a finite positive
    wall-loss coefficient. An infinite positive coefficient causes deterministic
    removal without consuming RNG state. Zero time and pre-launch validation
    failures leave supplied state unchanged.
    Rollback is not promised after a mutation kernel has launched.

    Args:
        particles: Caller-owned fixed-shape ``WarpParticleData``-like object.
        temperature: Scalar or per-box Warp temperature [K].
        pressure: Scalar or per-box Warp pressure [Pa].
        time_step: Finite nonnegative duration [s].
        config: Exact concrete geometry, representation, and charged
            configuration. Rectangular fields remain caller-owned storage.
        rng_seed: Unsigned 32-bit seed for private state or an explicit reset.
        rng_states: Optional caller-owned same-device ``uint32`` Warp array
            with shape ``(n_boxes,)``. The array is mutated in place only by
            eligible positive-time work.
        initialize_rng: Whether to reset supplied ``rng_states`` from
            ``rng_seed`` before a successful positive-time call.
        environment: Optional explicit Warp environment source. Cannot be
            combined with direct ``temperature`` or ``pressure`` inputs.

    Returns:
        The identical ``particles`` object. Private RNG state is not returned.

    Raises:
        TypeError: If the configuration, scalar charged input, time step, or
            RNG metadata uses an unsupported type.
        ValueError: If configuration, particle, time-step, environment, or RNG
            values, shapes, dtypes, or devices violate the direct contract,
            including an invalid charged field or combining ``environment``
            with direct environment inputs.
    """
    validated_config = _validate_config(config)
    n_boxes, device = _validate_particle_schema(particles)
    _validate_charged_rectangular_field(validated_config, device)
    _validate_particle_values(particles)
    validated_time_step = _validate_time_step(time_step)
    validate_environment_inputs(
        temperature,
        pressure,
        environment,
        n_boxes,
        device,
        caller_name="wall_loss_step_gpu",
    )
    _validate_rng(rng_seed, rng_states, initialize_rng, n_boxes, device)
    if validated_time_step == 0.0:
        return particles

    temperature_array, pressure_array = _ensure_environment_arrays(
        temperature,
        pressure,
        environment,
        n_boxes,
        device,
        caller_name="wall_loss_step_gpu",
    )
    temperature_array = _normalize_execution_environment_array(
        temperature_array, n_boxes
    )
    pressure_array = _normalize_execution_environment_array(
        pressure_array, n_boxes
    )
    n_particles = particles.masses.shape[1]
    n_species = particles.masses.shape[2]
    execution_rng_states = rng_states
    if execution_rng_states is None:
        execution_rng_states = wp.empty(n_boxes, dtype=wp.uint32, device=device)
        initialize_rng = True
    geometry_mode = 0 if validated_config.geometry == "spherical" else 1
    chamber_radius = float(validated_config.chamber_radius or 0.0)
    dimensions = validated_config.chamber_dimensions or (0.0, 0.0, 0.0)
    if validated_config.mode == "neutral":
        wp.launch(
            _wall_loss_remove,
            dim=n_boxes,
            inputs=[
                particles.masses,
                particles.concentration,
                particles.charge,
                particles.density,
                temperature_array,
                pressure_array,
                validated_time_step,
                float(validated_config.wall_eddy_diffusivity),
                chamber_radius,
                float(dimensions[0]),
                float(dimensions[1]),
                float(dimensions[2]),
                geometry_mode,
                n_particles,
                n_species,
                execution_rng_states,
                int(rng_seed),
                int(initialize_rng),
            ],
            device=device,
        )
    elif validated_config.geometry == "spherical":
        wp.launch(
            _charged_spherical_wall_loss_remove,
            dim=n_boxes,
            inputs=[
                particles.masses,
                particles.concentration,
                particles.charge,
                particles.density,
                temperature_array,
                pressure_array,
                validated_time_step,
                float(validated_config.wall_eddy_diffusivity),
                chamber_radius,
                float(validated_config.wall_potential),
                float(validated_config.wall_electric_field),
                n_particles,
                n_species,
                execution_rng_states,
                int(rng_seed),
                int(initialize_rng),
            ],
            device=device,
        )
    else:
        wp.launch(
            _charged_rectangular_wall_loss_remove,
            dim=n_boxes,
            inputs=[
                particles.masses,
                particles.concentration,
                particles.charge,
                particles.density,
                temperature_array,
                pressure_array,
                validated_time_step,
                float(validated_config.wall_eddy_diffusivity),
                float(dimensions[0]),
                float(dimensions[1]),
                float(dimensions[2]),
                float(validated_config.wall_potential),
                validated_config.wall_electric_field,
                n_particles,
                n_species,
                execution_rng_states,
                int(rng_seed),
                int(initialize_rng),
            ],
            device=device,
        )
    return particles
