"""Apply bounded neutral direct GPU wall loss to fixed particle slots.

The P4 boundary retains P3 preflight ordering, then evaluates neutral wall-loss
coefficients for usable particle-resolved slots. Each eligible slot receives one
call-local deterministic survival draw and removed slots are cleared in place.
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
    rectangle_wall_loss_coefficient_wp,
    spherical_wall_loss_coefficient_wp,
)
from particula.gpu.kernels.environment import (
    _ensure_environment_arrays,
    _is_warp_array_like,
    validate_environment_inputs,
)
from particula.gpu.properties import (
    effective_density_wp,
    particle_radius_from_volume_wp,
)
from particula.util.constants import (
    BOLTZMANN_CONSTANT,
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


@dataclass(frozen=True)
class NeutralWallLossConfig:
    """Define immutable neutral wall-loss geometry and representation inputs.

    This concrete-module-only configuration is accepted by the direct P4
    boundary, which retains frozen P3 preflight. ``geometry`` selects spherical
    or rectangular SI dimensions, while ``distribution_type`` must remain
    ``"particle_resolved"``. It contains no charged-wall-loss settings.

    Attributes:
        geometry: Exact chamber geometry, ``"spherical"`` or ``"rectangular"``.
        wall_eddy_diffusivity: Positive wall eddy diffusivity [m^2 s^-1].
        chamber_radius: Positive spherical chamber radius [m]; required only
            for spherical geometry.
        chamber_dimensions: Tuple of three positive rectangular chamber
            dimensions [m]; required only for rectangular geometry.
        distribution_type: Required ``"particle_resolved"`` representation
            selector.
    """

    geometry: str
    wall_eddy_diffusivity: float
    chamber_radius: float | None = None
    chamber_dimensions: tuple[float, float, float] | None = None
    distribution_type: str = "particle_resolved"


@wp.kernel
def _scan_nonnegative_finite_1d(
    values: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid entries in one-dimensional nonnegative data."""
    index = wp.tid()
    if not wp.isfinite(values[index]) or values[index] < 0.0:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _scan_nonnegative_finite_2d(
    values: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid entries in two-dimensional nonnegative data."""
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]) or values[row, column] < 0.0:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _scan_finite_2d(
    values: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record non-finite entries in two-dimensional metadata."""
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
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


@wp.kernel
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
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return scalar


def _validate_config(config: Any) -> NeutralWallLossConfig:
    """Validate immutable neutral geometry payload without coercing it."""
    if type(config) is not NeutralWallLossConfig:
        raise TypeError("config must be a NeutralWallLossConfig.")
    if config.geometry not in ("spherical", "rectangular"):
        raise ValueError(
            "config.geometry must be 'spherical' or 'rectangular'."
        )
    if config.distribution_type != "particle_resolved":
        raise ValueError(
            "config.distribution_type must be 'particle_resolved'."
        )
    _require_positive_real(
        config.wall_eddy_diffusivity,
        "config.wall_eddy_diffusivity",
    )

    if config.geometry == "spherical":
        if config.chamber_dimensions is not None:
            raise ValueError(
                "spherical config.chamber_dimensions must be None."
            )
        if config.chamber_radius is None:
            raise ValueError("spherical config.chamber_radius is required.")
        _require_positive_real(config.chamber_radius, "config.chamber_radius")
        return config

    if config.chamber_radius is not None:
        raise ValueError("rectangular config.chamber_radius must be None.")
    dimensions = config.chamber_dimensions
    if type(dimensions) is not tuple or len(dimensions) != 3:
        raise ValueError(
            "rectangular config.chamber_dimensions must be a tuple of three "
            "positive real scalars."
        )
    for index, dimension in enumerate(dimensions):
        _require_positive_real(dimension, f"config.chamber_dimensions[{index}]")
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
        kernel = _scan_finite_2d
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


def _validate_particles(particles: Any) -> tuple[int, Any]:
    """Validate fixed particle metadata and fields in P3 preflight order."""
    masses = _validate_array_schema(
        _get_required_field(particles, "masses"), "particles.masses", 3
    )
    n_boxes, n_particles, n_species = masses.shape
    device = masses.device
    concentration = _validate_array_schema(
        _get_required_field(particles, "concentration"),
        "particles.concentration",
        2,
        (n_boxes, n_particles),
        device,
    )
    charge = _validate_array_schema(
        _get_required_field(particles, "charge"),
        "particles.charge",
        2,
        (n_boxes, n_particles),
        device,
    )
    density = _validate_array_schema(
        _get_required_field(particles, "density"),
        "particles.density",
        1,
        (n_species,),
        device,
    )
    volume = _validate_array_schema(
        _get_required_field(particles, "volume"),
        "particles.volume",
        1,
        (n_boxes,),
        device,
    )
    for values, name, positive, finite_only in (
        (masses, "particles.masses", False, False),
        (concentration, "particles.concentration", False, False),
        (charge, "particles.charge", False, True),
        (density, "particles.density", True, False),
        (volume, "particles.volume", True, False),
    ):
        _validate_values(values, name, positive, finite_only)
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
    scalar = float(time_step)
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
@wp.kernel
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
@wp.kernel
def _wall_loss_removal_mask(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
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
    rng_seed: wp.uint32,
    removal_mask: wp.array2d(dtype=wp.int32),
) -> None:
    """Calculate stochastic removal flags without mutating caller fields."""
    box, particle = wp.tid()
    if concentration[box, particle] <= wp.float64(0.0):
        return

    total_mass = wp.float64(0.0)
    total_volume = wp.float64(0.0)
    for species in range(n_species):
        mass = masses[box, particle, species]
        total_mass += mass
        total_volume += mass / density[species]
    if total_mass <= wp.float64(0.0) or total_volume <= wp.float64(0.0):
        return

    particle_radius = particle_radius_from_volume_wp(total_volume)
    particle_density = effective_density_wp(total_mass, total_volume)
    # Invalid derived transport values survive without coefficient or RNG work.
    if (
        not wp.isfinite(particle_radius)
        or particle_radius <= wp.float64(0.0)
        or not wp.isfinite(particle_density)
        or particle_density <= wp.float64(0.0)
    ):
        return

    coefficient = wp.float64(0.0)
    if geometry_mode == wp.int32(0):
        coefficient = spherical_wall_loss_coefficient_wp(
            wall_eddy_diffusivity,
            particle_radius,
            particle_density,
            temperature[box],
            pressure[box],
            chamber_radius,
            _BOLTZMANN_CONSTANT,
            _GAS_CONSTANT,
            _MOLECULAR_WEIGHT_AIR,
            _REF_VISCOSITY_AIR_STP,
            _REF_TEMPERATURE_STP,
            _SUTHERLAND_CONSTANT,
        )
    else:
        coefficient = rectangle_wall_loss_coefficient_wp(
            wall_eddy_diffusivity,
            particle_radius,
            particle_density,
            temperature[box],
            pressure[box],
            chamber_length,
            chamber_width,
            chamber_height,
            _BOLTZMANN_CONSTANT,
            _GAS_CONSTANT,
            _MOLECULAR_WEIGHT_AIR,
            _REF_VISCOSITY_AIR_STP,
            _REF_TEMPERATURE_STP,
            _SUTHERLAND_CONSTANT,
        )
    if not wp.isfinite(coefficient) or coefficient <= wp.float64(0.0):
        return

    # The seed and flattened fixed slot define one call-local random stream.
    state = wp.rand_init(
        wp.int32(rng_seed), wp.int32(box * n_particles + particle)
    )
    if _should_remove_for_survival_draw(
        wp.randf(state), wp.exp(-coefficient * time_step)
    ):
        removal_mask[box, particle] = wp.int32(1)


@wp.kernel
def _apply_wall_loss_mask(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    removal_mask: wp.array2d(dtype=wp.int32),
    n_species: wp.int32,
) -> None:
    """Clear all mutable slot fields selected by a removal mask."""
    box, particle = wp.tid()
    if removal_mask[box, particle] != wp.int32(0):
        for species in range(n_species):
            masses[box, particle, species] = wp.float64(0.0)
        concentration[box, particle] = wp.float64(0.0)
        charge[box, particle] = wp.float64(0.0)


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
    """Apply P4 neutral wall loss to eligible particle-resolved slots.

    The configuration supports only particle-resolved neutral wall loss.  SI
    inputs are wall eddy diffusivity [m^2 s^-1], geometry dimensions [m],
    temperature [K], pressure [Pa], and ``time_step`` [s]. After frozen P3
    preflight, positive-time calls evaluate neutral coefficients for usable
    slots, apply survival probability ``exp(-k * time_step)``, and clear every
    mass lane, concentration, and charge for removed slots. Zero time is a
    post-preflight write-free no-op.

    Supplied ``rng_states`` is neither initialized nor mutated. Eligible slots
    have positive concentration, positive usable mass and derived transport
    properties, and a finite positive wall-loss coefficient. P4 uses one
    call-local seed-plus-slot draw per eligible slot; P5 owns persistent RNG
    lifecycle semantics. Pre-launch validation errors are atomic, while rollback
    is not promised after a mutation kernel has launched.

    Args:
        particles: Caller-owned fixed-shape ``WarpParticleData``-like object.
        temperature: Scalar or per-box Warp temperature [K].
        pressure: Scalar or per-box Warp pressure [Pa].
        time_step: Finite nonnegative duration [s].
        config: Exact neutral geometry and representation configuration.
        rng_seed: Unsigned 32-bit seed for call-local slot-derived draws.
        rng_states: Optional P3-validated, unchanged per-box unsigned state.
        initialize_rng: P3-validated deferred P5 lifecycle request.
        environment: Optional explicit Warp environment source.

    Returns:
        The identical ``particles`` object after successful execution.

    Raises:
        TypeError: If the configuration, time step, RNG metadata, or direct
            environment inputs use an unsupported type.
        ValueError: If configuration, particle, time-step, environment, or RNG
            values, shapes, dtypes, or devices violate the P3/P4 contract.
    """
    validated_config = _validate_config(config)
    n_boxes, device = _validate_particles(particles)
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
    removal_mask = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    geometry_mode = 0 if validated_config.geometry == "spherical" else 1
    chamber_radius = float(validated_config.chamber_radius or 0.0)
    dimensions = validated_config.chamber_dimensions or (0.0, 0.0, 0.0)
    wp.launch(
        _wall_loss_removal_mask,
        dim=(n_boxes, n_particles),
        inputs=[
            particles.masses,
            particles.concentration,
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
            int(rng_seed),
            removal_mask,
        ],
        device=device,
    )
    wp.launch(
        _apply_wall_loss_mask,
        dim=(n_boxes, n_particles),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.charge,
            removal_mask,
            n_species,
        ],
        device=device,
    )
    return particles
