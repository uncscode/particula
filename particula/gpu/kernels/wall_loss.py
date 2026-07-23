"""Validate neutral direct GPU wall-loss inputs without executing removal.

This P3 boundary accepts only neutral particle-resolved configurations. It
performs read-only schema and domain preflight and intentionally does not
assemble coefficients, allocate output or RNG storage, advance RNG state, or
mutate caller-owned particle data.  Later wall-loss phases reuse this callable
interface to add the deferred execution behavior.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled by test guards
    raise ImportError(
        "Warp is required for GPU wall-loss helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import (
    _is_warp_array_like,
    validate_environment_inputs,
)


@dataclass(frozen=True)
class NeutralWallLossConfig:
    """Define immutable neutral wall-loss geometry and representation inputs.

    This concrete-module-only configuration is accepted only by the direct P3
    preflight boundary. ``geometry`` selects spherical or rectangular SI
    dimensions, while ``distribution_type`` must remain ``"particle_resolved"``.
    It contains no charged-wall-loss settings.

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
    """Perform write-free P3 neutral wall-loss preflight.

    The configuration supports only particle-resolved neutral wall loss.  SI
    inputs are wall eddy diffusivity [m^2 s^-1], geometry dimensions [m],
    temperature [K], pressure [Pa], and ``time_step`` [s].  This deferred P3
    boundary validates every input but performs no coefficient assembly, RNG
    initialization or advancement, allocation of output resources, or particle
    mutation.  P4 and P5 reuse this exact signature for execution.

    Args:
        particles: Caller-owned fixed-shape ``WarpParticleData``-like object.
        temperature: Scalar or per-box Warp temperature [K].
        pressure: Scalar or per-box Warp pressure [Pa].
        time_step: Finite nonnegative duration [s].
        config: Exact neutral geometry and representation configuration.
        rng_seed: Deferred unsigned 32-bit seed metadata.
        rng_states: Optional deferred per-box unsigned 32-bit state.
        initialize_rng: Deferred request to initialize supplied state.
        environment: Optional explicit Warp environment source.

    Returns:
        The identical ``particles`` object after successful preflight.

    Raises:
        TypeError: If the configuration, time step, RNG metadata, or direct
            environment inputs use an unsupported type.
        ValueError: If configuration, particle, time-step, environment, or RNG
            values, shapes, dtypes, or devices violate the P3 contract.
    """
    _validate_config(config)
    n_boxes, device = _validate_particles(particles)
    _validate_time_step(time_step)
    validate_environment_inputs(
        temperature,
        pressure,
        environment,
        n_boxes,
        device,
        caller_name="wall_loss_step_gpu",
    )
    _validate_rng(rng_seed, rng_states, initialize_rng, n_boxes, device)
    return particles
