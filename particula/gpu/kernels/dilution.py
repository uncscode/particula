"""Apply fixed-shape direct GPU dilution to caller-owned concentrations.

The dilution coefficient is ``alpha = Q / V`` [s^-1], where ``Q`` is flow
rate and ``V`` is volume. This concrete P2 module applies the finite-step
update ``c_new = c * exp(-alpha * time_step)`` in place to particle and gas
concentrations, while returning the identical containers.

Scalar coefficients are normalized to private active-device ``wp.float64``
storage only after complete read-only preflight; valid same-device per-box
coefficients retain caller ownership and identity. Zero scalar coefficients and
zero time steps are write-free no-ops after that preflight. Rollback after a
successfully launched kernel failure is not promised.
"""

# mypy: disable-error-code="valid-type, misc"

from __future__ import annotations

from numbers import Real
from typing import Any, cast

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled via import guards
    raise ImportError(
        "Warp is required for GPU dilution helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import _is_warp_array_like


@wp.kernel
def _dilution_factors(
    coefficient: wp.array(dtype=wp.float64),
    time_step: wp.float64,
    factors: wp.array(dtype=wp.float64),
) -> None:
    """Calculate one dilution factor for each simulation box."""
    box = wp.tid()
    factors[box] = wp.exp(-coefficient[box] * time_step)


@wp.kernel
def _apply_particle_dilution(
    concentration: wp.array2d(dtype=wp.float64),
    coefficient: wp.array(dtype=wp.float64),
    factors: wp.array(dtype=wp.float64),
) -> None:
    """Apply precomputed per-box dilution factors to particle concentration."""
    box, particle = wp.tid()
    if coefficient[box] != 0.0:
        concentration[box, particle] = (
            concentration[box, particle] * factors[box]
        )


@wp.kernel
def _apply_gas_dilution(
    concentration: wp.array2d(dtype=wp.float64),
    coefficient: wp.array(dtype=wp.float64),
    factors: wp.array(dtype=wp.float64),
) -> None:
    """Apply precomputed per-box dilution factors to gas concentration."""
    box, species = wp.tid()
    if coefficient[box] != 0.0:
        concentration[box, species] = concentration[box, species] * factors[box]


def _coerce_nonnegative_real(value: Any, name: str) -> float:
    """Return one supported finite, nonnegative real scalar.

    Args:
        value: Candidate Python or NumPy scalar.
        name: Input label used in stable validation messages.

    Returns:
        The scalar as a Python float.

    Raises:
        TypeError: If ``value`` is not a supported real scalar form.
        ValueError: If ``value`` is non-finite or negative.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be a real scalar.")
        value = value.item()

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")

    try:
        scalar = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and nonnegative.") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return scalar


def _validate_warp_coefficient_form(coefficient: Any) -> Any:
    """Validate Warp coefficient metadata independent of particle metadata.

    Shape and device compatibility require particle metadata and are checked by
    ``_normalize_coefficient`` after time-step validation. Dtype and rank are
    checked first so malformed coefficient inputs fail before time or container
    access.
    """
    warp_coefficient = cast(Any, coefficient)
    if warp_coefficient.dtype != wp.float64:
        raise ValueError("coefficient must use dtype float64.")
    if warp_coefficient.ndim != 1:
        raise ValueError("coefficient must have rank 1.")
    return warp_coefficient


def _normalize_coefficient(
    coefficient: float | Any,
    n_boxes: int,
    device: Any,
) -> Any:
    """Normalize an already validated coefficient into a Warp array.

    A valid per-box Warp array is returned by identity. Scalar inputs allocate
    private ``wp.float64`` storage of shape ``(n_boxes,)`` without caching.

    Args:
        coefficient: Real scalar [s^-1] or per-box Warp coefficient array.
        n_boxes: Number of simulation boxes.
        device: Active particle-mass device.

    Returns:
        A private scalar broadcast or the original valid Warp array.

    Raises:
        This helper assumes the public entry point has completed all input
        validation. It performs allocation only for valid scalar inputs.
    """
    if _is_warp_array_like(coefficient):
        return coefficient

    scalar = cast(float, coefficient)
    return wp.full(
        n_boxes,
        wp.float64(scalar),
        dtype=wp.float64,
        device=device,
    )


def _normalize_time_step(time_step: float | Any) -> float:
    """Validate and return a finite, nonnegative time step in seconds.

    Args:
        time_step: Candidate Python or NumPy scalar duration [s].

    Returns:
        The duration as a Python float.

    Raises:
        TypeError: If ``time_step`` is not a supported real scalar.
        ValueError: If ``time_step`` is non-finite or negative.
    """
    return _coerce_nonnegative_real(time_step, "time_step")


def _get_required_field(container: Any, field: str, name: str) -> Any:
    """Return a required container field with a stable schema failure."""
    try:
        return getattr(container, field)
    except AttributeError as exc:
        raise ValueError(f"{name} must be a Warp array.") from exc


def _validate_mass_schema(particles: Any) -> tuple[Any, int, int, int, Any]:
    """Validate masses before deriving authoritative dimensions and device."""
    masses = _get_required_field(particles, "masses", "particles.masses")
    if not _is_warp_array_like(masses):
        raise ValueError("particles.masses must be a Warp array.")
    if masses.dtype != wp.float64:
        raise ValueError("particles.masses must use dtype float64.")
    if masses.ndim != 3:
        raise ValueError("particles.masses must have rank 3.")
    n_boxes, n_particles, n_species = masses.shape
    return masses, n_boxes, n_particles, n_species, masses.device


def _validate_nonnegative_finite_values(values: Any, name: str) -> None:
    """Read Warp values safely and reject non-finite or negative entries."""
    if getattr(values.device, "is_cpu", False) or str(values.device).startswith(
        "cpu"
    ):
        values_np = np.asarray(values.numpy(), dtype=np.float64)
    else:
        values_np = np.asarray(type(values).numpy(values), dtype=np.float64)
    if not np.all(np.isfinite(values_np)) or np.any(values_np < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative.")


def _validate_per_box_coefficient(
    coefficient: Any,
    n_boxes: int,
    device: Any,
) -> Any:
    """Validate exact same-device coefficient schema and physical values."""
    if coefficient.shape != (n_boxes,):
        raise ValueError("coefficient shape must match expected (n_boxes,).")
    if str(coefficient.device) != str(device):
        raise ValueError("coefficient device must match particle device.")
    _validate_nonnegative_finite_values(coefficient, "coefficient")
    return coefficient


def _validate_concentration_schema(
    values: Any,
    name: str,
    expected_shape: tuple[int, int],
    device: Any,
) -> None:
    """Validate exact concentration schema and physical values read-only."""
    if not _is_warp_array_like(values):
        raise ValueError(f"{name} must be a Warp array.")
    if values.ndim != 2:
        raise ValueError(f"{name} must have rank 2.")
    if values.dtype != wp.float64:
        raise ValueError(f"{name} must use dtype float64.")
    if str(values.device) != str(device):
        raise ValueError(f"{name} device must match particle device.")
    if values.shape[0] != expected_shape[0]:
        raise ValueError(f"{name} box dimension must match particle masses.")
    if values.shape != expected_shape:
        raise ValueError(f"{name} shape must match particle masses.")
    _validate_nonnegative_finite_values(values, name)


def dilution_step_gpu(
    particles: Any,
    gas: Any,
    coefficient: float | Any,
    time_step: float,
) -> tuple[Any, Any]:
    """Apply GPU dilution after complete atomic entry-point preflight.

    Arguments are positional in the order ``particles``, ``gas``,
    ``coefficient``, and ``time_step``. The coefficient ``alpha = Q / V`` has
    SI units [s^-1], and ``time_step`` is in seconds. This function applies the
    finite-step update ``c_new = c * exp(-alpha * time_step)`` in place.

    P3 retains P1's coefficient-input contract: a finite, nonnegative
    Python/NumPy real scalar or a caller-owned, active-device ``wp.float64``
    array shaped ``(n_boxes,)``. Valid per-box arrays are retained by identity
    and their values are read only during preflight. Masses must be a
    same-device ``wp.float64`` rank-3 array; particle and gas concentrations
    must be same-device ``wp.float64`` rank-2 arrays with exact mass-derived
    shapes. Coefficients and concentrations must be finite and nonnegative.
    Only the two concentration arrays are mutated; masses and all other
    caller-owned fields are preserved.

    Every rejected call completes no allocation, kernel launch, or mutation.
    A zero scalar coefficient or zero time step returns only after full
    preflight, without private allocation or a launch. Rollback after a
    successfully launched kernel failure is not promised.

    Args:
        particles: Particle container with fixed-shape concentration storage.
        gas: Gas container with fixed-shape concentration storage.
        coefficient: Dilution coefficient [s^-1], scalar or per-box Warp array.
        time_step: Finite, nonnegative duration [s].

    Returns:
        The identical ``(particles, gas)`` input objects after in-place decay.

    Raises:
        TypeError: If a scalar coefficient or ``time_step`` is not a supported
            real scalar.
        ValueError: If a schema, shape, device, or physical domain is invalid.
    """
    is_per_box_coefficient = _is_warp_array_like(coefficient)
    if is_per_box_coefficient:
        validated_coefficient = _validate_warp_coefficient_form(coefficient)
    else:
        scalar_coefficient = _coerce_nonnegative_real(
            coefficient, "coefficient"
        )
    normalized_time_step = _normalize_time_step(time_step)

    _, n_boxes, n_particles, n_species, device = _validate_mass_schema(
        particles
    )
    if is_per_box_coefficient:
        normalized_coefficient = _validate_per_box_coefficient(
            validated_coefficient,
            n_boxes,
            device,
        )

    particle_concentration = _get_required_field(
        particles,
        "concentration",
        "particles.concentration",
    )
    _validate_concentration_schema(
        particle_concentration,
        "particles.concentration",
        (n_boxes, n_particles),
        device,
    )
    gas_concentration = _get_required_field(
        gas,
        "concentration",
        "gas.concentration",
    )
    _validate_concentration_schema(
        gas_concentration,
        "gas.concentration",
        (n_boxes, n_species),
        device,
    )

    if (
        n_boxes == 0
        or normalized_time_step == 0.0
        or (not is_per_box_coefficient and scalar_coefficient == 0.0)
    ):
        return particles, gas

    if not is_per_box_coefficient:
        normalized_coefficient = _normalize_coefficient(
            scalar_coefficient,
            n_boxes,
            device,
        )

    factors = wp.empty(n_boxes, dtype=wp.float64, device=device)
    wp.launch(
        _dilution_factors,
        dim=n_boxes,
        inputs=[normalized_coefficient, normalized_time_step, factors],
        device=device,
    )
    if n_particles > 0:
        wp.launch(
            _apply_particle_dilution,
            dim=(n_boxes, n_particles),
            inputs=[particle_concentration, normalized_coefficient, factors],
            device=device,
        )
    if n_species > 0:
        wp.launch(
            _apply_gas_dilution,
            dim=(n_boxes, n_species),
            inputs=[gas_concentration, normalized_coefficient, factors],
            device=device,
        )
    return particles, gas
