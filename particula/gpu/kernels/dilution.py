"""Validate direct GPU dilution inputs without changing caller-owned state.

The dilution coefficient is ``alpha = Q / V`` [s^-1], where ``Q`` is flow
rate and ``V`` is volume. This concrete P1 module freezes input ownership and
validation only. P2 will apply the finite-step update
``c_new = c * exp(-alpha * time_step)``. P1 launches no kernel and returns the
identical particle and gas containers without writing caller-owned state.

Scalar coefficients are normalized to private active-device ``wp.float64``
storage; valid same-device per-box coefficients retain caller ownership and
identity. Validation of values within per-box coefficient arrays and complete
particle/gas container-state preflight are deliberately deferred to P3.
"""

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

    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return scalar


def _normalize_coefficient(
    coefficient: float | Any,
    n_boxes: int,
    device: Any,
) -> Any:
    """Normalize a dilution coefficient into an active-device Warp array.

    A valid per-box Warp array is returned by identity. Scalar inputs allocate
    private ``wp.float64`` storage of shape ``(n_boxes,)`` without caching.
    Per-box coefficient values are intentionally not read in P1.

    Args:
        coefficient: Real scalar [s^-1] or per-box Warp coefficient array.
        n_boxes: Number of simulation boxes.
        device: Active particle-mass device.

    Returns:
        A private scalar broadcast or the original valid Warp array.

    Raises:
        TypeError: If a non-Warp input is not a supported real scalar.
        ValueError: If a scalar domain or Warp-array metadata is invalid.
    """
    if _is_warp_array_like(coefficient):
        warp_coefficient = cast(Any, coefficient)
        if warp_coefficient.dtype != wp.float64:
            raise ValueError("coefficient must use dtype float64.")
        if warp_coefficient.ndim != 1:
            raise ValueError("coefficient must have rank 1.")
        if warp_coefficient.shape != (n_boxes,):
            raise ValueError(
                "coefficient shape must match expected (n_boxes,)."
            )
        if str(warp_coefficient.device) != str(device):
            raise ValueError("coefficient device must match particle device.")
        return warp_coefficient

    scalar = _coerce_nonnegative_real(coefficient, "coefficient")
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


def dilution_step_gpu(
    particles: Any,
    gas: Any,
    coefficient: float | Any,
    time_step: float,
) -> tuple[Any, Any]:
    """Validate P1 GPU dilution inputs and return the identical containers.

    Arguments are positional in the order ``particles``, ``gas``,
    ``coefficient``, and ``time_step``. The coefficient ``alpha = Q / V`` has
    SI units [s^-1], and ``time_step`` is in seconds. P2 will implement the
    finite-step update ``c_new = c * exp(-alpha * time_step)``.

    P1 accepts a finite, nonnegative Python/NumPy real scalar coefficient or a
    caller-owned, active-device ``wp.float64`` array shaped ``(n_boxes,)``.
    Valid per-box arrays are retained by identity; P1 does not scan their
    values, which is deferred to P3. Complete particle/gas state validation is
    also deferred. This phase performs no kernel launch, state preflight,
    allocation beyond private scalar normalization, or caller-state write.
    Thus, zero scalar coefficients, zero time steps, and all valid per-box
    coefficients return the same ``particles`` and ``gas`` objects unchanged.

    Args:
        particles: Particle container whose mass array supplies box/device
            metadata only.
        gas: Gas container returned unchanged.
        coefficient: Dilution coefficient [s^-1], scalar or per-box Warp array.
        time_step: Finite, nonnegative duration [s].

    Returns:
        The identical ``(particles, gas)`` input objects.

    Raises:
        TypeError: If a scalar coefficient or ``time_step`` is not a supported
            real scalar.
        ValueError: If a scalar value is non-finite or negative, or a per-box
            coefficient has invalid dtype, rank, shape, or device metadata.
    """
    if not _is_warp_array_like(coefficient):
        _coerce_nonnegative_real(coefficient, "coefficient")
    _normalize_time_step(time_step)

    masses = particles.masses
    _normalize_coefficient(coefficient, masses.shape[0], masses.device)
    return particles, gas
