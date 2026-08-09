"""Apply prescribed per-box volume evolution to resident Warp containers.

This concrete-only direct kernel accepts caller-owned active-device final
volumes in m^3.  It changes only particle volume and particle and gas
concentrations, preserving their extensive inventories.  It performs no host
transfer, synchronization, resizing, transport, or CPU fallback.
"""

# mypy: disable-error-code="valid-type, misc, operator"

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled by test guards
    raise ImportError(
        "Warp is required for GPU communication helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import _is_warp_array_like


@wp.kernel
def _scan_positive_finite(
    values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
) -> None:
    """Flag values that are not finite and positive."""
    index = wp.tid()
    if not wp.isfinite(values[index]) or values[index] <= 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_nonnegative_finite(
    values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
) -> None:
    """Flag values that are not finite and nonnegative."""
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]) or values[row, column] < 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _validate_factors(
    old_volumes: wp.array(dtype=wp.float64),
    final_volumes: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
) -> None:
    """Validate factors and record whether any volume differs."""
    box = wp.tid()
    factor = old_volumes[box] / final_volumes[box]
    if not wp.isfinite(factor) or factor <= 0.0:
        wp.atomic_max(invalid, 0, 1)
    if old_volumes[box] != final_volumes[box]:
        wp.atomic_max(changed, 0, 1)


@wp.kernel
def _validate_scaled_concentration(
    concentration: wp.array2d(dtype=wp.float64),
    old_volumes: wp.array(dtype=wp.float64),
    final_volumes: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Reject overflowing or positive-to-zero scaled concentrations."""
    box, column = wp.tid()
    old_value = concentration[box, column]
    scaled = old_value * old_volumes[box] / final_volumes[box]
    if not wp.isfinite(scaled) or (old_value > 0.0 and scaled == 0.0):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _apply_volume_evolution(
    volume: wp.array(dtype=wp.float64),
    final_volumes: wp.array(dtype=wp.float64),
) -> None:
    """Write validated final volume values."""
    box = wp.tid()
    volume[box] = final_volumes[box]


@wp.kernel
def _apply_scaled_concentration(
    concentration: wp.array2d(dtype=wp.float64),
    old_volumes: wp.array(dtype=wp.float64),
    final_volumes: wp.array(dtype=wp.float64),
) -> None:
    """Apply validated per-box concentration scaling."""
    box, column = wp.tid()
    concentration[box, column] = (
        concentration[box, column] * old_volumes[box] / final_volumes[box]
    )


def _get_field(container: Any, field: str, name: str) -> Any:
    """Return a required field with a stable malformed-container error."""
    try:
        return getattr(container, field)
    except AttributeError as exc:
        raise ValueError(f"{name} must be a Warp array.") from exc


def _array_range(array: Any, name: str) -> tuple[int, int] | None:
    """Validate pointer-backed contiguous capacity and return its byte range."""
    item_size = np.dtype(
        np.float64 if array.dtype == wp.float64 else np.int32
    ).itemsize
    expected: list[int] = []
    stride = item_size
    for length in reversed(array.shape):
        expected.insert(0, stride)
        stride *= length
    if getattr(array, "strides", None) != tuple(expected):
        raise ValueError(f"{name} must be contiguous.")
    count = int(np.prod(array.shape, dtype=np.int64))
    if count == 0:
        return None
    pointer = getattr(array, "ptr", None)
    capacity = getattr(array, "capacity", None)
    required = count * item_size
    if not isinstance(pointer, Integral) or pointer <= 0:
        raise ValueError(f"{name} must have a valid pointer.")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or capacity < required
    ):
        raise ValueError(f"{name} must have sufficient storage capacity.")
    return int(pointer), int(pointer) + required


def _validate_array(
    value: Any,
    name: str,
    dtype: Any,
    rank: int,
    shape: tuple[int, ...] | None = None,
    device: Any | None = None,
) -> tuple[Any, tuple[int, int] | None]:
    """Validate one primary array's fixed schema and memory backing."""
    if not _is_warp_array_like(value):
        raise ValueError(f"{name} must be a Warp array.")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}.")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} shape must match particle masses.")
    if device is not None and str(value.device) != str(device):
        raise ValueError(f"{name} device must match particle device.")
    return value, _array_range(value, name)


def _reject_aliases(
    arrays: tuple[Any, ...], ranges: tuple[tuple[int, int] | None, ...]
) -> None:
    """Reject shared identities and nonempty overlapping primary storage."""
    for index, value in enumerate(arrays):
        for previous in range(index):
            left, right = ranges[index], ranges[previous]
            if value is arrays[previous] or (
                left is not None
                and right is not None
                and left[0] < right[1]
                and right[0] < left[1]
            ):
                raise ValueError("communication primary arrays must not alias.")


def _scan_1d(values: Any, kernel: Any, message: str) -> None:
    """Run a read-only one-dimensional validation scan."""
    if values.shape[0] == 0:
        return
    invalid = wp.zeros(1, dtype=wp.int32, device=values.device)
    wp.launch(
        kernel,
        dim=values.shape[0],
        inputs=[values, invalid],
        device=values.device,
    )
    if int(invalid.numpy()[0]):
        raise ValueError(message)


def _scan_2d(
    values: Any, old_volumes: Any, final_volumes: Any, product: bool
) -> None:
    """Run a read-only concentration domain or scaled-product scan."""
    if values.shape[0] == 0 or values.shape[1] == 0:
        return
    invalid = wp.zeros(1, dtype=wp.int32, device=values.device)
    if product:
        wp.launch(
            _validate_scaled_concentration,
            dim=values.shape,
            inputs=[values, old_volumes, final_volumes, invalid],
            device=values.device,
        )
    else:
        wp.launch(
            _scan_nonnegative_finite,
            dim=values.shape,
            inputs=[values, invalid],
            device=values.device,
        )
    if int(invalid.numpy()[0]):
        message = (
            "scaled concentration underflow or overflow."
            if product
            else "concentration must be finite and nonnegative."
        )
        raise ValueError(message)


def volume_evolution_step_gpu(
    particles: Any, gas: Any, final_volumes: object
) -> tuple[Any, Any]:
    """Apply prescribed final volumes while preserving extensive inventories.

    ``final_volumes`` must be a caller-owned, contiguous active-device
    ``wp.float64`` array shaped ``(B,)`` in m^3.  Complete read-only preflight
    validates schemas, ownership, domains, factors, and proposed products before
    mutation.  Successful calls update only ``particles.volume`` and particle
    and gas concentrations by ``old_volume / final_volume`` and return the exact
    input containers.  Equal final volumes are a write-free no-op. Callers own
    synchronization; rollback is not promised after an apply writer launches.
    """
    # Final-volume form intentionally precedes all container access.
    final_volumes, final_range = _validate_array(
        final_volumes, "final_volumes", wp.float64, 1
    )
    masses, mass_range = _validate_array(
        _get_field(particles, "masses", "particles.masses"),
        "particles.masses",
        wp.float64,
        3,
    )
    n_boxes, n_particles, n_species = masses.shape
    device = masses.device
    particle_concentration, particle_range = _validate_array(
        _get_field(particles, "concentration", "particles.concentration"),
        "particles.concentration",
        wp.float64,
        2,
        (n_boxes, n_particles),
        device,
    )
    charge, charge_range = _validate_array(
        _get_field(particles, "charge", "particles.charge"),
        "particles.charge",
        wp.float64,
        2,
        (n_boxes, n_particles),
        device,
    )
    density, density_range = _validate_array(
        _get_field(particles, "density", "particles.density"),
        "particles.density",
        wp.float64,
        1,
        (n_species,),
        device,
    )
    volume, volume_range = _validate_array(
        _get_field(particles, "volume", "particles.volume"),
        "particles.volume",
        wp.float64,
        1,
        (n_boxes,),
        device,
    )
    gas_concentration, gas_concentration_range = _validate_array(
        _get_field(gas, "concentration", "gas.concentration"),
        "gas.concentration",
        wp.float64,
        2,
        device=device,
    )
    n_gas = gas_concentration.shape[1]
    if gas_concentration.shape[0] != n_boxes:
        raise ValueError("gas.concentration shape must match particle masses.")
    molar_mass, molar_mass_range = _validate_array(
        _get_field(gas, "molar_mass", "gas.molar_mass"),
        "gas.molar_mass",
        wp.float64,
        1,
        (n_gas,),
        device,
    )
    vapor_pressure, vapor_pressure_range = _validate_array(
        _get_field(gas, "vapor_pressure", "gas.vapor_pressure"),
        "gas.vapor_pressure",
        wp.float64,
        2,
        (n_boxes, n_gas),
        device,
    )
    partitioning, partitioning_range = _validate_array(
        _get_field(gas, "partitioning", "gas.partitioning"),
        "gas.partitioning",
        wp.int32,
        2,
        (n_boxes, n_gas),
        device,
    )
    if final_volumes.shape != (n_boxes,):
        raise ValueError("final_volumes shape must match particle masses.")
    if str(final_volumes.device) != str(device):
        raise ValueError("final_volumes device must match particle device.")
    _reject_aliases(
        (
            masses,
            particle_concentration,
            charge,
            density,
            volume,
            gas_concentration,
            molar_mass,
            vapor_pressure,
            partitioning,
            final_volumes,
        ),
        (
            mass_range,
            particle_range,
            charge_range,
            density_range,
            volume_range,
            gas_concentration_range,
            molar_mass_range,
            vapor_pressure_range,
            partitioning_range,
            final_range,
        ),
    )
    _scan_1d(
        volume,
        _scan_positive_finite,
        "particles.volume must be finite positive.",
    )
    _scan_1d(
        final_volumes,
        _scan_positive_finite,
        "final_volumes must be finite positive.",
    )
    _scan_2d(particle_concentration, volume, final_volumes, False)
    _scan_2d(gas_concentration, volume, final_volumes, False)
    if n_boxes == 0:
        return particles, gas
    factor_invalid = wp.zeros(1, dtype=wp.int32, device=device)
    changed = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _validate_factors,
        dim=n_boxes,
        inputs=[volume, final_volumes, factor_invalid, changed],
        device=device,
    )
    if int(factor_invalid.numpy()[0]):
        raise ValueError("volume evolution factor must be finite positive.")
    _scan_2d(particle_concentration, volume, final_volumes, True)
    _scan_2d(gas_concentration, volume, final_volumes, True)
    if not int(changed.numpy()[0]):
        return particles, gas
    if n_particles:
        wp.launch(
            _apply_scaled_concentration,
            dim=(n_boxes, n_particles),
            inputs=[particle_concentration, volume, final_volumes],
            device=device,
        )
    if n_gas:
        wp.launch(
            _apply_scaled_concentration,
            dim=(n_boxes, n_gas),
            inputs=[gas_concentration, volume, final_volumes],
            device=device,
        )
    wp.launch(
        _apply_volume_evolution,
        dim=n_boxes,
        inputs=[volume, final_volumes],
        device=device,
    )
    return particles, gas
