"""Apply direct-Warp resident volume evolution and gas communication.

Concrete-only kernels preserve extensive inventories during prescribed
per-box volume changes and synchronously transport gas using caller-owned
device-resident ledgers. They perform no hidden host transfer,
synchronization, resizing, conversion, or CPU fallback.
"""

# mypy: disable-error-code="valid-type, misc, operator"

from __future__ import annotations

import sys
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled by test guards
    raise ImportError(
        "Warp is required for GPU communication helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationResourceShape,
    CommunicationShapeKind,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
)
from particula.gpu.kernels.environment import _is_warp_array_like


@dataclass(frozen=True, eq=False)
class GasCommunicationBuffers:
    """Caller-owned gas-transport work and open-boundary ledgers.

    All arrays are contiguous active-device ``wp.float64`` arrays shaped
    ``(B, S)``. ``amounts``, ``amount_deltas``, and ``outbound_amounts`` are
    required work storage. ``source_amounts`` and ``sink_amounts`` are the
    optional accounting ledgers required for enabled maps containing,
    respectively, a ``-1`` source or destination endpoint. The communication
    step overwrites applicable work and accounting ledgers; callers retain
    ownership and synchronize before inspection.

    Attributes:
        amounts: Immutable per-step gas amount ledger.
        amount_deltas: Net in-domain amount change for each box and species.
        outbound_amounts: Aggregate amount debited from each in-domain source.
        source_amounts: Optional positive amount admitted at each destination
            from declared open sources.
        sink_amounts: Optional positive amount removed from each source by
            declared open sinks.
    """

    amounts: object
    amount_deltas: object
    outbound_amounts: object
    source_amounts: object | None = None
    sink_amounts: object | None = None


@wp.kernel
def _communication_validate_primaries(
    volume: wp.array(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Flag invalid volume or gas concentration values."""
    box, species = wp.tid()
    if (
        not wp.isfinite(volume[box])
        or volume[box] <= 0.0
        or not wp.isfinite(concentration[box, species])
        or concentration[box, species] < 0.0
    ):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _communication_validate_final_volumes(
    final_volumes: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Flag invalid read-only prescribed-volume metadata."""
    box = wp.tid()
    if not wp.isfinite(final_volumes[box]) or final_volumes[box] <= 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _communication_validate_map(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    rates: wp.array(dtype=wp.float64),
    boxes: int,
    one_dimensional: int,
    has_source_ledger: int,
    has_sink_ledger: int,
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate enabled edges, including the P3 ``-1`` boundary sentinel."""
    edge = wp.tid()
    if enabled[edge] != 0 and enabled[edge] != 1:
        wp.atomic_max(invalid, 0, 1)
    if not wp.isfinite(rates[edge]) or rates[edge] < 0.0:
        wp.atomic_max(invalid, 0, 1)
    if enabled[edge] == 1:
        wp.atomic_max(active, 0, 1)
        left = source[edge]
        right = destination[edge]
        source_open = left == -1
        sink_open = right == -1
        if (
            left < -1
            or right < -1
            or left >= boxes
            or right >= boxes
            or (source_open and sink_open)
            or (source_open and has_source_ledger == 0)
            or (sink_open and has_sink_ledger == 0)
            or (not source_open and not sink_open and left == right)
            or (
                one_dimensional == 1
                and not source_open
                and not sink_open
                and wp.abs(left - right) != 1
            )
        ):
            wp.atomic_max(invalid, 0, 1)
        if not source_open and not sink_open:
            for previous in range(edge):
                if (
                    enabled[previous] == 1
                    and source[previous] == left
                    and destination[previous] == right
                ):
                    wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _communication_clear_and_stage(
    concentration: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    amounts: wp.array2d(dtype=wp.float64),
    deltas: wp.array2d(dtype=wp.float64),
    outbound: wp.array2d(dtype=wp.float64),
    source_ledger: wp.array2d(dtype=wp.float64),
    sink_ledger: wp.array2d(dtype=wp.float64),
    has_source_ledger: int,
    has_sink_ledger: int,
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Stage immutable extensive amounts and clear documented work ledgers."""
    box, species = wp.tid()
    if invalid[0] == 0 and active[0] != 0:
        amount = concentration[box, species] * volume[box]
        if not wp.isfinite(amount):
            wp.atomic_max(invalid, 0, 1)
        amounts[box, species] = amount
        deltas[box, species] = wp.float64(0.0)
        outbound[box, species] = wp.float64(0.0)
        if has_source_ledger != 0:
            source_ledger[box, species] = wp.float64(0.0)
        if has_sink_ledger != 0:
            sink_ledger[box, species] = wp.float64(0.0)


@wp.kernel
def _communication_propose(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    rates: wp.array(dtype=wp.float64),
    time_step: wp.float64,
    amounts: wp.array2d(dtype=wp.float64),
    deltas: wp.array2d(dtype=wp.float64),
    outbound: wp.array2d(dtype=wp.float64),
    source_ledger: wp.array2d(dtype=wp.float64),
    sink_ledger: wp.array2d(dtype=wp.float64),
    has_source_ledger: int,
    has_sink_ledger: int,
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Aggregate synchronous transfers from the immutable amount ledger."""
    edge, species = wp.tid()
    if invalid[0] == 0 and active[0] != 0 and enabled[edge] == 1:
        left = source[edge]
        right = destination[edge]
        if left == -1:
            transfer = amounts[right, species] * rates[edge] * time_step
            if not wp.isfinite(transfer):
                wp.atomic_max(invalid, 0, 1)
            else:
                wp.atomic_add(deltas, right, species, transfer)
                if has_source_ledger != 0:
                    wp.atomic_add(source_ledger, right, species, transfer)
        else:
            transfer = amounts[left, species] * rates[edge] * time_step
            if not wp.isfinite(transfer):
                wp.atomic_max(invalid, 0, 1)
            else:
                wp.atomic_add(outbound, left, species, transfer)
                wp.atomic_add(deltas, left, species, -transfer)
                if right == -1:
                    if has_sink_ledger != 0:
                        wp.atomic_add(sink_ledger, left, species, transfer)
                else:
                    wp.atomic_add(deltas, right, species, transfer)


@wp.kernel
def _communication_validate_final(
    amounts: wp.array2d(dtype=wp.float64),
    deltas: wp.array2d(dtype=wp.float64),
    outbound: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Reject aggregate overdraw and invalid final normalization values."""
    box, species = wp.tid()
    if active[0] == 0:
        return
    final_amount = amounts[box, species] + deltas[box, species]
    concentration = final_amount / volume[box]
    if (
        outbound[box, species] > amounts[box, species]
        or final_amount < 0.0
        or not wp.isfinite(final_amount)
        or not wp.isfinite(concentration)
    ):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _communication_commit(
    concentration: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    amounts: wp.array2d(dtype=wp.float64),
    deltas: wp.array2d(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Commit the validated gas concentration exactly once."""
    box, species = wp.tid()
    if invalid[0] == 0 and active[0] != 0:
        concentration[box, species] = (
            amounts[box, species] + deltas[box, species]
        ) / volume[box]


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


def _safe_dimension_product(lengths: tuple[int, ...], name: str) -> int:
    """Return the element count for a shape or raise on overflow."""
    count = 1
    for length in lengths:
        if length != 0 and count > sys.maxsize // length:
            raise ValueError(f"{name} shape exceeds safe address range.")
        count *= length
    return count


def _safe_byte_range(
    pointer: int, count: int, item_size: int, name: str
) -> tuple[int, int]:
    """Return a validated inclusive-exclusive byte range for storage."""
    if count > sys.maxsize // item_size:
        raise ValueError(f"{name} shape exceeds safe address range.")
    required = count * item_size
    if pointer > sys.maxsize - required:
        raise ValueError(f"{name} storage exceeds safe address range.")
    return pointer, pointer + required


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
    changed: wp.array(dtype=wp.int32),
) -> None:
    """Reject overflowing or positive-to-zero scaled concentrations."""
    box, column = wp.tid()
    old_value = concentration[box, column]
    if invalid[0] == 0 and changed[0] != 0:
        factor = old_volumes[box] / final_volumes[box]
        scaled = old_value * factor
        if not wp.isfinite(scaled) or (old_value > 0.0 and scaled == 0.0):
            wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _apply_volume_evolution(
    volume: wp.array(dtype=wp.float64),
    final_volumes: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
) -> None:
    """Write validated final volume values."""
    box = wp.tid()
    if invalid[0] == 0 and changed[0] != 0:
        volume[box] = final_volumes[box]


@wp.kernel
def _apply_scaled_concentration(
    concentration: wp.array2d(dtype=wp.float64),
    old_volumes: wp.array(dtype=wp.float64),
    final_volumes: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
) -> None:
    """Apply validated per-box concentration scaling."""
    box, column = wp.tid()
    if invalid[0] == 0 and changed[0] != 0:
        factor = old_volumes[box] / final_volumes[box]
        concentration[box, column] = concentration[box, column] * factor


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
        if (
            isinstance(length, bool)
            or not isinstance(length, Integral)
            or length < 0
            or (stride != 0 and length > sys.maxsize // stride)
        ):
            raise ValueError(f"{name} shape exceeds safe address range.")
        expected.insert(0, stride)
        stride *= length
    if getattr(array, "strides", None) != tuple(expected):
        raise ValueError(f"{name} must be contiguous.")
    count = _safe_dimension_product(array.shape, name)
    if count == 0:
        return None
    pointer = getattr(array, "ptr", None)
    capacity = getattr(array, "capacity", None)
    if (
        isinstance(pointer, bool)
        or not isinstance(pointer, Integral)
        or pointer <= 0
        or pointer % item_size != 0
    ):
        raise ValueError(f"{name} must have a valid pointer.")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or capacity < count * item_size
    ):
        raise ValueError(f"{name} must have sufficient storage capacity.")
    return _safe_byte_range(int(pointer), count, item_size, name)


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
        if value is None:
            continue
        for previous in range(index):
            if arrays[previous] is None:
                continue
            left, right = ranges[index], ranges[previous]
            if value is arrays[previous] or (
                left is not None
                and right is not None
                and left[0] < right[1]
                and right[0] < left[1]
            ):
                raise ValueError("communication primary arrays must not alias.")


def _scan_1d(values: Any, kernel: Any, invalid: Any) -> None:
    """Launch a device-resident one-dimensional validation scan."""
    if values.shape[0] == 0:
        return
    wp.launch(
        kernel,
        dim=values.shape[0],
        inputs=[values, invalid],
        device=values.device,
    )


def _scan_2d(
    values: Any,
    old_volumes: Any,
    final_volumes: Any,
    invalid: Any,
    changed: Any,
    product: bool,
) -> None:
    """Launch a device-resident concentration validation scan."""
    if values.shape[0] == 0 or values.shape[1] == 0:
        return
    if product:
        wp.launch(
            _validate_scaled_concentration,
            dim=values.shape,
            inputs=[values, old_volumes, final_volumes, invalid, changed],
            device=values.device,
        )
    else:
        wp.launch(
            _scan_nonnegative_finite,
            dim=values.shape,
            inputs=[values, invalid],
            device=values.device,
        )


def volume_evolution_step_gpu(
    particles: Any, gas: Any, final_volumes: object
) -> tuple[Any, Any]:
    """Apply prescribed final volumes while preserving extensive inventories.

    ``final_volumes`` must be a caller-owned, contiguous active-device
    ``wp.float64`` array shaped ``(B,)`` in m^3. Complete read-only preflight
    validates schemas and launches device-resident domain, factor, and proposed
    product checks before gated writers. Invalid device values suppress every
    writer; errors from asynchronous device execution are observed by the caller
    at its explicit synchronization boundary. Successful calls update only
    ``particles.volume`` and particle and gas concentrations by
    ``old_volume / final_volume``. Equal final volumes are write-free no-ops.
    This direct kernel performs no hidden transfer, synchronization, or CPU
    fallback; callers synchronize before inspection.

    Args:
        particles: Warp particle container with fixed-shape primary fields.
        gas: Warp gas container with fixed-shape primary fields sharing the
            particle box count and device.
        final_volumes: Caller-owned active-device ``wp.float64`` array of final
            per-box volumes in m^3 with shape ``(B,)``.

    Returns:
        The exact input ``(particles, gas)`` containers after an in-place update
        or unchanged after a valid equal-volume no-op.

    Raises:
        ValueError: If host-observable schemas, devices, or storage ownership
            are invalid. Invalid device values suppress writers and leave
            caller-owned inputs unchanged.

    Note:
        Rollback is not promised if an asynchronous apply writer fails after it
        launches.
    """
    # Final-volume form intentionally precedes all container access.
    final_volumes_array, final_range = _validate_array(
        final_volumes, "final_volumes", wp.float64, 1
    )
    final_volumes = cast(Any, final_volumes_array)
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
    if n_boxes == 0:
        return particles, gas
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    changed = wp.zeros(1, dtype=wp.int32, device=device)
    _scan_1d(volume, _scan_positive_finite, invalid)
    _scan_1d(final_volumes, _scan_positive_finite, invalid)
    _scan_2d(
        particle_concentration,
        volume,
        final_volumes,
        invalid,
        changed,
        False,
    )
    _scan_2d(
        gas_concentration,
        volume,
        final_volumes,
        invalid,
        changed,
        False,
    )
    wp.launch(
        _validate_factors,
        dim=n_boxes,
        inputs=[volume, final_volumes, invalid, changed],
        device=device,
    )
    _scan_2d(
        particle_concentration,
        volume,
        final_volumes,
        invalid,
        changed,
        True,
    )
    _scan_2d(
        gas_concentration,
        volume,
        final_volumes,
        invalid,
        changed,
        True,
    )
    if n_particles:
        wp.launch(
            _apply_scaled_concentration,
            dim=(n_boxes, n_particles),
            inputs=[
                particle_concentration,
                volume,
                final_volumes,
                invalid,
                changed,
            ],
            device=device,
        )
    if n_gas:
        wp.launch(
            _apply_scaled_concentration,
            dim=(n_boxes, n_gas),
            inputs=[
                gas_concentration,
                volume,
                final_volumes,
                invalid,
                changed,
            ],
            device=device,
        )
    wp.launch(
        _apply_volume_evolution,
        dim=n_boxes,
        inputs=[volume, final_volumes, invalid, changed],
        device=device,
    )
    return particles, gas


def _validate_time_step(time_step: object) -> float:
    """Validate a nonnegative finite scalar time in seconds before access."""
    if isinstance(time_step, (bool, np.bool_)) or not isinstance(
        time_step, Real
    ):
        raise TypeError("time_step must be a non-boolean real scalar.")
    value = float(time_step)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("time_step must be finite and nonnegative.")
    return value


def _validate_resource_shapes(resource_shapes: object) -> None:
    """Validate the exact P1 resource-shape declaration metadata."""
    if type(resource_shapes) is not tuple:
        raise TypeError("resource_shapes must be an exact tuple.")
    roles: set[str] = set()
    for resource in resource_shapes:
        if type(resource) is not CommunicationResourceShape:
            raise TypeError(
                "resource_shapes must contain exact "
                "CommunicationResourceShape values."
            )
        if type(resource.role) is not str:
            raise TypeError("resource role must be an exact str.")
        if not resource.role or resource.role != resource.role.strip():
            raise ValueError("resource role must be nonempty and unpadded.")
        if type(resource.shape_kind) is not CommunicationShapeKind:
            raise TypeError(
                "resource shape_kind must be CommunicationShapeKind."
            )
        if resource.role in roles:
            raise ValueError("resource_shapes roles must be unique.")
        roles.add(resource.role)


def _validate_gas_communication_configuration(
    configuration: object,
) -> CommunicationConfiguration:
    """Exact-check the P1 declaration carriers needed by the P3 boundary."""
    if type(configuration) is not CommunicationConfiguration:
        raise TypeError(
            "configuration must be an exact CommunicationConfiguration."
        )
    typed = cast(CommunicationConfiguration, configuration)
    if type(typed.communication_map) is not CommunicationMap:
        raise TypeError("communication_map must be an exact CommunicationMap.")
    if type(typed.prescribed_volume) is not PrescribedVolumeUpdate:
        raise TypeError(
            "prescribed_volume must be an exact PrescribedVolumeUpdate."
        )
    map_data = typed.communication_map
    if type(map_data.form) is not CommunicationMapForm:
        raise TypeError("map form must be CommunicationMapForm.")
    if type(map_data.transport_mode) is not CommunicationTransportMode:
        raise TypeError("transport_mode must be CommunicationTransportMode.")
    if map_data.transport_mode is not CommunicationTransportMode.GAS:
        raise ValueError("communication transport_mode must be GAS.")
    if (
        isinstance(map_data.edge_capacity, bool)
        or not isinstance(map_data.edge_capacity, Integral)
        or map_data.edge_capacity < 0
    ):
        raise ValueError("edge_capacity must be a nonnegative integral.")
    _validate_resource_shapes(typed.resource_shapes)
    return typed


def gas_communication_step_gpu(
    particles: Any,
    gas: Any,
    configuration: object,
    time_step: object,
    amounts: object | GasCommunicationBuffers,
    amount_deltas: object | None = None,
    outbound_amounts: object | None = None,
    source_amounts: object | None = None,
    sink_amounts: object | None = None,
) -> tuple[Any, Any]:
    """Synchronously mix gas with caller-owned extensive amount ledgers.

    The immutable staged ledger is ``amounts = concentration * volume`` in
    amount units. An in-domain ``source -> destination`` edge transfers
    ``amounts[source] * rate * time_step``, debiting and crediting
    ``amount_deltas``. An open sink ``source -> -1`` uses that same source
    amount, debits ``amount_deltas``, and records a positive ``sink_amounts``
    entry. An open source ``-1 -> destination`` instead uses the destination
    box's pre-step amount and records its positive transfer in
    ``source_amounts``. The sole primary writer assigns
    ``gas.concentration = (amounts + amount_deltas) / volume``.

    ``amounts``, ``amount_deltas``, and ``outbound_amounts`` are caller-owned
    contiguous ``wp.float64`` arrays shaped ``(B, S)``. A
    ``GasCommunicationBuffers`` carrier may provide them. Open accounting
    ledgers have the same schema and are required for enabled edges with their
    respective boundary direction. Closed maps conserve each species' extensive
    amount up to atomic floating-point tolerance. Particle fields, prescribed
    volumes, and map metadata are read-only. Valid zero-size, empty-map, and
    zero-time calls validate observable schemas then leave all buffers and
    primaries unchanged.

    The direct call performs no hidden transfer, synchronization, conversion,
    resize, CPU fallback, or post-launch rollback. It has device work of
    ``O(E * S + B * S)`` for ``E`` edge slots, ``B`` boxes, and ``S`` gas
    species. Callers synchronize before inspecting results.

    Args:
        particles: Complete caller-owned Warp particle container.
        gas: Complete caller-owned Warp gas container.
        configuration: Exact P1 gas ``CommunicationConfiguration``.
        time_step: Finite nonnegative explicit-Euler step in seconds.
        amounts: Work ledger or concrete-only ``GasCommunicationBuffers``
            carrier.
        amount_deltas: Caller-owned net-delta work ledger when no carrier is
            provided.
        outbound_amounts: Caller-owned source-debit work ledger when no carrier
            is provided.
        source_amounts: Optional open-source accounting ledger.
        sink_amounts: Optional open-sink accounting ledger.

    Returns:
        The exact input ``(particles, gas)`` objects.

    Raises:
        TypeError: If scalar or declaration carrier types are invalid.
        ValueError: If observable array schemas, devices, aliases, buffer
            requirements, or transport declaration are invalid. Device-domain
            failures gate primary writes.

    Note:
        Applicable work and accounting ledgers may be written before an
        asynchronous device-domain failure. Rollback is not promised after the
        primary commit kernel launches.
    """
    step = _validate_time_step(time_step)
    typed = _validate_gas_communication_configuration(configuration)
    map_data = typed.communication_map
    if type(amounts) is GasCommunicationBuffers:
        if any(
            item is not None
            for item in (
                amount_deltas,
                outbound_amounts,
                source_amounts,
                sink_amounts,
            )
        ):
            raise ValueError(
                "buffer carrier cannot be combined with work arrays."
            )
        buffers = cast(GasCommunicationBuffers, amounts)
        amounts = buffers.amounts
        amount_deltas = buffers.amount_deltas
        outbound_amounts = buffers.outbound_amounts
        source_amounts = buffers.source_amounts
        sink_amounts = buffers.sink_amounts
    if amount_deltas is None or outbound_amounts is None:
        raise ValueError("amount_deltas and outbound_amounts are required.")

    masses, mass_range = _validate_array(
        _get_field(particles, "masses", "particles.masses"),
        "particles.masses",
        wp.float64,
        3,
    )
    boxes, slots, particle_species = masses.shape
    device = masses.device
    particle_concentration, particle_range = _validate_array(
        _get_field(particles, "concentration", "particles.concentration"),
        "particles.concentration",
        wp.float64,
        2,
        (boxes, slots),
        device,
    )
    charge, charge_range = _validate_array(
        _get_field(particles, "charge", "particles.charge"),
        "particles.charge",
        wp.float64,
        2,
        (boxes, slots),
        device,
    )
    density, density_range = _validate_array(
        _get_field(particles, "density", "particles.density"),
        "particles.density",
        wp.float64,
        1,
        (particle_species,),
        device,
    )
    volume, volume_range = _validate_array(
        _get_field(particles, "volume", "particles.volume"),
        "particles.volume",
        wp.float64,
        1,
        (boxes,),
        device,
    )
    concentration, concentration_range = _validate_array(
        _get_field(gas, "concentration", "gas.concentration"),
        "gas.concentration",
        wp.float64,
        2,
        device=device,
    )
    if concentration.shape[0] != boxes:
        raise ValueError("gas.concentration shape must match particle masses.")
    species = concentration.shape[1]
    molar_mass, molar_mass_range = _validate_array(
        _get_field(gas, "molar_mass", "gas.molar_mass"),
        "gas.molar_mass",
        wp.float64,
        1,
        (species,),
        device,
    )
    vapor_pressure, vapor_pressure_range = _validate_array(
        _get_field(gas, "vapor_pressure", "gas.vapor_pressure"),
        "gas.vapor_pressure",
        wp.float64,
        2,
        (boxes, species),
        device,
    )
    partitioning, partitioning_range = _validate_array(
        _get_field(gas, "partitioning", "gas.partitioning"),
        "gas.partitioning",
        wp.int32,
        2,
        (boxes, species),
        device,
    )
    edge_shape = (int(map_data.edge_capacity),)
    source, source_range = _validate_array(
        map_data.source_boxes, "source_boxes", wp.int32, 1, edge_shape, device
    )
    destination, destination_range = _validate_array(
        map_data.destination_boxes,
        "destination_boxes",
        wp.int32,
        1,
        edge_shape,
        device,
    )
    enabled, enabled_range = _validate_array(
        map_data.enabled, "enabled", wp.int32, 1, edge_shape, device
    )
    rates, rates_range = _validate_array(
        map_data.rates, "rates", wp.float64, 1, edge_shape, device
    )
    final_volumes = typed.prescribed_volume.final_volumes
    final_range = None
    if final_volumes is not None:
        final_volumes, final_range = _validate_array(
            final_volumes, "final_volumes", wp.float64, 1, (boxes,), device
        )
    amounts, amounts_range = _validate_array(
        amounts, "amounts", wp.float64, 2, (boxes, species), device
    )
    deltas, deltas_range = _validate_array(
        amount_deltas, "amount_deltas", wp.float64, 2, (boxes, species), device
    )
    outbound, outbound_range = _validate_array(
        outbound_amounts,
        "outbound_amounts",
        wp.float64,
        2,
        (boxes, species),
        device,
    )
    source_ledger, source_ledger_range = (None, None)
    sink_ledger, sink_ledger_range = (None, None)
    if source_amounts is not None:
        source_ledger, source_ledger_range = _validate_array(
            source_amounts,
            "source_amounts",
            wp.float64,
            2,
            (boxes, species),
            device,
        )
    if sink_amounts is not None:
        sink_ledger, sink_ledger_range = _validate_array(
            sink_amounts,
            "sink_amounts",
            wp.float64,
            2,
            (boxes, species),
            device,
        )
    arrays = (
        masses,
        particle_concentration,
        charge,
        density,
        volume,
        concentration,
        molar_mass,
        vapor_pressure,
        partitioning,
        source,
        destination,
        enabled,
        rates,
        final_volumes,
        amounts,
        deltas,
        outbound,
        source_ledger,
        sink_ledger,
    )
    ranges = (
        mass_range,
        particle_range,
        charge_range,
        density_range,
        volume_range,
        concentration_range,
        molar_mass_range,
        vapor_pressure_range,
        partitioning_range,
        source_range,
        destination_range,
        enabled_range,
        rates_range,
        final_range,
        amounts_range,
        deltas_range,
        outbound_range,
        source_ledger_range,
        sink_ledger_range,
    )
    _reject_aliases(arrays, ranges)
    if (
        boxes == 0
        or species == 0
        or int(map_data.edge_capacity) == 0
        or step == 0.0
    ):
        return particles, gas

    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    active = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _communication_validate_primaries,
        dim=(boxes, species),
        inputs=[volume, concentration, invalid],
        device=device,
    )
    if final_volumes is not None:
        wp.launch(
            _communication_validate_final_volumes,
            dim=boxes,
            inputs=[final_volumes, invalid],
            device=device,
        )
    wp.launch(
        _communication_validate_map,
        dim=edge_shape,
        inputs=[
            source,
            destination,
            enabled,
            rates,
            boxes,
            int(map_data.form is CommunicationMapForm.ONE_DIMENSIONAL),
            int(source_ledger is not None),
            int(sink_ledger is not None),
            active,
            invalid,
        ],
        device=device,
    )
    empty_ledger = wp.empty((0, 0), dtype=wp.float64, device=device)
    wp.launch(
        _communication_clear_and_stage,
        dim=(boxes, species),
        inputs=[
            concentration,
            volume,
            amounts,
            deltas,
            outbound,
            source_ledger if source_ledger is not None else empty_ledger,
            sink_ledger if sink_ledger is not None else empty_ledger,
            int(source_ledger is not None),
            int(sink_ledger is not None),
            active,
            invalid,
        ],
        device=device,
    )
    wp.launch(
        _communication_propose,
        dim=(edge_shape[0], species),
        inputs=[
            source,
            destination,
            enabled,
            rates,
            step,
            amounts,
            deltas,
            outbound,
            source_ledger if source_ledger is not None else empty_ledger,
            sink_ledger if sink_ledger is not None else empty_ledger,
            int(source_ledger is not None),
            int(sink_ledger is not None),
            active,
            invalid,
        ],
        device=device,
    )
    wp.launch(
        _communication_validate_final,
        dim=(boxes, species),
        inputs=[amounts, deltas, outbound, volume, active, invalid],
        device=device,
    )
    wp.launch(
        _communication_commit,
        dim=(boxes, species),
        inputs=[concentration, volume, amounts, deltas, active, invalid],
        device=device,
    )
    return particles, gas
