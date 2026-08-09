"""Declare and preflight fixed-shape resident communication maps.

This concrete-only, direct-import boundary performs read-only P1 validation of
caller-owned Warp map arrays. It neither reads resident primaries, copies map
payloads, nor launches a writer, and is deliberately absent from
``particula.execution`` exports. Empty and all-disabled maps are successful
write-free cases only after complete applicable carrier, schema, alias, and
domain preflight.

P1 validates topology and rates in 1/s, but receives neither source inventory
nor time step. Population-dependent outbound-overdraw validation therefore
belongs to P3, before its writers launch. P2 owns volume writers, P4 owns
particle transport, and P5 owns resident binding and primary/sidecar aliasing.
This module provides none of those operations, synchronization, host
conversion, fallback, or scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, cast

import warp as wp

from particula.execution.gpu_session import ResidentDimensions


class CommunicationMapForm(str, Enum):
    """Declare the enabled-edge topology accepted by a communication map.

    ``ONE_DIMENSIONAL`` permits only neighboring enabled box pairs, while
    ``ARBITRARY_PAIRS`` permits any distinct in-range directed pair.
    """

    ONE_DIMENSIONAL = "one_dimensional"
    ARBITRARY_PAIRS = "arbitrary_pairs"


class CommunicationTransportMode(str, Enum):
    """Declare the resident payload class a later phase may transport.

    P1 records this immutable mode but does not read, transport, or otherwise
    validate resident gas or particle payloads.
    """

    GAS = "gas"
    PARTICLES = "particles"
    GAS_AND_PARTICLES = "gas_and_particles"


class CommunicationShapeKind(str, Enum):
    """Declare fixed resource-shape vocabulary without retaining arrays.

    The letters denote fixed dimensions: ``B`` boxes, ``E`` edges, ``S`` gas
    species, and ``N`` particle slots.
    """

    B = "b"
    E = "e"
    BE = "be"
    BS = "bs"
    BN = "bn"
    BNS = "bns"


@dataclass(frozen=True)
class CommunicationResourceShape:
    """Describe one named fixed-shape communication resource.

    Args:
        role: Nonempty, unpadded resource role. This carrier owns no array.
        dtype: Declared resource dtype metadata, retained unchanged.
        shape_kind: Exact fixed-shape vocabulary value.
    """

    role: str
    dtype: object
    shape_kind: CommunicationShapeKind

    def __post_init__(self) -> None:
        """Perform cheap role and shape metadata validation."""
        if type(self.role) is not str:
            raise TypeError("resource role must be an exact str.")
        if not self.role or self.role != self.role.strip():
            raise ValueError("resource role must be nonempty and unpadded.")
        if type(self.shape_kind) is not CommunicationShapeKind:
            raise TypeError(
                "resource shape_kind must be CommunicationShapeKind."
            )


@dataclass(frozen=True, eq=False)
class CommunicationMap:
    """Retain fixed-capacity caller-owned map arrays by identity.

    The frozen carrier keeps the supplied array objects, rather than copies or
    normalized replacements. During validation, ``source_boxes``,
    ``destination_boxes``, and ``enabled`` must be distinct, nonoverlapping,
    same-device contiguous pointer-backed ``wp.int32`` arrays of shape
    ``(E,)``. ``rates`` must be a similarly distinct ``wp.float64`` ``(E,)``
    array in 1/s. Validation neither normalizes nor mutates these arrays.

    Attributes:
        form: Enabled-edge topology rule.
        transport_mode: Payload class that a later phase may transport.
        edge_capacity: Non-boolean integral edge capacity ``E >= 0``.
        source_boxes: Caller-owned source-box indices with shape ``(E,)``.
        destination_boxes: Caller-owned destination-box indices with shape
            ``(E,)``.
        enabled: Caller-owned zero-or-one edge flags with shape ``(E,)``.
        rates: Caller-owned nonnegative finite per-edge rates in 1/s.
    """

    form: CommunicationMapForm
    transport_mode: CommunicationTransportMode
    edge_capacity: int
    source_boxes: object
    destination_boxes: object
    enabled: object
    rates: object

    def __post_init__(self) -> None:
        """Validate only fixed-cost carrier metadata."""
        if type(self.form) is not CommunicationMapForm:
            raise TypeError("map form must be CommunicationMapForm.")
        if type(self.transport_mode) is not CommunicationTransportMode:
            raise TypeError(
                "transport_mode must be CommunicationTransportMode."
            )
        if isinstance(self.edge_capacity, bool) or not isinstance(
            self.edge_capacity, Integral
        ):
            raise TypeError("edge_capacity must be an integral, not bool.")
        if self.edge_capacity < 0:
            raise ValueError("edge_capacity must be nonnegative.")


@dataclass(frozen=True, eq=False)
class PrescribedVolumeUpdate:
    """Retain optional caller-owned final per-box volumes by identity.

    ``final_volumes`` is retained by identity and is ``None`` or later validates
    as a distinct, nonoverlapping, same-device contiguous pointer-backed
    ``wp.float64`` array shaped ``(B,)`` in m³ with finite positive values. P1
    only preflights it; P2 owns any volume write.

    Attributes:
        final_volumes: Optional caller-owned final volumes in m³ with shape
            ``(B,)``.
    """

    final_volumes: object | None


@dataclass(frozen=True, eq=False)
class CommunicationConfiguration:
    """Combine immutable communication declarations without binding a session.

    The configuration preserves nested carrier and Warp-array identities, has
    no session or registry binding, and carries no resident primary state. It
    neither reads source population nor accepts a time step, so it cannot
    validate population-dependent outbound totals; P3 must do so before its
    writers run.

    Attributes:
        communication_map: Fixed-capacity map declaration and caller-owned
            arrays.
        prescribed_volume: Optional final-volume declaration; P2 owns writes.
        resource_shapes: Unique-role metadata declarations that own no arrays.
    """

    communication_map: CommunicationMap
    prescribed_volume: PrescribedVolumeUpdate
    resource_shapes: tuple[CommunicationResourceShape, ...]

    def __post_init__(self) -> None:
        """Validate cheap nested carrier and unique-role metadata."""
        if type(self.communication_map) is not CommunicationMap:
            raise TypeError(
                "communication_map must be an exact CommunicationMap."
            )
        if type(self.prescribed_volume) is not PrescribedVolumeUpdate:
            raise TypeError(
                "prescribed_volume must be an exact PrescribedVolumeUpdate."
            )
        if type(self.resource_shapes) is not tuple:
            raise TypeError("resource_shapes must be an exact tuple.")
        if not all(
            type(item) is CommunicationResourceShape
            for item in self.resource_shapes
        ):
            raise TypeError(
                "resource_shapes must contain exact "
                "CommunicationResourceShape values."
            )
        roles = tuple(item.role for item in self.resource_shapes)
        if len(set(roles)) != len(roles):
            raise ValueError("resource_shapes roles must be unique.")


@wp.kernel
def _scan_enabled(
    enabled: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)
):
    """Flag any enabled-entry value other than 0 or 1.

    Args:
        enabled: One-dimensional integer flag array for enabled edges.
        invalid: Single-element integer status buffer updated in place.
    """
    index = wp.tid()
    value = enabled[index]
    if value != 0 and value != 1:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_rates(
    rates: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
):
    """Flag any rate that is not finite and nonnegative.

    Args:
        rates: One-dimensional floating-point rate array in 1/s.
        invalid: Single-element integer status buffer updated in place.
    """
    index = wp.tid()
    value = rates[index]
    if not wp.isfinite(value) or value < 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_volumes(
    volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)
):
    """Flag any prescribed volume that is not finite and positive.

    Args:
        volumes: One-dimensional floating-point final-volume array in m^3.
        invalid: Single-element integer status buffer updated in place.
    """
    index = wp.tid()
    value = volumes[index]
    if not wp.isfinite(value) or value <= 0.0:
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _scan_topology(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    boxes: int,
    one_dimensional: int,
    invalid: wp.array(dtype=wp.int32),
):
    """Flag any enabled edge with invalid bounds or one-dimensional gap.

    Args:
        source: One-dimensional integer source-box array.
        destination: One-dimensional integer destination-box array.
        enabled: One-dimensional integer enabled-edge array.
        boxes: Number of resident boxes; valid enabled endpoints are in
            ``[0, boxes)``.
        one_dimensional: Integer flag for the one-dimensional topology rule.
        invalid: Single-element integer status buffer updated in place.
    """
    index = wp.tid()
    if enabled[index] == 1:
        left = source[index]
        right = destination[index]
        if (
            left < 0
            or left >= boxes
            or right < 0
            or right >= boxes
            or left == right
            or (one_dimensional == 1 and wp.abs(left - right) != 1)
        ):
            wp.atomic_max(invalid, 0, 1)


@wp.kernel
def _count_enabled(
    enabled: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)
):
    """Count enabled edges into validator-owned scalar storage."""
    index = wp.tid()
    if enabled[index] == 1:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def _collect_enabled_keys(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    boxes: int,
    keys: wp.array(dtype=wp.int64),
    cursor: wp.array(dtype=wp.int32),
):
    """Collect enabled directed-pair keys into private compact scratch.

    The atomic cursor makes the compact order unspecified, which is harmless
    because a later deterministic sort compares keys rather than edge order.
    """
    index = wp.tid()
    if enabled[index] == 1:
        key = wp.int64(source[index]) * wp.int64(boxes) + wp.int64(
            destination[index]
        )
        position = wp.atomic_add(cursor, 0, 1)
        keys[position] = key


@wp.kernel
def _bitonic_sort_step(keys: wp.array(dtype=wp.int64), stage: int, offset: int):
    """Execute one compare-exchange stage of an ascending bitonic sort."""
    index = wp.tid()
    partner = index ^ offset
    if partner > index:
        ascending = (index & stage) == 0
        left = keys[index]
        right = keys[partner]
        if (ascending and left > right) or (not ascending and left < right):
            keys[index] = right
            keys[partner] = left


@wp.kernel
def _scan_sorted_duplicates(
    keys: wp.array(dtype=wp.int64),
    count: int,
    invalid: wp.array(dtype=wp.int32),
):
    """Flag adjacent equal valid keys after private sorting."""
    index = wp.tid()
    if index > 0 and index < count and keys[index - 1] == keys[index]:
        wp.atomic_max(invalid, 0, 1)


def _is_warp_array(value: object) -> bool:
    """Return whether value is a concrete Warp array carrier."""
    return (
        type(value).__module__.startswith("warp")
        and type(value).__name__ == "array"
    )


def _array_range(
    value: Any, shape: tuple[int, ...], name: str, item_size: int
) -> tuple[int, int] | None:
    """Validate contiguous backing metadata and return a nonempty byte range."""
    expected: list[int] = []
    stride = item_size
    for length in reversed(shape):
        expected.insert(0, stride)
        stride *= length
    if getattr(value, "strides", None) != tuple(expected):
        raise ValueError(f"{name} must be contiguous.")
    count = 1
    for length in shape:
        count *= length
    if count == 0:
        return None
    pointer = getattr(value, "ptr", None)
    if not isinstance(pointer, Integral) or pointer <= 0:
        raise ValueError(f"{name} must have a valid pointer.")
    if pointer % item_size:
        raise ValueError(f"{name} pointer must be {item_size}-byte aligned.")
    capacity = getattr(value, "capacity", None)
    required = count * item_size
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or capacity < required
        or capacity % item_size
    ):
        raise ValueError(
            f"{name} must have sufficient integral storage capacity."
        )
    return int(pointer), int(pointer) + required


def _validate_array(
    value: object,
    shape: tuple[int, ...],
    dtype: object,
    device: object,
    name: str,
    item_size: int,
) -> tuple[int, int] | None:
    """Validate one fixed-schema Warp array without reading its payload."""
    if not _is_warp_array(value):
        raise ValueError(f"{name} must be a Warp array.")
    array = cast(Any, value)
    if len(getattr(array, "shape", ())) != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}.")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}.")
    if array.device != device:
        raise ValueError(f"{name} device must match device.")
    return _array_range(array, shape, name, item_size)


def _overlaps(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> bool:
    """Return whether two nonempty half-open byte ranges overlap."""
    return (
        left is not None
        and right is not None
        and left[0] < right[1]
        and right[0] < left[1]
    )


def _scan(value: Any, size: int, kernel: Any, name: str, message: str) -> None:
    """Run one read-only scalar-status payload validation scan."""
    if size == 0:
        return
    device = cast(Any, value.device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel, dim=size, inputs=[value, invalid], device=device)
    if int(invalid.numpy()[0]) != 0:
        raise ValueError(f"{name} {message}.")


def _duplicate_scratch_size(enabled_edges: int) -> int:
    """Return power-of-two private sort storage for enabled edges only.

    Disabled fixed-capacity padding never increases this allocation. The
    validator uses bounded ``O(E_enabled * log(E_enabled)^2)`` compare-exchange
    work, avoiding input-dependent collision probing.
    """
    size = 1
    while size < max(1, enabled_edges):
        size *= 2
    return size


def _validate_resource_shapes(
    resource_shapes: object,
) -> tuple[CommunicationResourceShape, ...]:
    """Validate exact resource declarations before payload metadata."""
    if type(resource_shapes) is not tuple:
        raise TypeError("resource_shapes must be an exact tuple.")
    if not all(
        type(item) is CommunicationResourceShape for item in resource_shapes
    ):
        raise TypeError(
            "resource_shapes must contain exact "
            "CommunicationResourceShape values."
        )
    typed_shapes = cast(tuple[CommunicationResourceShape, ...], resource_shapes)
    for resource in typed_shapes:
        if type(resource.role) is not str:
            raise TypeError("resource role must be an exact str.")
        if not resource.role or resource.role != resource.role.strip():
            raise ValueError("resource role must be nonempty and unpadded.")
        if type(resource.shape_kind) is not CommunicationShapeKind:
            raise TypeError(
                "resource shape_kind must be CommunicationShapeKind."
            )
    roles = tuple(resource.role for resource in typed_shapes)
    if len(set(roles)) != len(roles):
        raise ValueError("resource_shapes roles must be unique.")
    return typed_shapes


def _validate_dimensions(dimensions: object) -> int:
    """Validate exact resident dimensions and return the box count."""
    if type(dimensions) is not ResidentDimensions:
        raise TypeError("dimensions must be an exact ResidentDimensions.")
    boxes_value = cast(ResidentDimensions, dimensions).n_boxes
    if isinstance(boxes_value, bool) or not isinstance(boxes_value, Integral):
        raise TypeError("dimensions.n_boxes must be an integral, not bool.")
    if boxes_value < 0:
        raise ValueError("dimensions.n_boxes must be nonnegative.")
    return int(boxes_value)


def _validate_device(device: object) -> None:
    """Validate explicit non-null Warp device metadata."""
    if (
        device is None
        or not type(device).__module__.startswith("warp")
        or type(device).__name__ != "Device"
    ):
        raise TypeError("device must be non-null Warp device metadata.")


def _validate_nested_metadata(typed: CommunicationConfiguration) -> None:
    """Validate nested carrier, enum, capacity, and resource metadata."""
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
    if isinstance(map_data.edge_capacity, bool) or not isinstance(
        map_data.edge_capacity, Integral
    ):
        raise TypeError("edge_capacity must be an integral, not bool.")
    if map_data.edge_capacity < 0:
        raise ValueError("edge_capacity must be nonnegative.")
    _validate_resource_shapes(typed.resource_shapes)


def _validate_metadata(
    configuration: object, dimensions: object, device: object
) -> tuple[CommunicationConfiguration, int]:
    """Validate exact outer carriers before inspecting any Warp metadata."""
    if type(configuration) is not CommunicationConfiguration:
        raise TypeError(
            "configuration must be an exact CommunicationConfiguration."
        )
    boxes = _validate_dimensions(dimensions)
    _validate_device(device)
    typed = cast(CommunicationConfiguration, configuration)
    _validate_nested_metadata(typed)
    return typed, boxes


def _validate_payload_arrays(
    map_data: CommunicationMap,
    volumes: object | None,
    boxes: int,
    device: Any,
) -> tuple[
    tuple[object | None, ...],
    tuple[tuple[int, int] | None, ...],
]:
    """Validate payload schemas and return values with their byte ranges."""
    edge_shape = (int(map_data.edge_capacity),)
    source_range = _validate_array(
        map_data.source_boxes,
        edge_shape,
        wp.int32,
        device,
        "source_boxes",
        4,
    )
    destination_range = _validate_array(
        map_data.destination_boxes,
        edge_shape,
        wp.int32,
        device,
        "destination_boxes",
        4,
    )
    enabled_range = _validate_array(
        map_data.enabled,
        edge_shape,
        wp.int32,
        device,
        "enabled",
        4,
    )
    rates_range = _validate_array(
        map_data.rates,
        edge_shape,
        wp.float64,
        device,
        "rates",
        8,
    )
    volume_range = None
    if volumes is not None:
        volume_range = _validate_array(
            volumes,
            (boxes,),
            wp.float64,
            device,
            "final_volumes",
            8,
        )
    return (
        (
            map_data.source_boxes,
            map_data.destination_boxes,
            map_data.enabled,
            map_data.rates,
            volumes,
        ),
        (
            source_range,
            destination_range,
            enabled_range,
            rates_range,
            volume_range,
        ),
    )


def _reject_payload_aliases(
    values: tuple[object | None, ...],
    ranges: tuple[tuple[int, int] | None, ...],
) -> None:
    """Reject shared identities and overlapping nonempty payload ranges."""
    for index, value in enumerate(values):
        if value is None:
            continue
        for previous in range(index):
            if value is values[previous] or _overlaps(
                ranges[index], ranges[previous]
            ):
                raise ValueError("communication map arrays must not alias.")


def _validate_payload_domains(
    map_data: CommunicationMap, volumes: object | None, boxes: int
) -> None:
    """Scan enabled flags, rates, and optional volumes in field order."""
    edges = int(map_data.edge_capacity)
    _scan(
        cast(Any, map_data.enabled),
        edges,
        _scan_enabled,
        "enabled",
        "values must be 0 or 1",
    )
    _scan(
        cast(Any, map_data.rates),
        edges,
        _scan_rates,
        "rates",
        "values must be finite and nonnegative",
    )
    if volumes is not None:
        _scan(
            cast(Any, volumes),
            boxes,
            _scan_volumes,
            "final_volumes",
            "values must be finite and positive",
        )


def _reject_duplicate_edges(
    source: Any,
    destination: Any,
    enabled: Any,
    boxes: int,
    edges: int,
    enabled_edges: int,
    device: Any,
) -> None:
    """Reject duplicate enabled directed pairs using private sort scratch."""
    scratch_size = _duplicate_scratch_size(enabled_edges)
    keys = wp.full(
        scratch_size,
        wp.int64(9223372036854775807),
        dtype=wp.int64,
        device=device,
    )
    cursor = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _collect_enabled_keys,
        dim=edges,
        inputs=[source, destination, enabled, boxes, keys, cursor],
        device=device,
    )
    stage = 2
    while stage <= scratch_size:
        offset = stage // 2
        while offset > 0:
            wp.launch(
                _bitonic_sort_step,
                dim=scratch_size,
                inputs=[keys, stage, offset],
                device=device,
            )
            offset //= 2
        stage *= 2
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _scan_sorted_duplicates,
        dim=enabled_edges,
        inputs=[keys, enabled_edges, invalid],
        device=device,
    )
    if int(invalid.numpy()[0]) != 0:
        raise ValueError("enabled directed edge pairs must be unique.")


def _validate_enabled_topology(
    map_data: CommunicationMap, boxes: int, device: Any
) -> None:
    """Validate enabled topology and reject duplicate directed pairs."""
    edges = int(map_data.edge_capacity)
    if edges == 0:
        return
    source = cast(Any, map_data.source_boxes)
    destination = cast(Any, map_data.destination_boxes)
    enabled = cast(Any, map_data.enabled)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _scan_topology,
        dim=edges,
        inputs=[
            source,
            destination,
            enabled,
            boxes,
            int(map_data.form is CommunicationMapForm.ONE_DIMENSIONAL),
            invalid,
        ],
        device=device,
    )
    if int(invalid.numpy()[0]) != 0:
        raise ValueError("enabled edges must have valid distinct topology.")
    enabled_count = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _count_enabled,
        dim=edges,
        inputs=[enabled, enabled_count],
        device=device,
    )
    enabled_edges = int(enabled_count.numpy()[0])
    if enabled_edges >= 2:
        _reject_duplicate_edges(
            source,
            destination,
            enabled,
            boxes,
            edges,
            enabled_edges,
            device,
        )


def validate_communication_configuration(
    configuration: object,
    dimensions: object,
    device: object,
) -> CommunicationConfiguration:
    """Read-only validate one fixed-shape communication configuration.

    The validator retains and returns the exact configuration object. It writes
    no caller-owned array or carrier and makes no payload copy; private status
    and duplicate-detection scratch are permitted. Map index and flag arrays
    must be same-device, contiguous, pointer-backed ``wp.int32`` arrays with
    shape ``(E,)``; rates must be ``wp.float64`` ``(E,)`` in 1/s. Optional final
    volumes must be ``wp.float64`` ``(B,)`` in m³. Array identities and nonempty
    half-open byte ranges must not overlap.

    Args:
        configuration: Exact immutable map, optional-volume, and resource-shape
            carrier.
        dimensions: Exact resident dimensions; only ``n_boxes`` (``B``) is used.
        device: Explicit active Warp device metadata for all payload arrays.

    Returns:
        The exact unchanged ``configuration`` object.

    Raises:
        TypeError: If outer carriers or device metadata are invalid.
        ValueError: If array schema, identity ownership, byte-range aliasing,
            value domains, or enabled topology is invalid.

    Notes:
        Empty and all-disabled maps still receive full applicable preflight and
        are write-free on success. This validates map topology and rate domains
        only. It has neither source-inventory nor time-step input and cannot
        validate outbound population totals; P3 must enforce that condition
        before writers run.
    """
    typed, boxes = _validate_metadata(configuration, dimensions, device)
    map_data = typed.communication_map
    volumes = typed.prescribed_volume.final_volumes
    warp_device = cast(Any, device)
    values, ranges = _validate_payload_arrays(
        map_data, volumes, boxes, warp_device
    )
    _reject_payload_aliases(values, ranges)
    _validate_payload_domains(map_data, volumes, boxes)
    _validate_enabled_topology(map_data, boxes, warp_device)
    return typed
