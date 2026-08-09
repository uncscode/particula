"""Define deterministic, caller-owned resident RNG stream state.

This concrete-only module keeps host stream identity independent from optional
GPU dependencies.  It derives reproducible initial words and can explicitly
initialize two caller-supplied Warp state arrays without allocating or binding
resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

STREAM_SCHEMA_VERSION = 1
MAX_LOGICAL_BOX_ID_BYTES = 256
MAX_ROOT_SEED = 2**32 - 1
PROCESS_IDS = ("coagulation", "wall_loss")
SUPPORTED_PROCESS_IDS = PROCESS_IDS
__all__ = [
    "STREAM_SCHEMA_VERSION",
    "MAX_LOGICAL_BOX_ID_BYTES",
    "MAX_ROOT_SEED",
    "PROCESS_IDS",
    "SUPPORTED_PROCESS_IDS",
    "StreamKey",
    "StreamDescriptor",
    "StreamRegistry",
]
_PAYLOAD_PREFIX = b"particula.execution.rng\x00"


@dataclass(frozen=True)
class StreamKey:
    """Identify one process-specific logical-box RNG stream.

    Args:
        schema_version: Supported stream-schema version.
        process_id: Supported process namespace.
        logical_box_id: Stable, unnormalized UTF-8 logical-box identifier.
    """

    schema_version: int
    process_id: str
    logical_box_id: str

    def __post_init__(self) -> None:
        """Validate immutable stream identity metadata."""
        _validate_integral(self.schema_version, "StreamKey.schema_version")
        if self.schema_version != STREAM_SCHEMA_VERSION:
            raise ValueError("StreamKey.schema_version must be 1.")
        if not isinstance(self.process_id, str):
            raise TypeError("StreamKey.process_id must be a str.")
        if self.process_id not in PROCESS_IDS:
            raise ValueError("StreamKey.process_id is unsupported.")
        _validate_logical_box_id(
            self.logical_box_id, "StreamKey.logical_box_id"
        )


@dataclass(frozen=True)
class StreamDescriptor:
    """Associate one stream key with a physical state-array lane.

    Args:
        key: Valid immutable stream identity.
        lane: Non-boolean integral physical lane. Registry admission checks its
            range against the registry dimensions.
    """

    key: StreamKey
    lane: int

    def __post_init__(self) -> None:
        """Validate descriptor carrier types without registry context."""
        if not isinstance(self.key, StreamKey):
            raise TypeError("StreamDescriptor.key must be a StreamKey.")
        _validate_integral(self.lane, "StreamDescriptor.lane")


def _validate_integral(value: object, field_name: str) -> None:
    """Validate one non-boolean integral metadata value."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a non-boolean Integral.")


def _validate_logical_box_id(value: object, field_name: str) -> bytes:
    """Validate and return the exact UTF-8 encoding of a logical box ID."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str.")
    if not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be nonempty without surrounding whitespace."
        )
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeError as error:
        raise ValueError(
            f"{field_name} must have strict UTF-8 encoding."
        ) from error
    if not 1 <= len(encoded) <= MAX_LOGICAL_BOX_ID_BYTES:
        raise ValueError(
            f"{field_name} must encode to 1 through "
            f"{MAX_LOGICAL_BOX_ID_BYTES} bytes."
        )
    return encoded


def _derive_initial_word(root_seed: int, key: StreamKey) -> int:
    """Derive one stable unsigned 32-bit initial state word."""
    process = key.process_id.encode("ascii")
    logical_id = key.logical_box_id.encode("utf-8", "strict")
    payload = b"".join(
        (
            _PAYLOAD_PREFIX,
            int(key.schema_version).to_bytes(4, "big", signed=False),
            int(root_seed).to_bytes(4, "big", signed=False),
            len(process).to_bytes(1, "big", signed=False),
            process,
            len(logical_id).to_bytes(2, "big", signed=False),
            logical_id,
        )
    )
    word = 0x811C9DC5
    for byte in payload:
        word = ((word ^ byte) * 0x01000193) & 0xFFFFFFFF
    return word


class StreamRegistry:
    """Retain one deterministic two-process state-array manifest by identity.

    Construction performs host-only metadata and collision validation. Optional
    dependencies and state-array schema validation are deferred to
    :meth:`initialize`.
    """

    def __init__(
        self,
        root_seed: int,
        n_boxes: int,
        logical_box_ids: tuple[str, ...],
        lanes: tuple[int, ...],
        state_arrays: tuple[tuple[str, Any], ...],
    ) -> None:
        """Validate and retain the immutable host manifest."""
        _validate_integral(root_seed, "root_seed")
        if not 0 <= root_seed <= MAX_ROOT_SEED:
            raise ValueError(f"root_seed must be in [0, {MAX_ROOT_SEED}].")
        _validate_integral(n_boxes, "n_boxes")
        if not 0 <= n_boxes <= MAX_ROOT_SEED:
            raise ValueError(f"n_boxes must be in [0, {MAX_ROOT_SEED}].")
        if type(logical_box_ids) is not tuple:
            raise TypeError("logical_box_ids must be an exact tuple.")
        if type(lanes) is not tuple:
            raise TypeError("lanes must be an exact tuple.")
        if len(logical_box_ids) != n_boxes or len(lanes) != n_boxes:
            raise ValueError(
                "logical_box_ids and lanes must have n_boxes entries."
            )
        for logical_box_id in logical_box_ids:
            _validate_logical_box_id(logical_box_id, "logical_box_ids entries")
        if len(set(logical_box_ids)) != n_boxes:
            raise ValueError("logical_box_ids must be unique.")
        for lane in lanes:
            _validate_integral(lane, "lanes entries")
        if set(lanes) != set(range(n_boxes)):
            raise ValueError("lanes must be a permutation of range(n_boxes).")
        _validate_state_manifest(state_arrays)

        self._root_seed = int(root_seed)
        self._n_boxes = int(n_boxes)
        self._logical_box_ids = logical_box_ids
        self._lanes = lanes
        self._state_arrays = state_arrays
        self._descriptors = tuple(
            StreamDescriptor(
                StreamKey(STREAM_SCHEMA_VERSION, process_id, logical_box_id),
                lane,
            )
            for process_id in PROCESS_IDS
            for logical_box_id, lane in zip(logical_box_ids, lanes, strict=True)
        )
        self._descriptor_by_key = {
            descriptor.key: descriptor for descriptor in self._descriptors
        }
        self._lane_by_id = dict(zip(logical_box_ids, lanes, strict=True))
        self._words_by_process = self._build_words_by_process()

    @property
    def root_seed(self) -> int:
        """Return the validated root seed."""
        return self._root_seed

    @property
    def n_boxes(self) -> int:
        """Return the fixed physical state-array size."""
        return self._n_boxes

    @property
    def logical_box_ids(self) -> tuple[str, ...]:
        """Return input-order immutable logical IDs."""
        return self._logical_box_ids

    @property
    def lanes(self) -> tuple[int, ...]:
        """Return input-order immutable physical lanes."""
        return self._lanes

    @property
    def descriptors(self) -> tuple[StreamDescriptor, ...]:
        """Return canonical process-then-logical-ID stream descriptors."""
        return self._descriptors

    @property
    def state_arrays(self) -> tuple[tuple[str, Any], ...]:
        """Return the retained ordered caller-owned arrays by identity."""
        return self._state_arrays

    def descriptor_for(
        self, process_id: str, logical_box_id: str
    ) -> StreamDescriptor:
        """Return the descriptor for an exact registered process and ID."""
        key = StreamKey(STREAM_SCHEMA_VERSION, process_id, logical_box_id)
        try:
            return self._descriptor_by_key[key]
        except KeyError as error:
            raise LookupError(
                "No stream is registered for process and logical ID."
            ) from error

    get_descriptor = descriptor_for

    def lane_for(self, logical_box_id: str) -> int:
        """Return the physical lane for an exact registered logical ID."""
        _validate_logical_box_id(logical_box_id, "logical_box_id")
        try:
            return self._lane_by_id[logical_box_id]
        except KeyError as error:
            raise LookupError(
                "No lane is registered for logical_box_id."
            ) from error

    get_lane = lane_for

    def word_for(self, process_id: str, logical_box_id: str) -> int:
        """Return the derived word for an exact registered stream."""
        descriptor = self.descriptor_for(process_id, logical_box_id)
        return self._words_by_process[process_id][descriptor.lane]

    get_derived_word = word_for

    def words_by_lane(self, process_id: str) -> tuple[int, ...]:
        """Return immutable derived words indexed by physical lane."""
        _validate_process_id(process_id)
        return self._words_by_process[process_id]

    def state_array_for(self, process_id: str) -> Any:
        """Return the retained caller-owned process state array by identity."""
        _validate_process_id(process_id)
        return self._state_arrays[PROCESS_IDS.index(process_id)][1]

    get_state_array = state_array_for

    def _build_words_by_process(self) -> dict[str, tuple[int, ...]]:
        """Build lane-indexed words and reject first same-process collision."""
        words_by_process: dict[str, tuple[int, ...]] = {}
        for process_id in PROCESS_IDS:
            words = [0] * self._n_boxes
            first_id_by_word: dict[int, str] = {}
            for logical_box_id, lane in zip(
                self._logical_box_ids, self._lanes, strict=True
            ):
                key = StreamKey(
                    STREAM_SCHEMA_VERSION, process_id, logical_box_id
                )
                word = _derive_initial_word(self._root_seed, key)
                if word in first_id_by_word:
                    raise ValueError(
                        "Derived stream-word collision for process "
                        f"{process_id!r}: {first_id_by_word[word]!r} and "
                        f"{logical_box_id!r}."
                    )
                first_id_by_word[word] = logical_box_id
                words[lane] = word
            words_by_process[process_id] = tuple(words)
        return words_by_process

    def _validate_state_arrays(self) -> None:
        """Validate retained Warp arrays fully before initialization writes."""
        import warp as wp

        arrays = tuple(array for _, array in self._state_arrays)
        for process_id, array in self._state_arrays:
            _validate_warp_state_array(array, process_id, self._n_boxes, wp)
        if arrays[0] is arrays[1]:
            raise ValueError("State arrays must be distinct objects.")
        if arrays[0].device != arrays[1].device:
            raise ValueError("State arrays must be on the same device.")
        if _arrays_overlap(arrays[0], arrays[1]):
            raise ValueError("State arrays must not alias.")

    def initialize(self) -> None:
        """Overwrite retained arrays with deterministic initial words.

        Preflight failures write neither buffer. A device failure after the
        successful copy has no rollback guarantee; callers may correct the
        failure and retry without replacing this registry's arrays.
        """
        self._validate_state_arrays()
        import numpy as np
        import warp as wp

        for process_id in PROCESS_IDS:
            host_source = np.asarray(
                self._words_by_process[process_id], dtype=np.uint32
            )
            wp.copy(
                self.state_array_for(process_id),
                wp.array(host_source, dtype=wp.uint32, device="cpu"),
            )


def _validate_process_id(process_id: object) -> None:
    """Validate an exact supported process ID."""
    if not isinstance(process_id, str):
        raise TypeError("process_id must be a str.")
    if process_id not in PROCESS_IDS:
        raise ValueError("process_id is unsupported.")


def _validate_state_manifest(value: object) -> None:
    """Validate the exact canonical ordered two-process array manifest."""
    if type(value) is not tuple:
        raise TypeError("state_arrays must be an exact tuple.")
    if len(value) != len(PROCESS_IDS):
        raise ValueError("state_arrays must contain exactly two process pairs.")
    process_ids: list[str] = []
    for entry in value:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError(
                "state_arrays entries must be exact (process_id, array) tuples."
            )
        process_id, _ = entry
        if not isinstance(process_id, str):
            raise TypeError("state_arrays process IDs must be str.")
        process_ids.append(process_id)
    if tuple(process_ids) != PROCESS_IDS:
        raise ValueError(
            "state_arrays must use canonical coagulation, wall_loss order."
        )


def _validate_warp_state_array(
    array: Any, process_id: str, n_boxes: int, wp: Any
) -> None:
    """Validate one caller-owned contiguous Warp uint32 state vector."""
    if not all(
        hasattr(array, name) for name in ("shape", "dtype", "device", "ptr")
    ):
        raise TypeError(f"{process_id} state array must be Warp-like.")
    if tuple(array.shape) != (n_boxes,):
        raise ValueError(
            f"{process_id} state array must have shape ({n_boxes},)."
        )
    if array.dtype != wp.uint32:
        raise TypeError(f"{process_id} state array must have dtype wp.uint32.")
    contiguous = getattr(array, "is_contiguous", False)
    contiguous = contiguous() if callable(contiguous) else contiguous
    if not contiguous:
        raise ValueError(f"{process_id} state array must be contiguous.")


def _arrays_overlap(first: Any, second: Any) -> bool:
    """Return whether two contiguous uint32 arrays overlap in memory."""
    if not first.shape or first.shape[0] == 0:
        return False
    byte_count = first.shape[0] * 4
    first_start = int(first.ptr)
    second_start = int(second.ptr)
    return (
        first_start < second_start + byte_count
        and second_start < first_start + byte_count
    )
