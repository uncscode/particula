"""Manage deterministic, caller-owned resident RNG stream state.

This concrete-only, direct-import module keeps host stream identity independent
of optional GPU dependencies. It derives reproducible initial words and
explicitly initializes caller-supplied Warp state arrays without acquiring,
replacing, or rebinding resources. Initialization allocates temporary host and
Warp copy sources only.

This module is intentionally not re-exported through ``particula.execution``
or the top-level package.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, no_type_check

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
    "StreamManifest",
    "StreamRegistry",
]
_PAYLOAD_PREFIX = b"particula.execution.rng\x00"


@dataclass(frozen=True)
class StreamKey:
    """Identify one process-specific RNG stream for a logical box.

    The logical-box identifier is retained exactly as supplied. It is not
    normalized before deterministic initial-word derivation.

    Args:
        schema_version: Supported stream-schema version.
        process_id: Supported process namespace.
        logical_box_id: Stable, unnormalized UTF-8 logical-box identifier.
    """

    schema_version: int
    process_id: str
    logical_box_id: str

    def __post_init__(self) -> None:
        """Validate immutable stream identity metadata.

        Raises:
            TypeError: If a field has an unsupported type.
            ValueError: If the schema version, process, or logical ID is
                invalid.
        """
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
    """Associate a stream key with a physical caller-owned array lane.

    Args:
        key: Valid immutable stream identity.
        lane: Non-boolean integral physical lane. Registry admission checks its
            range against the registry dimensions.
    """

    key: StreamKey
    lane: int

    def __post_init__(self) -> None:
        """Validate descriptor carrier types without registry context.

        Raises:
            TypeError: If ``key`` is not a StreamKey or ``lane`` is not a
                non-boolean integral value.
        """
        if not isinstance(self.key, StreamKey):
            raise TypeError("StreamDescriptor.key must be a StreamKey.")
        _validate_integral(self.lane, "StreamDescriptor.lane")


@dataclass(frozen=True)
class StreamManifest:
    """Expose immutable host-only stream identity metadata.

    This carrier deliberately contains no state arrays, pointers, device values,
    or current RNG words.
    """

    root_seed: int
    logical_box_ids: tuple[str, ...]
    lanes: tuple[int, ...]
    descriptors: tuple[StreamDescriptor, ...]


def _validate_integral(value: object, field_name: str) -> None:
    """Validate one non-boolean integral metadata value.

    Args:
        value: Value to validate.
        field_name: Field name included in a validation error.

    Raises:
        TypeError: If ``value`` is Boolean or not an integral value.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a non-boolean Integral.")


def _validate_logical_box_id(value: object, field_name: str) -> bytes:
    """Validate and encode an exact logical-box identifier as UTF-8.

    Args:
        value: Candidate logical-box identifier.
        field_name: Field name included in a validation error.

    Returns:
        Exact strict UTF-8 encoding of the validated identifier.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If the identifier is empty, surrounded by whitespace,
            not strictly UTF-8 encodable, or outside the byte-length limit.
    """
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
    """Derive a stable unsigned 32-bit initial state word.

    The word depends only on the root seed and stream key, never on the
    physical lane, registry order, capacity, unrelated identifiers, or
    Python's randomized hash implementation.

    Args:
        root_seed: Validated root seed in the unsigned 32-bit range.
        key: Validated process-specific logical stream identity.

    Returns:
        Deterministic unsigned 32-bit word for the stream identity.
    """
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

    Construction performs host-only metadata and collision validation, without
    importing optional dependencies or mutating state arrays. State-array
    schema validation and copying are deferred to :meth:`initialize`. Derived
    words are keyed only by the root seed and immutable stream identity, then
    stored by each descriptor's physical lane. The canonical manifest has
    independent coagulation and wall-loss namespaces for every logical box.

    Attributes:
        root_seed: Validated root seed used for deterministic derivation.
        n_boxes: Fixed number of physical state-array lanes.
        logical_box_ids: Input-order immutable logical-box identifiers.
        lanes: Input-order immutable physical lanes.
        descriptors: Canonical process-then-logical-ID stream descriptors.
        state_arrays: Canonical ordered caller-owned arrays, retained by
            identity.
    """

    def __init__(
        self,
        root_seed: int,
        n_boxes: int,
        logical_box_ids: tuple[str, ...],
        lanes: tuple[int, ...],
        state_arrays: tuple[tuple[str, Any], ...],
    ) -> None:
        """Validate and retain an immutable host-only stream manifest.

        Args:
            root_seed: Non-boolean unsigned 32-bit root seed.
            n_boxes: Non-boolean number of fixed physical lanes.
            logical_box_ids: Exact input-order tuple of unique logical IDs.
            lanes: Exact tuple mapping input-order IDs to physical lanes.
            state_arrays: Exact canonical two-process tuple of caller arrays.

        Raises:
            TypeError: If manifest metadata does not have the required types.
            ValueError: If metadata is out of range, malformed, noncanonical,
                or produces a same-process derived-word collision.
        """
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
        """Return the validated root seed.

        Returns:
            Root seed used for all registry stream derivations.
        """
        return self._root_seed

    @property
    def n_boxes(self) -> int:
        """Return the fixed physical state-array size.

        Returns:
            Number of physical lanes in each retained state array.
        """
        return self._n_boxes

    @property
    def logical_box_ids(self) -> tuple[str, ...]:
        """Return input-order immutable logical IDs.

        Returns:
            Exact logical-ID tuple retained during construction.
        """
        return self._logical_box_ids

    @property
    def lanes(self) -> tuple[int, ...]:
        """Return input-order immutable physical lanes.

        Returns:
            Exact input-order tuple of physical state-array lanes.
        """
        return self._lanes

    @property
    def descriptors(self) -> tuple[StreamDescriptor, ...]:
        """Return canonical process-then-logical-ID stream descriptors.

        Returns:
            Immutable descriptors ordered by process, then input logical ID.
        """
        return self._descriptors

    @property
    def state_arrays(self) -> tuple[tuple[str, Any], ...]:
        """Return retained ordered caller-owned arrays by identity.

        Returns:
            Canonical process-and-array manifest supplied at construction.
        """
        return self._state_arrays

    def descriptor_for(
        self, process_id: str, logical_box_id: str
    ) -> StreamDescriptor:
        """Return the descriptor for an exact registered process and ID.

        Args:
            process_id: Supported process namespace.
            logical_box_id: Exact registered logical-box identifier.

        Returns:
            Immutable descriptor associated with the requested stream.

        Raises:
            TypeError: If either identity component has an invalid type.
            ValueError: If the process or logical ID is invalid.
            LookupError: If the valid stream identity is not registered.
        """
        key = StreamKey(STREAM_SCHEMA_VERSION, process_id, logical_box_id)
        try:
            return self._descriptor_by_key[key]
        except KeyError as error:
            raise LookupError(
                "No stream is registered for process and logical ID."
            ) from error

    get_descriptor = descriptor_for

    def lane_for(self, logical_box_id: str) -> int:
        """Return the physical lane for an exact registered logical ID.

        Args:
            logical_box_id: Exact registered logical-box identifier.

        Returns:
            Physical state-array lane for the logical ID.

        Raises:
            TypeError: If ``logical_box_id`` is not a string.
            ValueError: If the logical ID is malformed.
            LookupError: If the valid logical ID is not registered.
        """
        _validate_logical_box_id(logical_box_id, "logical_box_id")
        try:
            return self._lane_by_id[logical_box_id]
        except KeyError as error:
            raise LookupError(
                "No lane is registered for logical_box_id."
            ) from error

    get_lane = lane_for

    def word_for(self, process_id: str, logical_box_id: str) -> int:
        """Return the derived word for an exact registered stream.

        Args:
            process_id: Supported process namespace.
            logical_box_id: Exact registered logical-box identifier.

        Returns:
            Deterministic unsigned 32-bit initial word for the stream.

        Raises:
            TypeError: If either identity component has an invalid type.
            ValueError: If the process or logical ID is invalid.
            LookupError: If the valid stream identity is not registered.
        """
        descriptor = self.descriptor_for(process_id, logical_box_id)
        return self._words_by_process[process_id][descriptor.lane]

    get_derived_word = word_for

    def words_by_lane(self, process_id: str) -> tuple[int, ...]:
        """Return immutable derived words indexed by physical lane.

        Args:
            process_id: Supported process namespace.

        Returns:
            Lane-indexed deterministic initial words for the process.

        Raises:
            TypeError: If ``process_id`` is not a string.
            ValueError: If ``process_id`` is unsupported.
        """
        _validate_process_id(process_id)
        return self._words_by_process[process_id]

    def state_array_for(self, process_id: str) -> Any:
        """Return the retained caller-owned process state array by identity.

        Args:
            process_id: Supported process namespace.

        Returns:
            Original caller-owned state array for the requested process.

        Raises:
            TypeError: If ``process_id`` is not a string.
            ValueError: If ``process_id`` is unsupported.
        """
        _validate_process_id(process_id)
        return self._state_arrays[PROCESS_IDS.index(process_id)][1]

    get_state_array = state_array_for

    def inspect(self) -> StreamManifest:
        """Return this registry's immutable host-only stream manifest."""
        return StreamManifest(
            self._root_seed,
            self._logical_box_ids,
            self._lanes,
            self._descriptors,
        )

    def initialize_selected(
        self,
        *,
        process_ids: tuple[str, ...] | None = None,
        logical_box_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Reinitialize only selected process and logical-box stream lanes.

        Full retained-array schema and nonaliasing preflight always precedes a
        writer, including for an explicitly empty selection.
        """
        selected_processes, selected_ids = _resolve_stream_selection(
            process_ids,
            logical_box_ids,
            registered_logical_box_ids=self._logical_box_ids,
        )
        self._validate_state_arrays()
        if not selected_processes or not selected_ids:
            return
        import numpy as np
        import warp as wp

        kernel = _selected_write_kernel(wp)
        lanes = np.asarray(
            [
                self._lane_by_id[logical_box_id]
                for logical_box_id in selected_ids
            ],
            dtype=np.int32,
        )
        lane_source: Any = wp.array(lanes, dtype=wp.int32, device="cpu")
        for process_id in selected_processes:
            words = np.asarray(
                [
                    self.word_for(process_id, logical_box_id)
                    for logical_box_id in selected_ids
                ],
                dtype=np.uint32,
            )
            word_source: Any = wp.array(words, dtype=wp.uint32, device="cpu")
            state = self.state_array_for(process_id)
            wp.launch(
                kernel,
                dim=len(selected_ids),
                inputs=[state, lane_source, word_source],
                device=state.device,
            )

    def _build_words_by_process(self) -> dict[str, tuple[int, ...]]:
        """Build lane-indexed words and reject same-process collisions.

        Returns:
            Process-indexed immutable tuples of words by physical lane.

        Raises:
            ValueError: If two input-order logical IDs derive the same word for
                one process.
        """
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
        """Validate retained Warp arrays fully before initialization writes.

        Raises:
            ImportError: If the optional Warp dependency is unavailable.
            TypeError: If a retained array is not a valid Warp uint32 array.
            ValueError: If an array has an invalid layout, device, identity, or
                memory-alias relationship.
        """
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

        NumPy and Warp are imported only after host manifest preflight. The
        temporary host and Warp copy sources do not change state-array
        ownership: this method deterministically overwrites retained caller
        arrays without acquiring, replacing, or rebinding them. Preflight
        failures write neither buffer. It copies lane-indexed words for
        coagulation first and wall loss second. A device or copy failure after
        the first successful copy has no rollback guarantee; callers may correct
        the failure and retry without replacing this registry's arrays.

        Raises:
            ImportError: If NumPy or Warp is unavailable.
            TypeError: If a retained state array has an invalid type or dtype.
            ValueError: If retained arrays have invalid schemas, devices, or
                aliasing relationships.
            RuntimeError: If Warp reports a device-copy failure after
                validation.
        """
        self._validate_state_arrays()

        for process_id in PROCESS_IDS:
            self.initialize_process(process_id)

    def initialize_process(self, process_id: str) -> None:
        """Initialize exactly one validated process state array.

        This narrow internal primitive lets a resident resource publish a new
        process sidecar without reseeding an already-published sibling stream,
        such as wall loss after coagulation.

        Args:
            process_id: Supported process namespace to initialize.

        Raises:
            TypeError: If the process ID or either manifest array is invalid.
            ValueError: If either retained array has invalid schema, device, or
                aliasing metadata.
            RuntimeError: If Warp reports a device-copy failure.
        """
        _validate_process_id(process_id)
        self._validate_state_arrays()
        import numpy as np
        import warp as wp

        host_source = np.asarray(
            self._words_by_process[process_id], dtype=np.uint32
        )
        wp.copy(
            self.state_array_for(process_id),
            wp.array(host_source, dtype=wp.uint32, device="cpu"),
        )


def _validate_process_id(process_id: object) -> None:
    """Validate an exact supported process ID.

    Args:
        process_id: Candidate process namespace.

    Raises:
        TypeError: If ``process_id`` is not a string.
        ValueError: If ``process_id`` is unsupported.
    """
    if not isinstance(process_id, str):
        raise TypeError("process_id must be a str.")
    if process_id not in PROCESS_IDS:
        raise ValueError("process_id is unsupported.")


def _resolve_stream_selection(
    process_ids: tuple[str, ...] | None,
    logical_box_ids: tuple[str, ...] | None,
    *,
    registered_logical_box_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate selectors and return canonical registered process/ID choices."""
    if process_ids is not None and type(process_ids) is not tuple:
        raise TypeError("process_ids must be an exact tuple.")
    if logical_box_ids is not None and type(logical_box_ids) is not tuple:
        raise TypeError("logical_box_ids must be an exact tuple.")
    selected_processes = PROCESS_IDS if process_ids is None else process_ids
    selected_ids = (
        registered_logical_box_ids
        if logical_box_ids is None
        else logical_box_ids
    )
    for process_id in selected_processes:
        _validate_process_id(process_id)
    for logical_box_id in selected_ids:
        _validate_logical_box_id(logical_box_id, "logical_box_ids entries")
        if logical_box_id not in registered_logical_box_ids:
            raise LookupError("No lane is registered for logical_box_id.")
    if len(set(selected_processes)) != len(selected_processes):
        raise ValueError("process_ids must be unique.")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("logical_box_ids must be unique.")
    return tuple(selected_processes), tuple(selected_ids)


_SELECTED_WRITE_KERNEL: Any | None = None


def _selected_write_kernel(wp: Any) -> Any:
    """Lazily create the indexed selected-lane Warp writer."""
    global _SELECTED_WRITE_KERNEL
    if _SELECTED_WRITE_KERNEL is None:

        @no_type_check
        @wp.kernel
        def selected_write(
            state: wp.array(dtype=wp.uint32),
            lanes: wp.array(dtype=wp.int32),
            words: wp.array(dtype=wp.uint32),
        ) -> None:
            index = wp.tid()
            state[lanes[index]] = words[index]

        _SELECTED_WRITE_KERNEL = selected_write
    return _SELECTED_WRITE_KERNEL


def _validate_state_manifest(value: object) -> None:
    """Validate an exact canonical ordered two-process array manifest.

    Args:
        value: Candidate process-and-array manifest.

    Raises:
        TypeError: If the manifest or a pair has an invalid exact tuple type.
        ValueError: If the manifest does not contain the canonical two-process
            order.
    """
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
    """Validate one caller-owned contiguous Warp uint32 state vector.

    Args:
        array: Candidate Warp-like one-dimensional state array.
        process_id: Process name included in validation errors.
        n_boxes: Required physical-lane count.
        wp: Lazily imported Warp module defining the required dtype.

    Raises:
        TypeError: If the array is not a Warp array or has the wrong dtype.
        ValueError: If the array shape or memory layout is invalid.
    """
    if not isinstance(array, wp.array):
        raise TypeError(f"{process_id} state array must be a Warp array.")
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
    """Return whether two contiguous uint32 arrays overlap in memory.

    Args:
        first: First validated contiguous state array.
        second: Second validated contiguous state array.

    Returns:
        True when the nonempty arrays have overlapping byte ranges.
    """
    if not first.shape or first.shape[0] == 0:
        return False
    byte_count = first.shape[0] * 4
    first_start = int(first.ptr)
    second_start = int(second.ptr)
    return (
        first_start < second_start + byte_count
        and second_start < first_start + byte_count
    )
