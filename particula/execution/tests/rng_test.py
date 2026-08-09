"""Tests for concrete deterministic resident RNG stream initialization."""

from __future__ import annotations

import builtins
import importlib
import subprocess
import sys
from typing import Any

import pytest

from particula.execution.rng import (
    MAX_ROOT_SEED,
    STREAM_SCHEMA_VERSION,
    StreamDescriptor,
    StreamKey,
    StreamRegistry,
    _arrays_overlap,
    _derive_initial_word,
)


def _arrays() -> tuple[tuple[str, object], ...]:
    """Return an intentionally opaque host-only state manifest."""
    return (("coagulation", object()), ("wall_loss", object()))


def test_stream_key_validates_stable_identity() -> None:
    """Test stream keys preserve exact valid Unicode IDs."""
    key = StreamKey(STREAM_SCHEMA_VERSION, "coagulation", "box-μ")

    assert key.logical_box_id == "box-μ"
    with pytest.raises(ValueError, match="schema_version"):
        StreamKey(2, "coagulation", "box")
    with pytest.raises(ValueError, match="process_id"):
        StreamKey(1, "other", "box")
    with pytest.raises(ValueError, match="whitespace"):
        StreamKey(1, "wall_loss", " box")
    with pytest.raises(ValueError, match="1 through"):
        StreamKey(1, "wall_loss", "a" * 257)
    with pytest.raises(TypeError, match="process_id must be a str"):
        StreamKey(1, 1, "box")  # type: ignore[arg-type]


def test_stream_descriptor_rejects_non_key_identity() -> None:
    """Test descriptors reject a key that was not fully validated."""
    with pytest.raises(TypeError, match="StreamDescriptor.key"):
        StreamDescriptor("coagulation", 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (None, 1, b"box", "", "box ", "\ud800"))
def test_stream_key_rejects_invalid_logical_box_ids(value: object) -> None:
    """Test logical IDs preserve the strict host-only identity contract."""
    with pytest.raises((TypeError, ValueError)):
        StreamKey(1, "coagulation", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, 1.5, "1"))
def test_stream_key_rejects_non_integral_schema_version(value: object) -> None:
    """Test schema versions require non-boolean integral metadata."""
    with pytest.raises(TypeError, match="non-boolean Integral"):
        StreamKey(value, "coagulation", "box")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, 1.2, "1"))
def test_stream_descriptor_rejects_non_integral_lane(value: object) -> None:
    """Test descriptors accept only non-boolean integral lanes."""
    with pytest.raises(TypeError, match="non-boolean Integral"):
        StreamDescriptor(StreamKey(1, "coagulation", "box"), value)  # type: ignore[arg-type]


def test_derivation_is_stable_and_process_separated() -> None:
    """Test FNV derivation uses the complete process-specific identity."""
    coagulation = StreamKey(1, "coagulation", "α")
    wall_loss = StreamKey(1, "wall_loss", "α")

    assert _derive_initial_word(0, coagulation) == 2012932154
    assert _derive_initial_word(MAX_ROOT_SEED, coagulation) == 2865388790
    assert _derive_initial_word(0, coagulation) != _derive_initial_word(
        0, wall_loss
    )


def test_derivation_is_independent_of_manifest_order_and_unrelated_ids() -> (
    None
):
    """Test a stream word depends only on its root seed and exact key."""
    base = StreamRegistry(19, 2, ("one", "two"), (0, 1), _arrays())
    reordered = StreamRegistry(
        19, 3, ("other", "two", "one"), (1, 2, 0), _arrays()
    )

    for process_id in ("coagulation", "wall_loss"):
        assert base.word_for(process_id, "one") == reordered.word_for(
            process_id, "one"
        )
        assert base.word_for(process_id, "two") == reordered.word_for(
            process_id, "two"
        )


def test_registry_builds_canonical_descriptors_and_lane_words() -> None:
    """Test registry ordering is process then input ID, independent of lanes."""
    registry = StreamRegistry(
        7,
        2,
        ("right", "left"),
        (1, 0),
        _arrays(),
    )

    assert tuple(item.key.process_id for item in registry.descriptors) == (
        "coagulation",
        "coagulation",
        "wall_loss",
        "wall_loss",
    )
    assert registry.lane_for("right") == 1
    assert (
        registry.word_for("coagulation", "right")
        == registry.words_by_lane("coagulation")[1]
    )
    assert registry.get_descriptor("wall_loss", "left").lane == 0
    assert registry.state_array_for("wall_loss") is registry.state_arrays[1][1]
    with pytest.raises(LookupError, match="No lane"):
        registry.lane_for("missing")
    with pytest.raises(ValueError, match="unsupported"):
        registry.words_by_lane("unknown")
    with pytest.raises(LookupError, match="No stream"):
        registry.descriptor_for("coagulation", "missing")


def test_arrays_overlap_detects_intersecting_and_empty_ranges() -> None:
    """Test overlap detection handles intersecting, separate, and empty arrays."""

    class Array:
        """Provide the validated attributes required by overlap detection."""

        def __init__(self, ptr: int, size: int) -> None:
            """Store synthetic contiguous uint32 storage metadata."""
            self.ptr = ptr
            self.shape = (size,)

    assert _arrays_overlap(Array(100, 2), Array(104, 2))
    assert not _arrays_overlap(Array(100, 1), Array(104, 1))
    assert not _arrays_overlap(Array(100, 0), Array(100, 0))


@pytest.mark.parametrize(
    ("root_seed", "n_boxes", "ids", "lanes", "message"),
    [
        (True, 1, ("box",), (0,), "root_seed"),
        (0, True, ("box",), (0,), "n_boxes"),
        (0, 1, ["box"], (0,), "exact tuple"),
        (0, 2, ("box", "box"), (0, 1), "unique"),
        (0, 2, ("a", "b"), (0, 0), "permutation"),
    ],
)
def test_registry_rejects_invalid_host_manifest(
    root_seed: object,
    n_boxes: object,
    ids: object,
    lanes: object,
    message: str,
) -> None:
    """Test invalid host metadata rejects before optional dependency use."""
    with pytest.raises((TypeError, ValueError), match=message):
        StreamRegistry(root_seed, n_boxes, ids, lanes, _arrays())  # type: ignore[arg-type]


def test_registry_rejects_noncanonical_lanes_and_mismatched_lengths() -> None:
    """Test registry metadata requires exact lanes and matching dimensions."""
    with pytest.raises(TypeError, match="lanes must be an exact tuple"):
        StreamRegistry(0, 1, ("box",), [0], _arrays())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must have n_boxes entries"):
        StreamRegistry(0, 1, (), (), _arrays())


def test_registry_requires_exact_canonical_process_array_manifest() -> None:
    """Test manifest ordering is explicit and cannot be supplied as a mapping."""
    with pytest.raises(ValueError, match="canonical"):
        StreamRegistry(
            0,
            0,
            (),
            (),
            (("wall_loss", object()), ("coagulation", object())),
        )
    with pytest.raises(TypeError, match="exact tuple"):
        StreamRegistry(0, 0, (), (), {"coagulation": object()})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="exactly two"):
        StreamRegistry(0, 0, (), (), (("coagulation", object()),))
    with pytest.raises(TypeError, match="entries"):
        bad_entry: Any = ["wall_loss", object()]
        StreamRegistry(
            0,
            0,
            (),
            (),
            (("coagulation", object()), bad_entry),
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "root_seed,n_boxes", ((-1, 0), (MAX_ROOT_SEED + 1, 0), (0, -1))
)
def test_registry_rejects_out_of_range_dimensions(
    root_seed: int, n_boxes: int
) -> None:
    """Test registry dimensions remain inside the fixed unsigned range."""
    with pytest.raises(ValueError, match="must be in"):
        StreamRegistry(root_seed, n_boxes, (), (), _arrays())


def test_registry_collision_rejects_first_input_order_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test same-process derived-word collisions identify the first two IDs."""
    rng_module = importlib.import_module("particula.execution.rng")
    monkeypatch.setattr(
        rng_module, "_derive_initial_word", lambda _root, _key: 3
    )

    with pytest.raises(ValueError, match="'first' and 'second'"):
        StreamRegistry(0, 2, ("first", "second"), (0, 1), _arrays())


def test_registry_returns_immutable_metadata_and_retained_identity() -> None:
    """Test registry lookup results cannot mutate retained manifest metadata."""
    arrays = _arrays()
    registry = StreamRegistry(3, 1, ("box",), (0,), arrays)

    assert registry.logical_box_ids is registry.logical_box_ids
    assert registry.lanes is registry.lanes
    assert registry.descriptors is registry.descriptors
    assert registry.state_arrays is arrays
    assert registry.root_seed == 3
    assert registry.n_boxes == 1
    with pytest.raises(AttributeError):
        registry.descriptors.append(None)  # type: ignore[attr-defined]


@pytest.mark.parametrize("process_id", (None, 1, "unknown"))
def test_registry_rejects_invalid_process_queries(process_id: object) -> None:
    """Test process-specific lookups validate process IDs before access."""
    registry = StreamRegistry(0, 1, ("box",), (0,), _arrays())

    with pytest.raises((TypeError, ValueError)):
        registry.words_by_lane(process_id)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        registry.state_array_for(process_id)  # type: ignore[arg-type]


def test_host_only_import_and_construction_do_not_load_optional_backends() -> (
    None
):
    """Test direct import and host manifest construction avoid optional modules."""
    source = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'warp' or name.startswith('warp.') or name == 'particula.gpu' or name.startswith('particula.gpu.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from particula.execution.rng import StreamRegistry
registry = StreamRegistry(0, 1, ('box',), (0,), (('coagulation', object()), ('wall_loss', object())))
assert registry.word_for('coagulation', 'box') >= 0
assert not any(name == 'warp' or name.startswith('particula.gpu') for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_unavailable_warp_initialization_fails_before_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test optional-backend import failure happens before an array write."""
    original_import: Any = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        """Reject only the initializer's optional Warp import."""
        if name == "warp":
            raise ModuleNotFoundError("warp unavailable")
        return original_import(name, *args, **kwargs)  # type: ignore[misc]

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    registry = StreamRegistry(0, 1, ("box",), (0,), _arrays())
    with pytest.raises(ModuleNotFoundError, match="warp unavailable"):
        registry.initialize()


def test_registry_construction_does_not_require_warp() -> None:
    """Test host metadata construction leaves opaque arrays uninspected."""
    registry = StreamRegistry(0, 0, (), (), _arrays())

    assert registry.words_by_lane("coagulation") == ()


def test_initialize_replaces_valid_caller_owned_warp_arrays() -> None:
    """Test initialization writes retained arrays by identity when Warp exists."""
    wp = pytest.importorskip("warp")
    coagulation = wp.zeros(2, dtype=wp.uint32, device="cpu")
    wall_loss = wp.zeros(2, dtype=wp.uint32, device="cpu")
    registry = StreamRegistry(
        5,
        2,
        ("first", "second"),
        (1, 0),
        (("coagulation", coagulation), ("wall_loss", wall_loss)),
    )

    registry.initialize()

    assert registry.state_array_for("coagulation") is coagulation
    assert registry.state_array_for("wall_loss") is wall_loss
    assert coagulation.numpy().tolist() == list(
        registry.words_by_lane("coagulation")
    )
    assert wall_loss.numpy().tolist() == list(
        registry.words_by_lane("wall_loss")
    )


def test_initialize_preflight_failure_preserves_both_state_arrays() -> None:
    """Test invalid state-array schemas reject before either array is overwritten."""
    wp = pytest.importorskip("warp")
    coagulation = wp.full(1, 17, dtype=wp.uint32, device="cpu")
    invalid_wall_loss = wp.full(1, 19.0, dtype=wp.float64, device="cpu")
    registry = StreamRegistry(
        5,
        1,
        ("box",),
        (0,),
        (("coagulation", coagulation), ("wall_loss", invalid_wall_loss)),
    )

    with pytest.raises(
        TypeError, match="wall_loss state array must have dtype"
    ):
        registry.initialize()

    assert coagulation.numpy().tolist() == [17]
    assert invalid_wall_loss.numpy().tolist() == [19.0]


@pytest.mark.parametrize("failure", ("shape", "dtype", "contiguous"))
def test_initialize_rejects_invalid_schema_before_copy(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Test each retained-array schema error rejects before writer work."""
    wp = pytest.importorskip("warp")
    coagulation = wp.full(1, 17, dtype=wp.uint32, device="cpu")
    wall_loss = wp.full(1, 19, dtype=wp.uint32, device="cpu")

    class InvalidArray:
        """Retain valid array metadata with one deliberate schema violation."""

        def __init__(self, array: object, kind: str) -> None:
            """Expose enough Warp-like metadata for preflight validation."""
            array_any: Any = array
            self.shape = (2,) if kind == "shape" else array_any.shape
            self.dtype = wp.float64 if kind == "dtype" else array_any.dtype
            self.device = array_any.device
            self.ptr = array_any.ptr
            self.is_contiguous = kind != "contiguous"

    if failure == "shape":
        invalid_wall_loss = InvalidArray(wall_loss, failure)
    elif failure == "dtype":
        invalid_wall_loss = InvalidArray(wall_loss, failure)
    else:
        invalid_wall_loss = InvalidArray(wall_loss, failure)
    registry = StreamRegistry(
        5,
        1,
        ("box",),
        (0,),
        (("coagulation", coagulation), ("wall_loss", invalid_wall_loss)),
    )
    copy_calls: list[object] = []
    monkeypatch.setattr(wp, "copy", lambda *args: copy_calls.append(args))

    with pytest.raises((TypeError, ValueError)):
        registry.initialize()

    assert copy_calls == []
    assert coagulation.numpy().tolist() == [17]


def test_initialize_rejects_spoofed_array_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test structural lookalikes cannot pass Warp-array preflight."""
    wp = pytest.importorskip("warp")
    coagulation = wp.full(1, 17, dtype=wp.uint32, device="cpu")
    wall_loss = wp.full(1, 19, dtype=wp.uint32, device="cpu")

    class SpoofedArray:
        """Expose copied Warp metadata without being a Warp array."""

        shape = wall_loss.shape
        dtype = wall_loss.dtype
        device = wall_loss.device
        ptr = wall_loss.ptr
        is_contiguous = wall_loss.is_contiguous

    registry = StreamRegistry(
        5,
        1,
        ("box",),
        (0,),
        (("coagulation", coagulation), ("wall_loss", SpoofedArray())),
    )
    copy_calls: list[object] = []
    monkeypatch.setattr(wp, "copy", lambda *args: copy_calls.append(args))

    with pytest.raises(TypeError, match="wall_loss state array must be a Warp"):
        registry.initialize()

    assert copy_calls == []
    assert coagulation.numpy().tolist() == [17]


def test_initialize_rejects_identical_and_overlapping_warp_arrays() -> None:
    """Test retained state arrays must be distinct nonoverlapping Warp arrays."""
    wp = pytest.importorskip("warp")
    coagulation = wp.zeros(1, dtype=wp.uint32, device="cpu")
    overlapping = wp.array(
        ptr=coagulation.ptr,
        capacity=coagulation.capacity,
        dtype=wp.uint32,
        shape=(1,),
        device="cpu",
        copy=False,
    )

    for wall_loss, message in (
        (coagulation, "distinct objects"),
        (overlapping, "must not alias"),
    ):
        registry = StreamRegistry(
            5,
            1,
            ("box",),
            (0,),
            (("coagulation", coagulation), ("wall_loss", wall_loss)),
        )
        with pytest.raises(ValueError, match=message):
            registry.initialize()


def test_initialize_rejects_different_warp_devices_when_available() -> None:
    """Test distinct real state arrays must be on the same Warp device."""
    wp = pytest.importorskip("warp")
    devices = wp.get_devices()
    if len(devices) < 2:
        pytest.skip("A second Warp device is unavailable.")
    coagulation = wp.zeros(1, dtype=wp.uint32, device=devices[0])
    wall_loss = wp.zeros(1, dtype=wp.uint32, device=devices[1])
    registry = StreamRegistry(
        5,
        1,
        ("box",),
        (0,),
        (("coagulation", coagulation), ("wall_loss", wall_loss)),
    )

    with pytest.raises(ValueError, match="same device"):
        registry.initialize()


def test_initialize_second_copy_failure_keeps_first_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a second-copy failure does not promise rollback or rebinding."""
    wp = pytest.importorskip("warp")
    coagulation = wp.full(1, 17, dtype=wp.uint32, device="cpu")
    wall_loss = wp.full(1, 19, dtype=wp.uint32, device="cpu")
    registry = StreamRegistry(
        5,
        1,
        ("box",),
        (0,),
        (("coagulation", coagulation), ("wall_loss", wall_loss)),
    )
    original_copy = wp.copy
    copy_count = 0

    def fail_second_copy(*args: object) -> None:
        """Perform the first copy and fail the second copy deterministically."""
        nonlocal copy_count
        if args[0] is not coagulation and args[0] is not wall_loss:
            original_copy(*args)
            return
        copy_count += 1
        if copy_count == 2:
            raise RuntimeError("second copy failed")
        original_copy(*args)

    monkeypatch.setattr(wp, "copy", fail_second_copy)

    with pytest.raises(RuntimeError, match="second copy failed"):
        registry.initialize()

    assert registry.state_array_for("coagulation") is coagulation
    assert registry.state_array_for("wall_loss") is wall_loss
    assert coagulation.numpy().tolist() == list(
        registry.words_by_lane("coagulation")
    )
    assert wall_loss.numpy().tolist() == [19]


def test_registry_permits_equal_words_for_different_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test derived-word collision detection is limited to one process."""
    rng_module = importlib.import_module("particula.execution.rng")
    monkeypatch.setattr(
        rng_module, "_derive_initial_word", lambda _root, _key: 3
    )

    registry = StreamRegistry(0, 1, ("box",), (0,), _arrays())

    assert registry.words_by_lane("coagulation") == (3,)
    assert registry.words_by_lane("wall_loss") == (3,)
