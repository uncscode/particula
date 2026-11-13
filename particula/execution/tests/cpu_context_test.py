"""Verify CPU stream identity, isolation, and transactional continuation."""

import numpy as np
import pytest

from particula.execution import CPUExecutionContext


def _draw(context, namespace="coagulation", size=8, commit=True):
    transaction = context._begin(len(context.box_ids))
    result = {
        name: transaction.generator(namespace, box).random(size)
        for box, name in enumerate(context.box_ids)
    }
    if commit:
        transaction.commit()
    return result


def test_box_order_and_unrelated_boxes_do_not_change_streams():
    """Box identity is independent of lane and batch membership."""
    full = CPUExecutionContext(17, ("a", "b", "c"))
    alone = CPUExecutionContext(17, ("b",))
    expected = _draw(alone)["b"]
    np.testing.assert_array_equal(_draw(full)["b"], expected)
    reordered = full.with_box_ids(("c", "b", "new"))
    np.testing.assert_array_equal(_draw(reordered)["b"], _draw(alone)["b"])
    # The view and original context continue the same stream.
    np.testing.assert_array_equal(_draw(full)["b"], _draw(alone)["b"])


def test_abandoned_transaction_preserves_continuation():
    """Failure before publication leaves acquired and new streams unchanged."""
    actual = CPUExecutionContext(23, ("a", "b"))
    reference = CPUExecutionContext(23, ("a", "b"))
    for _ in range(2):
        _draw(actual, size=100, commit=False)
        result = _draw(actual)
        expected = _draw(reference)
        for key in result:
            np.testing.assert_array_equal(result[key], expected[key])


def test_namespace_and_seed_separate_streams():
    """Different processes and seeds do not reuse identical draws."""
    first = CPUExecutionContext(1, ("a",))
    reference = CPUExecutionContext(1, ("a",))
    coagulation = _draw(first)["a"]
    wall = _draw(first, "wall_loss")["a"]
    np.testing.assert_array_equal(wall, _draw(reference, "wall_loss")["a"])
    assert not np.array_equal(coagulation, wall)
    assert not np.array_equal(
        coagulation, _draw(CPUExecutionContext(2, ("a",)))["a"]
    )


def test_staged_generator_is_reused_within_transaction():
    """Repeated acquisition continues one staged stream."""
    context = CPUExecutionContext(42, ("a",))
    transaction = context._begin(1)
    generator = transaction.generator("condensation", 0)
    assert transaction.generator("condensation", 0) is generator
    generator.random(3)
    expected = generator.random(3)
    reference = context._begin(1).generator("condensation", 0)
    np.testing.assert_array_equal(reference.random(6)[3:], expected)


@pytest.mark.parametrize("seed", [True, np.bool_(True), -1, 2.0, "1"])
def test_invalid_seed_rejected(seed):
    """Seeds cannot silently truncate or coerce."""
    with pytest.raises(ValueError, match="root_seed"):
        CPUExecutionContext(seed, ("a",))


@pytest.mark.parametrize("ids", [(), ("",), ("a", "a"), (1,)])
def test_invalid_identity_rejected(ids):
    """Reject empty, duplicate, and non-string identities."""
    with pytest.raises(ValueError, match="box_ids"):
        CPUExecutionContext(1, ids)


def test_context_layout_and_immutable_metadata():
    """Binding requires exact box count and protects context metadata."""
    context = CPUExecutionContext(np.int64(4), ("a",))
    with pytest.raises(ValueError, match="box count"):
        context._begin(2)
    with pytest.raises(AttributeError):
        context.box_ids = ("b",)
    with pytest.raises(AttributeError):
        context.root_seed = 2
    with pytest.raises(TypeError, match="tuple"):
        context.with_box_ids(["a"])
