"""Public Python processes preserve staging and physical-time semantics."""

import numpy as np
import pytest
from particula import FunctionProcess
from particula.dynamics.tests.native_runnable_test import state


def test_sequence_uses_shared_substep_time():
    """Each operation sees the same interval, without advancing per process."""
    seen = []

    def advance(aerosol, *, time, time_step, context):
        seen.append((time, time_step))
        aerosol.environment.temperature[:] += time_step

    first = FunctionProcess(
        advance, name="first", writes=("environment.temperature",)
    )
    second = FunctionProcess(
        advance, name="second", writes=("environment.temperature",)
    )
    aerosol = state()
    (first | second).execute(aerosol, 2, sub_steps=2, time=10)
    assert seen == [(10, 1), (10, 1), (11, 1), (11, 1)]
    np.testing.assert_allclose(aerosol.environment.temperature, [302.15])


@pytest.mark.parametrize("failure", ["raise", "replace", "invalid"])
def test_rejected_callback_preserves_original(failure):
    """No candidate write leaks through exception or postvalidation failure."""
    aerosol = state()
    original = aerosol.gas.concentration.copy()

    def advance(box, **_):
        box.gas.concentration[:] = 7
        if failure == "raise":
            raise ValueError("callback")
        if failure == "replace":
            box.gas.concentration = box.gas.concentration.copy()
        if failure == "invalid":
            box.gas.concentration[:] = -1

    process = FunctionProcess(
        advance, name="reject", writes=("gas.concentration",)
    )
    with pytest.raises(ValueError):
        process.execute(aerosol, 1)
    np.testing.assert_array_equal(aerosol.gas.concentration, original)


def test_zero_duration_does_not_call_user():
    """A zero interval validates declarations without executing user code."""

    def forbidden(*_, **__):
        pytest.fail("callback invoked")

    process = FunctionProcess(forbidden, name="noop")
    process.execute(state(), 0)


def test_late_box_failure_rolls_back_data_and_rng():
    """Rejected batches consume neither numeric state nor managed streams."""
    from particula import CPUExecutionContext
    from particula.tests.cpu_batch_test import state as batch_state

    aerosol = batch_state()
    reference = batch_state()
    context = CPUExecutionContext(42, ("a", "b"))
    fresh = CPUExecutionContext(42, ("a", "b"))

    def update(box, *, context, **_):
        box.gas.concentration[:] += context.rng.random()

    def fail(box, *, context, **kwargs):
        update(box, context=context, **kwargs)
        if context.box_id == "b":
            raise ValueError("late box")

    bad = FunctionProcess(
        fail, name="source", stochastic=True, writes=("gas.concentration",)
    )
    good = FunctionProcess(
        update, name="source", stochastic=True, writes=("gas.concentration",)
    )
    with pytest.raises(ValueError, match="late box"):
        bad.execute(aerosol, 1, context=context)
    np.testing.assert_array_equal(aerosol.gas.concentration, 1)
    good.execute(aerosol, 1, context=context)
    good.execute(reference, 1, context=fresh)
    np.testing.assert_array_equal(
        aerosol.gas.concentration, reference.gas.concentration
    )


@pytest.mark.parametrize("mutation", ["species", "container", "readonly"])
def test_metadata_and_undeclared_changes_reject(mutation):
    """Callbacks cannot replace containers or alter undeclared state."""
    from copy import copy

    aerosol = state()

    def update(box, **_):
        if mutation == "species":
            box.gas.name[0] = "other"
        elif mutation == "container":
            box._gas = copy(box.gas)
        else:
            box.particles.charge[:] = 0

    with pytest.raises(ValueError):
        FunctionProcess(update, name="bad").execute(aerosol, 1)
    assert aerosol.gas.name == ["inert", "B", "A"]
    np.testing.assert_array_equal(aerosol.particles.charge, [[1, -2]])


def test_duplicate_instance_names_reject_before_any_callback():
    """Independent stream ownership requires explicit distinct instances."""

    def forbidden(*_, **__):
        pytest.fail("callback invoked before complete sequence validation")

    process = FunctionProcess(forbidden, name="duplicate")
    with pytest.raises(ValueError, match="unique"):
        (process | process).execute(state(), 1)


def test_validator_cannot_mutate_state():
    """Requirements are checked on isolated state and must be read-only."""

    def validate(box):
        box.gas.concentration[:] = 0

    process = FunctionProcess(
        lambda *a, **k: None, name="bad", validate=validate
    )
    aerosol = state()
    before = aerosol.gas.concentration.copy()
    with pytest.raises(ValueError, match="undeclared"):
        process.execute(aerosol, 0)
    np.testing.assert_array_equal(aerosol.gas.concentration, before)
