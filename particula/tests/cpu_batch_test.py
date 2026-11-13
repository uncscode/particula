"""Exercise the shared runnable transaction independently of physics."""

import numpy as np
import pytest
from particula import Aerosol
from particula.gas import EnvironmentData, GasData
from particula.particles import create_resolved_particle_data
from particula.runnable import RunnableABC


def state():
    """Construct two distinct boxes with unequal represented volumes."""
    return Aerosol(
        particles=create_resolved_particle_data(
            masses=np.full((2, 3, 1), 1e-18),
            density=1000.0,
            volume=np.array([0.25, 4.0]),
        ),
        gas=GasData(["water"], np.array([0.018]), np.ones((2, 1)), [True]),
        environment=EnvironmentData([280, 300], [101325, 101325], [[1], [1]]),
    )


class Process(RunnableABC):
    """Simple hook process with an optional late-box failure."""

    fail = False

    def _writer_fields(self, aerosol):
        return ("particles.concentration",)

    def _rate_box(self, aerosol):
        return aerosol.particles.concentration[0]

    def _execute_box(self, aerosol, time_step, sub_steps=1):
        aerosol.particles.concentration *= np.exp(-time_step)
        if self.fail and aerosol.environment.temperature[0] > 290:
            raise ValueError("late box failure")
        return aerosol


def test_shared_driver_identity_shapes_and_box_independence():
    """Rates keep B and execution publishes into original arrays."""
    aerosol = state()
    target = aerosol.particles.concentration
    process = Process()
    assert process.rate(aerosol).shape == (2, 3)
    assert process.execute(aerosol, 1, sub_steps=2) is aerosol
    assert aerosol.particles.concentration is target
    np.testing.assert_allclose(target, np.exp(-1))


def test_later_box_failure_preserves_whole_batch():
    """An earlier staged mutation cannot escape a rejected substep."""
    aerosol = state()
    process = Process()
    process.fail = True
    with pytest.raises(ValueError, match="late box"):
        process.execute(aerosol, 1)
    np.testing.assert_array_equal(aerosol.particles.concentration, 1)


def test_original_readonly_storage_rejects_before_staging():
    """Copies cannot conceal unwritable original destinations."""
    aerosol = state()
    aerosol.particles.concentration.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        Process().execute(aerosol, 1)


def test_zero_step_never_publishes_hook_writes():
    """A successful zero-duration call retains the original values."""
    aerosol = state()
    Process().execute(aerosol, 0)
    np.testing.assert_array_equal(aerosol.particles.concentration, 1)


@pytest.mark.parametrize("axis", [0, 1])
def test_overlapping_destinations_reject_before_hooks(axis):
    """Both inter-box and within-box overlaps reject before calculation."""
    aerosol = state()
    strides = list(aerosol.particles.concentration.strides)
    strides[axis] = 0
    aerosol.particles.concentration = np.lib.stride_tricks.as_strided(
        aerosol.particles.concentration, shape=(2, 3), strides=strides
    )
    with pytest.raises(ValueError, match="overlap"):
        Process().execute(aerosol, 1)
    np.testing.assert_array_equal(aerosol.particles.concentration, 1)


def test_publication_failure_restores_all_written_fields(monkeypatch):
    """A failed second copy rolls back the first field and partial second."""
    from particula import _cpu_batch

    aerosol = state()
    staged = _cpu_batch.detached(aerosol)
    staged.particles.concentration *= 2
    staged.gas.concentration *= 3
    original_copy = np.copyto
    calls = 0

    def fail_second(target, source):
        nonlocal calls
        calls += 1
        original_copy(target, source)
        if calls == 2:
            raise RuntimeError("write failure")

    monkeypatch.setattr(np, "copyto", fail_second)
    with pytest.raises(RuntimeError, match="write failure"):
        _cpu_batch.publish(
            aerosol, staged, ("particles.concentration", "gas.concentration")
        )
    np.testing.assert_array_equal(aerosol.particles.concentration, 1)
    np.testing.assert_array_equal(aerosol.gas.concentration, 1)


def test_sequence_order_and_adapter_context_delegation():
    """The adapter delegates once and the loop is substep/process/box."""
    from particula.execution import (
        CPUExecutionAdapter,
        CPUExecutionContext,
        CPUExecutionState,
    )

    events = []

    class RecordingProcess(Process):
        def __init__(self, name):
            self.name = name

        def _box_worker(self, box, boxes, transaction):
            assert transaction is not None
            return super()._box_worker(box, boxes, transaction)

        def _execute_box(self, aerosol, time_step, sub_steps=1):
            events.append((self.name, aerosol.environment.temperature[0]))
            return super()._execute_box(aerosol, time_step, sub_steps)

    sequence = RecordingProcess("a") | RecordingProcess("b")
    aerosol = state()
    carrier = CPUExecutionState(
        aerosol, 1.0, 2, CPUExecutionContext(1, ("cold", "warm"))
    )
    CPUExecutionAdapter(sequence).execute(carrier)
    assert events == [("a", 280), ("a", 300), ("b", 280), ("b", 300)] * 2
    np.testing.assert_allclose(aerosol.particles.concentration, np.exp(-2))


def test_interleaved_nonoverlapping_storage_is_supported():
    """A nonstandard but disjoint layout retains its original array identity."""
    aerosol = state()
    backing = np.ones(8)
    target = np.lib.stride_tricks.as_strided(
        backing, shape=(2, 3), strides=(24, 16)
    )
    aerosol.particles.concentration = target
    Process().execute(aerosol, 1)
    assert aerosol.particles.concentration is target
    np.testing.assert_allclose(target, np.exp(-1))


@pytest.mark.parametrize("fail", [False, True])
def test_environment_hook_writes_are_transactional(fail):
    """Declared environment writes publish by identity or roll back together."""

    class Heating(Process):
        def _writer_fields(self, aerosol):
            return (
                "environment.temperature",
                "environment.pressure",
                "environment.saturation_ratio",
            )

        def _execute_box(self, aerosol, time_step, sub_steps=1):
            aerosol.environment.temperature += time_step
            aerosol.environment.pressure += time_step
            aerosol.environment.saturation_ratio += time_step
            if fail and aerosol.environment.temperature[0] > 300:
                raise ValueError("late heating failure")
            return aerosol

    aerosol = state()
    fields = (
        aerosol.environment.temperature,
        aerosol.environment.pressure,
        aerosol.environment.saturation_ratio,
    )
    snapshots = [value.copy() for value in fields]
    if fail:
        with pytest.raises(ValueError, match="late heating"):
            Heating().execute(aerosol, 1)
    else:
        Heating().execute(aerosol, 1)
    for field, before in zip(fields, snapshots, strict=True):
        np.testing.assert_array_equal(field, before + (0 if fail else 1))
    assert aerosol.environment.temperature is fields[0]
