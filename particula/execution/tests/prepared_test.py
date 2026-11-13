"""Prepared ownership, transfers, stream selection and failure boundaries."""

import numpy as np
import pytest

from particula._cpu_batch import arrays
from particula.dynamics.tests.native_runnable_test import state
from particula.execution import Backend, Device
from particula.execution.prepared import PreparedSimulation

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Require actual CUDA for the CUDA parameter."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return Device(Backend.WARP, request.param)


def upload(aerosol, device):
    """Exercise ownership before process translation is connected."""
    return PreparedSimulation._upload(
        aerosol,
        device=device,
        time_step=0.125,
        root_seed=92,
        logical_box_ids=("chamber",),
    )


@wp.kernel
def _consume_gas(values: wp.array2d(dtype=wp.float64)):
    box, species = wp.tid()
    values[box, species] *= wp.float64(0.5)


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_snapshot_detaches_every_field_and_preserves_semantics(device, kind):
    """Names, independent species widths and raw counts survive both ways."""
    initial = state(
        volume=0.25 if kind == "particle_resolved" else 1.0, kind=kind
    )
    baseline = {name: value.copy() for name, value in arrays(initial).items()}
    with upload(initial, device) as simulation:
        first = simulation.snapshot()
        second = simulation.snapshot()
        assert first.gas.name == ["inert", "B", "A"]
        assert first.particles.distribution_type == kind
        assert first.particles.n_species == 2
        assert first.gas.concentration.shape == (1, 3)
        for name, value in arrays(first).items():
            np.testing.assert_array_equal(value, baseline[name])
            assert not np.shares_memory(value, arrays(initial)[name])
            assert not np.shares_memory(value, arrays(second)[name])
            value[...] = 0
        first.gas.name[0] = "changed"
        initial.gas.concentration[:] = 1
        initial.particles.masses[:] = 1
        third = simulation.snapshot()
        for name, value in arrays(third).items():
            np.testing.assert_array_equal(value, baseline[name])
        assert third.gas.name == ["inert", "B", "A"]
        assert simulation.device == device
        assert simulation._session.metadata.stream.logical_box_ids == (
            "chamber",
        )
    assert simulation.closed
    assert (
        simulation._session is simulation._registry is simulation._guard is None
    )
    simulation.close()
    for observe in (
        simulation.snapshot,
        simulation.synchronize,
        simulation.step,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            observe()


def test_enqueue_uses_owned_stream_and_no_host_observation(device, monkeypatch):
    """Prepared stepping retains buffers and does no synchronization/readback."""
    initial = state()
    with upload(initial, device) as simulation:
        gas = simulation._session.gas.concentration

        def enqueue():
            if device.native != "cpu":
                assert wp.get_stream(device.native) is simulation._stream
            wp.launch(
                _consume_gas, dim=gas.shape, inputs=[gas], device=device.native
            )

        simulation._operations = (enqueue,)
        # Compile/load before applying the no-allocation observation guard.
        wp.load_module(device=device.native)
        saved_stream = (
            wp.get_stream(device.native) if device.native != "cpu" else None
        )

        def forbidden(*_args, **_kwargs):
            pytest.fail("a prepared step performed host work or allocation")

        with monkeypatch.context() as patch:
            for name in (
                "synchronize",
                "synchronize_device",
                "synchronize_stream",
                "zeros",
                "empty",
            ):
                patch.setattr(wp, name, forbidden)
            patch.setattr(wp.array, "numpy", forbidden)
            simulation.run(3)
        if saved_stream is not None:
            assert wp.get_stream(device.native) is saved_stream
        assert simulation.completed_steps == 3
        assert simulation.simulated_time == 0.375
        np.testing.assert_array_equal(
            simulation.snapshot().gas.concentration,
            initial.gas.concentration / 8,
        )
    assert simulation.completed_steps == 3
    assert simulation.simulated_time == 0.375


def test_writer_failure_faults_and_close_never_downloads(device, monkeypatch):
    """A failed launch sequence cannot be reused or inspected as successful."""
    simulation = upload(state(), device)
    gas = simulation._session.gas.concentration

    def failing_operation():
        wp.launch(
            _consume_gas, dim=gas.shape, inputs=[gas], device=device.native
        )
        raise RuntimeError("writer failed")

    simulation._operations = (failing_operation,)
    with pytest.raises(RuntimeError, match="writer failed"):
        simulation.step()
    assert simulation.completed_steps == 0
    with pytest.raises(RuntimeError, match="faulted"):
        simulation.snapshot()
    monkeypatch.setattr(
        wp.array, "numpy", lambda *_: pytest.fail("close downloaded")
    )
    simulation.close()
    assert simulation.closed


@pytest.mark.parametrize("steps", [-1, 0.5, True])
def test_invalid_run_control_does_not_enqueue(device, steps):
    """Invalid work counts cannot advance simulation time."""
    with upload(state(), device) as simulation:
        simulation._operations = (lambda: pytest.fail("invalid run enqueued"),)
        with pytest.raises((TypeError, ValueError)):
            simulation.run(steps)
        assert simulation.completed_steps == 0


def test_unsupported_population_rejects_before_device_lookup(monkeypatch):
    """PDF state cannot allocate or silently fall back to a different backend."""
    monkeypatch.setattr(
        wp, "get_device", lambda *_: pytest.fail("device lookup")
    )
    with pytest.raises(ValueError, match="continuous_pdf"):
        upload(
            state(volume=1.0, kind="continuous_pdf"),
            Device(Backend.WARP, "cuda:0"),
        )


def test_async_failure_faults_on_observation(device, monkeypatch):
    """A deferred runtime error faults a binding even after enqueue returns."""
    simulation = upload(state(), device)
    simulation._operations = ()
    simulation.run(2)

    def failure():
        raise RuntimeError("asynchronous failure")

    with monkeypatch.context() as patch:
        patch.setattr(simulation, "_wait", failure)
        with pytest.raises(RuntimeError, match="asynchronous failure"):
            simulation.synchronize()
    with pytest.raises(RuntimeError, match="faulted"):
        simulation.run(1)
    simulation.close()


def test_context_exception_closes_owner(device):
    """Caller exceptions release resources without requiring a snapshot."""
    with pytest.raises(ValueError, match="analysis failed"):
        with upload(state(), device) as simulation:
            raise ValueError("analysis failed")
    assert simulation.closed
