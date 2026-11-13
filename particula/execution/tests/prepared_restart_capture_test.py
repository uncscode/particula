"""Canonical restart and actual CUDA capture of prepared process subsets."""

from dataclasses import replace

import numpy as np
import pytest

from particula._cpu_batch import arrays
from particula.dynamics import (
    BrownianCoagulationStrategy,
    Coagulation,
    Dilution,
    DilutionStrategy,
    WallLoss,
)
from particula.dynamics.tests.native_runnable_test import state
from particula.dynamics.wall_loss.wall_loss_strategies import (
    SphericalWallLossStrategy,
)
from particula.execution import SimulationCheckpoint
from particula.execution.prepared import PreparedSimulation
from particula.execution.tests.prepared_test import (
    condensation_case,
    nucleation_case,
)
from particula.particles import ParticleData

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Keep explicit Warp CPU regression separate from RTX evidence."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def scenario(name):
    """Use nonzero physics and unequal particle/gas widths for every family."""
    if name == "condensation":
        initial, process = condensation_case(
            latent=True, kappa=True, weighted=True
        )
        return initial, process, 0.01
    if name.startswith("nucleation"):
        initial, process = nucleation_case(resampling=True)
        if name == "nucleation_resampling":
            initial.particles.masses[0, 1:3] = initial.particles.masses[0, 0]
            initial.particles.concentration[0, :3] = 1
        return initial, process, 1.0
    initial = state(volume=1e-13)
    old = initial.particles
    initial.particles = ParticleData(
        masses=np.tile(old.masses, (1, 32, 1)),
        concentration=np.ones((1, 64)),
        charge=np.tile(old.charge, (1, 32)),
        density=old.density.copy(),
        volume=old.volume.copy(),
        distribution_type="particle_resolved",
    )
    dilution = Dilution(DilutionStrategy(0.2))
    coagulation = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    wall = WallLoss(
        SphericalWallLossStrategy(
            wall_eddy_diffusivity=0.1,
            chamber_radius=0.1,
            distribution_type="particle_resolved",
        )
    )
    choices = {
        "dilution": dilution,
        "coagulation": coagulation,
        "wall_loss": wall,
        "sequence": coagulation | dilution | wall,
    }
    return initial, choices[name], 0.1


def canonical_equal(left, right):
    """Compare authoritative bytes, streams, counters and last-step energy."""
    assert left.processes == right.processes
    assert left.time_step == right.time_step
    assert left.resident.payloads == right.resident.payloads
    assert left.resident.rng_continuation == right.resident.rng_continuation
    assert left.resident.completed_steps == right.resident.completed_steps
    assert left.resident.simulated_time == right.resident.simulated_time
    assert left.auxiliary == right.auxiliary


@pytest.mark.parametrize(
    "name",
    [
        "dilution",
        "coagulation",
        "wall_loss",
        "condensation",
        "nucleation",
        "nucleation_resampling",
        "sequence",
    ],
)
def test_restart_preserves_canonical_bytes_and_future_trajectory(device, name):
    """Resume survives edited inspection state and keeps independent ownership."""
    initial, process, dt = scenario(name)
    with process.prepare(
        initial,
        device=device,
        time_step=dt,
        root_seed=17,
        logical_box_ids=("chamber",),
    ) as original:
        original.step()
        if name.startswith("nucleation"):
            # Existing source admits integral unit events. Shut off the pulse
            # explicitly rather than silently rounding the remaining demand.
            original.update_gas(np.zeros_like(initial.gas.concentration))
        saved = original.checkpoint()
        assert isinstance(saved, SimulationCheckpoint)
        saved.resident.particles.masses[:] = -123
        saved.resident.gas.name[:] = ["edited"] * len(saved.resident.gas.name)
        saved.resident.environment.temperature[:] = -1
        with process.resume(saved) as restored:
            assert not restored.captured
            assert restored._session is not original._session
            assert (
                restored._session.particles.masses
                is not original._session.particles.masses
            )
            canonical_equal(saved, restored.checkpoint())
            original.run(3)
            restored.run(1)
            restored.run(2)
            canonical_equal(original.checkpoint(), restored.checkpoint())
            for key, value in arrays(original.snapshot()).items():
                np.testing.assert_array_equal(
                    value, arrays(restored.snapshot())[key]
                )
        assert original.completed_steps == 4


@pytest.mark.parametrize(
    "change",
    [
        "order",
        "coefficient",
        "device",
        "schema",
        "payload",
        "time_step",
        "auxiliary",
    ],
)
def test_restart_rejects_drift_before_upload(device, change, monkeypatch):
    """Changing process semantics or corrupting continuation cannot allocate."""
    initial, process, dt = scenario("sequence")
    with process.prepare(initial, device=device, time_step=dt) as simulation:
        saved = simulation.checkpoint()
    kwargs = {}
    if change == "order":
        process = (
            process.processes[1] | process.processes[0] | process.processes[2]
        )
    elif change == "coefficient":
        process.processes[1].dilution_strategy.coefficient = 0.4
    elif change == "device":
        kwargs["device"] = "cuda:1" if device != "cpu" else "cuda:0"
    elif change == "schema":
        saved = replace(saved, schema_version=999)
    elif change == "payload":
        bad = replace(saved.resident.payloads[0], data=b"bad")
        saved = replace(
            saved,
            resident=replace(
                saved.resident, payloads=(bad, *saved.resident.payloads[1:])
            ),
        )
    elif change == "auxiliary":
        saved = replace(saved, auxiliary=(saved.resident.payloads[0],))
    else:
        saved = replace(saved, time_step=float("nan"))
    monkeypatch.setattr(
        PreparedSimulation,
        "_empty",
        lambda *_: pytest.fail("restart allocated"),
    )
    with pytest.raises((TypeError, ValueError)):
        process.resume(saved, **kwargs)


def test_cpu_capture_rejects_without_advancing():
    """An explicit Warp CPU run cannot be described as CUDA graph evidence."""
    initial, process, dt = scenario("dilution")
    with process.prepare(initial, device="cpu", time_step=dt) as simulation:
        with pytest.raises(ValueError, match="CUDA"):
            simulation.capture()
        assert simulation.completed_steps == 0
        simulation.step()
        assert simulation.completed_steps == 1


def test_restart_rechecks_evolved_equal_count_capability(device, monkeypatch):
    """A late source can create a state that the next Brownian step rejects."""
    initial, source = nucleation_case()
    initial.particles.concentration *= 2
    process = (
        Coagulation(BrownianCoagulationStrategy("particle_resolved")) | source
    )
    with process.prepare(initial, device=device, time_step=1.0) as simulation:
        simulation.step()
        saved = simulation.checkpoint()
        monkeypatch.setattr(
            PreparedSimulation,
            "_empty",
            lambda *_: pytest.fail("restart allocated"),
        )
        with pytest.raises(ValueError, match="equal active counts"):
            process.resume(saved)
        simulation.step()
        with pytest.raises(ValueError, match="equal active counts"):
            simulation.synchronize()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "name",
    [
        "dilution",
        "coagulation",
        "wall_loss",
        "condensation",
        "nucleation",
        "nucleation_resampling",
        "sequence",
    ],
)
def test_capture_records_without_advancing_and_replays_exactly(name):
    """Actual CUDA graphs preserve step counts, forcing and stochastic paths."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process, dt = scenario(name)
    with process.prepare(initial, time_step=dt, root_seed=47) as ordinary:
        with process.prepare(initial, time_step=dt, root_seed=47) as captured:
            before = captured.checkpoint()
            captured.capture()
            assert captured.captured
            assert captured.completed_steps == 0
            canonical_equal(before, captured.checkpoint())
            for step in range(3):
                for simulation in (ordinary, captured):
                    if step == 1:
                        simulation.update_environment(
                            temperature=285.0, pressure=99000.0
                        )
                        if name.startswith("nucleation"):
                            simulation.update_gas(
                                np.zeros_like(initial.gas.concentration)
                            )
                    simulation.step()
                canonical_equal(ordinary.checkpoint(), captured.checkpoint())
            saved = captured.checkpoint()
            with process.resume(saved) as resumed:
                resumed.capture()
                resumed.step()
                captured.step()
                canonical_equal(resumed.checkpoint(), captured.checkpoint())
            with pytest.raises(ValueError, match="already owns"):
                captured.capture()
            captured.uncapture()
            assert not captured.captured
            captured.step()
            captured.capture()
            assert captured.completed_steps == 5
        assert not captured.captured
        assert captured.closed


@pytest.mark.cuda
@pytest.mark.parametrize("failure", ["dispatch", "launch", "binding"])
def test_capture_failure_revokes_replay_and_closes(failure, monkeypatch):
    """Recording errors, writer errors and drift leave no usable stale graph."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process, dt = scenario("dilution")
    simulation = process.prepare(initial, time_step=dt)

    def failed(*_args, **_kwargs):
        raise RuntimeError("injected capture failure")

    try:
        if failure == "dispatch":
            simulation._operations = (failed,)
            with pytest.raises(RuntimeError, match="injected"):
                simulation.capture()
            assert not simulation._stream.is_capturing
        else:
            simulation.capture()
            if failure == "launch":
                monkeypatch.setattr(wp, "capture_launch", failed)
            else:
                simulation._operations = (lambda: None,)
            with pytest.raises((ValueError, RuntimeError)):
                simulation.step()
        assert not simulation.captured
        assert simulation.completed_steps == 0
        with pytest.raises(RuntimeError, match="faulted"):
            simulation.step()
    finally:
        simulation.close()
    assert simulation._capture is None


@pytest.mark.cuda
def test_captured_steps_do_not_dispatch_transfer_or_allocate(monkeypatch):
    """Replay launches its recorded graph with no Python physics dispatch."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process, dt = scenario("dilution")
    with process.prepare(initial, time_step=dt) as simulation:
        simulation.capture()

        def forbidden(*_args, **_kwargs):
            pytest.fail(
                "captured replay performed host dispatch or observation"
            )

        with monkeypatch.context() as patch:
            for name in (
                "launch",
                "copy",
                "zeros",
                "empty",
                "synchronize",
                "synchronize_stream",
                "synchronize_device",
                "capture_begin",
            ):
                patch.setattr(wp, name, forbidden)
            patch.setattr(wp.array, "numpy", forbidden)
            simulation.run(4)
        result = simulation.snapshot()
        assert simulation.completed_steps == 4
        np.testing.assert_allclose(
            result.gas.concentration,
            initial.gas.concentration * np.exp(-0.2 * dt * 4),
            rtol=1e-14,
        )


@pytest.mark.cuda
def test_captured_device_failure_stops_later_queued_writers():
    """A fractional second pulse latches failure across graph replays."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, source = nucleation_case()
    process = source | Dilution(DilutionStrategy(0.2))
    with process.prepare(initial, time_step=1.0) as simulation:
        simulation.capture()
        simulation.step()
        first = simulation.snapshot()
        simulation.run(3)
        with pytest.raises(ValueError):
            simulation.synchronize()
        assert not simulation.captured
        for family, names in (
            ("particles", ("masses", "concentration", "charge", "volume")),
            ("gas", ("concentration",)),
        ):
            for name in names:
                np.testing.assert_array_equal(
                    getattr(getattr(simulation._session, family), name).numpy(),
                    getattr(getattr(first, family), name),
                )
        with pytest.raises(RuntimeError, match="faulted"):
            simulation.checkpoint()


@pytest.mark.cuda
def test_graph_lifetime_ends_on_uncapture_and_close():
    """Warp's native graph finalizer can run after each explicit release."""
    import weakref

    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    initial, process, dt = scenario("dilution")
    simulation = process.prepare(initial, time_step=dt)
    simulation.capture()
    first = weakref.ref(simulation._capture.graph)
    simulation.step()
    simulation.uncapture()
    assert first() is None
    simulation.capture()
    second = weakref.ref(simulation._capture.graph)
    simulation.step()
    simulation.close()
    simulation.close()
    assert second() is None
    assert simulation.completed_steps == 2
