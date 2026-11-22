"""Coupled event scheduling, explicit transfers and exact restart evidence."""

from dataclasses import replace

import numpy as np
import pytest

from particula import _cpu_batch
from particula.dynamics import Dilution, DilutionStrategy
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.gpu_coupled import (
    GPUCondensationControls,
    GPUCoupledSimulation,
)
from particula.execution.gpu_coupled_restart import (
    load_coupled_checkpoint,
    resume_coupled,
    save_coupled_checkpoint,
)
from particula.execution.tests.condensation_ode_test import case
from particula.execution.tests.coupled_event_case import coupled_event_case
from particula.execution.tests.prepared_restart_capture_test import (
    canonical_equal,
)

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Require native CUDA for the CUDA half of each regression."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    return request.param


def equal_native(left, right):
    """Compare all primary native arrays, independently of owner helpers."""
    for family, names in (
        (
            "particles",
            ("masses", "concentration", "charge", "density", "volume"),
        ),
        ("gas", ("concentration", "molar_mass", "partitioning")),
        ("environment", ("temperature", "pressure")),
    ):
        for name in names:
            np.testing.assert_array_equal(
                getattr(getattr(left, family), name),
                getattr(getattr(right, family), name),
            )


@pytest.mark.parametrize("order", ["events-first", "condensation-first"])
def test_order_matches_explicit_flows_and_independent_budget(device, order):
    """Retain exact dilution, native identity and explicit sequential order."""
    from dataclasses import asdict

    initial, strategy = case(2)
    frozen = _cpu_batch.detached(initial)
    events = (Dilution(DilutionStrategy(0.2)),)
    controls = GPUCondensationControls()
    dt = 0.01
    reference = _cpu_batch.detached(initial)
    if order == "events-first":
        reference.particles.concentration *= np.exp(-0.2 * dt)
        reference.gas.concentration *= np.exp(-0.2 * dt)
    flow = GPUCondensationODE(
        reference, strategy, device=device, **asdict(controls)
    )
    while flow.time < dt:
        flow.step(dt)
    reference = flow.snapshot()
    if order == "condensation-first":
        reference.particles.concentration *= np.exp(-0.2 * dt)
        reference.gas.concentration *= np.exp(-0.2 * dt)
    with GPUCoupledSimulation(
        initial,
        events,
        strategy,
        device=device,
        time_step=dt,
        order=order,
    ) as owner:
        bindings = owner._events._session.particles.masses
        owner.step()
        final = owner.snapshot()
        equal_native(final, reference)
        assert owner._events._session.particles.masses is bindings
        assert owner.time == dt
        assert owner.completed_steps == 1
        assert owner.coupling_downloads == owner.coupling_uploads == 2
        before = (
            initial.gas.concentration[:, 1]
            + np.sum(
                initial.particles.masses[..., 0]
                * initial.particles.concentration,
                axis=1,
            )
            / initial.particles.volume
        )
        after = (
            final.gas.concentration[:, 1]
            + np.sum(
                final.particles.masses[..., 0] * final.particles.concentration,
                axis=1,
            )
            / final.particles.volume
        )
        np.testing.assert_allclose(
            after, before * np.exp(-0.2 * dt), rtol=3e-13
        )
    equal_native(initial, frozen)


def test_rejection_cannot_repeat_stochastic_events(device, monkeypatch):
    """Rejected condensation trials leave the fixed event RNG schedule intact."""
    initial, source, collision, strategy = coupled_event_case()
    events = (source, collision)
    called = 0
    original = GPUCondensationODE._trial

    def reject_once(self, h):
        nonlocal called
        called += 1
        if called == 1:
            return 8.0
        return original(self, h)

    with (source | collision).prepare(
        initial, device=device, time_step=1, root_seed=17
    ) as reference:
        reference.step()
        event_checkpoint = reference.checkpoint()
    with GPUCoupledSimulation(
        initial,
        events,
        strategy,
        device=device,
        time_step=1,
        root_seed=17,
    ) as owner:
        monkeypatch.setattr(GPUCondensationODE, "_trial", reject_once)
        owner.step()
        actual = owner.checkpoint()
        assert owner.statistics.rejected >= 1
        assert actual.events.resident.completed_steps == 1
        assert (
            actual.events.resident.rng_continuation
            == event_checkpoint.resident.rng_continuation
        )
        final = owner.snapshot()
        np.testing.assert_array_equal(
            final.particles.concentration,
            event_checkpoint.resident.particles.concentration,
        )
        np.testing.assert_array_equal(
            final.particles.charge, event_checkpoint.resident.particles.charge
        )


@pytest.mark.parametrize("capture", [False, True])
def test_disk_restart_preserves_events_physics_and_random_words(
    device, capture, tmp_path
):
    """A complete cycle checkpoint preserves stochastic continuation exactly."""
    if capture and device == "cpu":
        pytest.skip("native CUDA graph test")
    initial, source, collision, strategy = coupled_event_case()
    # Deliver the same two integral source events in the first 0.01 s pulse;
    # the precursor is explicitly disabled immediately after that pulse.
    source.source_config = replace(
        source.source_config,
        strategy=replace(source.source_config.strategy, coefficient=25.0),
    )
    events = (source, collision)
    # Resolve the growing-population collision demand explicitly, preserving
    # the original 1 s checkpoint and 3 s final physical observation times.
    with GPUCoupledSimulation(
        initial, events, strategy, device=device, time_step=0.01, root_seed=42
    ) as owner:
        if capture:
            before = owner.checkpoint()
            owner.capture_events()
            canonical_equal(before.events, owner.checkpoint().events)
        owner.step()
        collision_binding = owner._events._bindings["coagulation"]
        np.testing.assert_array_equal(
            owner._events._session.particles.concentration.numpy().sum(axis=1)
            + collision_binding.n_collisions.numpy(),
            initial.particles.concentration.sum(axis=1) + 2,
        )
        gas = owner.diagnostics().gas_concentration.copy()
        gas[:, 1] = 0
        owner.update_gas(gas)
        owner.run(99)
        owner.update_environment(temperature=np.array([288.0, 299.0]))
        saved = owner.checkpoint()
        # Inspection carriers remain lossy, non-authoritative host views.
        saved.events.resident.particles.masses[:] = -1
        path = tmp_path / "coupled.npz"
        save_coupled_checkpoint(saved, path)
        loaded = load_coupled_checkpoint(path)
        owner.run(200)
        final, final_checkpoint = owner.snapshot(), owner.checkpoint()
        with resume_coupled(loaded, events, strategy, device=device) as resumed:
            if capture:
                resumed.capture_events()
            resumed.run(200)
            equal_native(final, resumed.snapshot())
            canonical_equal(
                final_checkpoint.events, resumed.checkpoint().events
            )
            assert final_checkpoint.statistics == resumed.statistics
            assert final_checkpoint.counters == resumed.checkpoint().counters
            assert resumed.time == owner.time
            assert owner.time == pytest.approx(3.0)


def test_short_bound_rejects_before_events_and_failure_is_terminal(
    device, monkeypatch
):
    """Never expose partially advanced physics as an accepted complete cycle."""
    initial, strategy = case(2)
    with GPUCoupledSimulation(
        initial,
        (Dilution(DilutionStrategy(0.2)),),
        strategy,
        device=device,
        time_step=0.1,
    ) as owner:
        before = owner.checkpoint()
        with pytest.raises(ValueError, match="complete fixed event"):
            owner.step(0.05)
        canonical_equal(before.events, owner.checkpoint().events)

        def fail(*args):
            raise RuntimeError("injected condensation failure")

        monkeypatch.setattr(owner, "_condense", fail)
        with pytest.raises(RuntimeError, match="injected"):
            owner.step()
        assert owner.time == 0
        assert owner.completed_steps == 0
        assert owner._events.completed_steps == 1
        for action in (owner.snapshot, owner.checkpoint, owner.step):
            with pytest.raises(RuntimeError, match="faulted"):
                action()


def test_bad_restart_does_not_replace_last_archive_or_allocate(
    device, tmp_path, monkeypatch
):
    """Reject unknown schemas and changed condensation before creating owners."""
    initial, strategy = case()
    events = (Dilution(DilutionStrategy(0.2)),)
    with GPUCoupledSimulation(
        initial, events, strategy, device=device, time_step=0.1
    ) as owner:
        saved = owner.checkpoint()
    path = tmp_path / "checkpoint.npz"
    save_coupled_checkpoint(saved, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema/runtime"):
        save_coupled_checkpoint(replace(saved, schema_version=999), path)
    assert path.read_bytes() == before
    strategy.diffusion_coefficient *= 2

    def forbidden(*args, **kwargs):
        pytest.fail("changed physics accessed device")

    monkeypatch.setattr(wp, "get_device", forbidden)
    with pytest.raises(ValueError, match="physics differs"):
        resume_coupled(saved, events, strategy, device=device)


def test_empty_population_can_continue_and_restart(device, tmp_path):
    """Complete evaporation/removal is not a reason to lose event continuation."""
    initial, strategy = case()
    initial.particles.masses[:] = 0
    initial.particles.concentration[:] = 0
    initial.particles.charge[:] = 0
    events = (Dilution(DilutionStrategy(0.2)),)
    with GPUCoupledSimulation(
        initial, events, strategy, device=device, time_step=0.1
    ) as owner:
        owner.step()
        assert owner.statistics.accepted == 0
        saved = owner.checkpoint()
        save_coupled_checkpoint(saved, tmp_path / "empty.npz")
        with resume_coupled(
            load_coupled_checkpoint(tmp_path / "empty.npz"),
            events,
            strategy,
            device=device,
        ) as restored:
            owner.step()
            restored.step()
            equal_native(owner.snapshot(), restored.snapshot())
            canonical_equal(
                owner.checkpoint().events, restored.checkpoint().events
            )
        np.testing.assert_allclose(
            owner.snapshot().gas.concentration,
            initial.gas.concentration * np.exp(-0.04),
            rtol=2e-15,
        )


def test_whole_cycle_capture_rejects_before_event_launch():
    """The captured event optimization cannot imply capturable host control."""
    if not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    initial, strategy = case()
    with GPUCoupledSimulation(
        initial,
        (Dilution(DilutionStrategy(0.2)),),
        strategy,
        device="cuda:0",
        time_step=0.1,
    ) as owner:
        before = owner.checkpoint()
        with wp.ScopedCapture(device="cuda:0"):
            with pytest.raises(RuntimeError, match="during CUDA capture"):
                owner.step()
            with pytest.raises(RuntimeError, match="during CUDA capture"):
                owner.checkpoint()
        canonical_equal(before.events, owner.checkpoint().events)


def test_unsupported_events_reject_before_allocation(monkeypatch):
    """Do not execute fixed condensation in addition to the adaptive flow."""
    from particula.dynamics import MassCondensation

    initial, strategy = case()

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported coupled plan reached device")

    monkeypatch.setattr(wp, "get_device", forbidden)
    with pytest.raises(ValueError, match="event family"):
        GPUCoupledSimulation(
            initial,
            (MassCondensation(strategy),),
            strategy,
            device="cpu",
            time_step=0.1,
        )


def test_direct_checkpoint_waits_for_complete_cycle(monkeypatch):
    """Concurrent direct checkpointing must not inspect intermediate events."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from particula.execution.gpu_coupled_restart import checkpoint_coupled

    initial, strategy = case()
    events = (Dilution(DilutionStrategy(0.2)),)
    with GPUCoupledSimulation(
        initial, events, strategy, device="cpu", time_step=0.001
    ) as owner:
        midcycle, release, attempted = Event(), Event(), Event()
        original_lock = owner._events._lock
        original_condense = owner._condense
        original_ready = owner._ready
        checkpoint_thread = []

        class ObservedLock:
            """Expose an acquisition attempt without changing lock behavior."""

            def __enter__(self):
                if (
                    checkpoint_thread
                    and current_thread() is checkpoint_thread[0]
                ):
                    attempted.set()
                original_lock.acquire()
                return self

            def __exit__(self, *_exc):
                original_lock.release()

        def held_condense(start, end):
            midcycle.set()
            assert release.wait(10), "test failed to release the cycle"
            return original_condense(start, end)

        def checked_ready():
            if checkpoint_thread and current_thread() is checkpoint_thread[0]:
                unsafe = midcycle.is_set() and not release.is_set()
                attempted.set()
                assert not unsafe, "checkpoint observed an incomplete cycle"
            return original_ready()

        def direct_checkpoint():
            checkpoint_thread.append(current_thread())
            return checkpoint_coupled(owner)

        monkeypatch.setattr(owner._events, "_lock", ObservedLock())
        monkeypatch.setattr(owner, "_condense", held_condense)
        monkeypatch.setattr(owner, "_ready", checked_ready)
        with ThreadPoolExecutor(max_workers=2) as pool:
            advance = pool.submit(owner.step)
            assert midcycle.wait(10), "cycle did not start"
            checkpoint = pool.submit(direct_checkpoint)
            try:
                assert attempted.wait(10), "checkpoint did not start"
            finally:
                release.set()
            advance.result(timeout=10)
            saved = checkpoint.result(timeout=10)
        assert saved.events.resident.completed_steps == saved.counters[0] == 1
        assert saved.statistics == owner.statistics


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("origin", [0.0, 2.0])
def test_decimal_bounds_preserve_cycles_and_restart(device, resident, origin):
    """Decimal grid requests retain exact unbounded physics and continuation."""
    if resident and (
        device == "cpu" or not wp.is_conditional_graph_supported()
    ):
        pytest.skip(
            "resident adaptation requires native CUDA conditional graphs"
        )
    initial, strategy = case()
    initial.particles.masses[:] = 0
    initial.particles.concentration[:] = 0
    initial.particles.charge[:] = 0
    events = (Dilution(DilutionStrategy(0.2)),)
    options = dict(device=device, time_step=0.1, time=origin, resident=resident)
    with (
        GPUCoupledSimulation(initial, events, strategy, **options) as bounded,
        GPUCoupledSimulation(initial, events, strategy, **options) as reference,
    ):
        for index in (1, 2):
            bounded.step(origin + index / 10)
            reference.step()
        saved = bounded.checkpoint()
        with resume_coupled(saved, events, strategy, device=device) as resumed:
            for index in range(3, 11):
                bound = origin + index / 10
                before = bounded.checkpoint()
                with pytest.raises(ValueError, match="complete fixed event"):
                    bounded.step(bound - 0.001)
                canonical_equal(before.events, bounded.checkpoint().events)
                bounded.step(bound)
                resumed.step(bound)
                reference.step()
            equal_native(bounded.snapshot(), reference.snapshot())
            equal_native(bounded.snapshot(), resumed.snapshot())
            canonical_equal(
                bounded.checkpoint().events, resumed.checkpoint().events
            )
            assert bounded.time == resumed.time == reference.time
            assert bounded.completed_steps == resumed.completed_steps == 10
            assert bounded.statistics == resumed.statistics
        np.testing.assert_allclose(
            bounded.snapshot().gas.concentration,
            initial.gas.concentration * np.exp(-0.2),
            rtol=3e-14,
        )


def test_decimal_bounds_do_not_accumulate_admission_drift():
    """A thousand fixed decimal cycles are admitted on their integer grid."""
    initial, strategy = case()
    initial.particles.masses[:] = 0
    initial.particles.concentration[:] = 0
    initial.particles.charge[:] = 0
    events = (Dilution(DilutionStrategy(0.0)),)
    with GPUCoupledSimulation(
        initial, events, strategy, device="cpu", time_step=0.1
    ) as owner:
        for index in range(1, 1001):
            owner.step(index / 10)
        assert owner.completed_steps == 1000
        equal_native(initial, owner.snapshot())


@pytest.mark.parametrize(
    "origin, dt, admit", [(0.0, 1e-20, True), (1e15, 1.0, False)]
)
def test_bound_roundoff_is_scaled_and_smaller_than_a_cycle(origin, dt, admit):
    """Small times admit roundoff; coarse clocks cannot waive a partial cycle."""
    initial, strategy = case()
    initial.particles.masses[:] = 0
    initial.particles.concentration[:] = 0
    initial.particles.charge[:] = 0
    events = (Dilution(DilutionStrategy(0.0)),)
    with GPUCoupledSimulation(
        initial, events, strategy, device="cpu", time_step=dt, time=origin
    ) as owner:
        bound = np.nextafter(origin + dt, origin)
        before = owner.checkpoint()
        if admit:
            owner.step(bound)
            assert owner.completed_steps == 1
        else:
            with pytest.raises(ValueError, match="complete fixed event"):
                owner.step(bound)
            canonical_equal(before.events, owner.checkpoint().events)
            assert owner.completed_steps == 0


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("order", ["events-first", "condensation-first"])
def test_decimal_bounds_preserve_active_condensation(device, resident, order):
    """Bound admission does not change physical trials, budgets or event order."""
    if resident and (
        device == "cpu" or not wp.is_conditional_graph_supported()
    ):
        pytest.skip(
            "resident adaptation requires native CUDA conditional graphs"
        )
    initial, strategy = case(2)
    events = (Dilution(DilutionStrategy(0.2)),)
    options = dict(device=device, time_step=0.1, resident=resident, order=order)
    with (
        GPUCoupledSimulation(initial, events, strategy, **options) as bounded,
        GPUCoupledSimulation(initial, events, strategy, **options) as reference,
    ):
        for index in (1, 2, 3):
            bounded.step(index / 10)
            reference.step()
        equal_native(bounded.snapshot(), reference.snapshot())
        canonical_equal(
            bounded.checkpoint().events, reference.checkpoint().events
        )
        assert bounded.statistics == reference.statistics
        assert bounded.checkpoint().counters == reference.checkpoint().counters
        # A later, off-grid bound still includes the next complete interval.
        bounded.step(0.45)
        reference.step()
        equal_native(bounded.snapshot(), reference.snapshot())
