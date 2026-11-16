"""Native NumPy and adaptive disk continuation, independent of Warp execution."""

import numpy as np
import numpy.testing as npt
import pytest

from particula import _cpu_batch
from particula.execution.condensation_ode import CondensationODE
from particula.execution.cpu_restart import (
    checkpoint_condensation,
    load_cpu_checkpoint,
    resume_condensation,
    save_cpu_checkpoint,
)
from particula.execution.cpu_simulation import CPUSimulation
from particula.execution.recording import Recorder, run_until
from particula.execution.tests.condensation_ode_test import case
from particula.execution.tests.prepared_restart_capture_test import scenario


def equal_state(left, right):
    """Compare all stored native arrays, including auxiliary fields and ledgers."""
    a, b = _cpu_batch.arrays(left), _cpu_batch.arrays(right)
    assert set(a) == set(b)
    for key in a:
        npt.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize(
    "name", ["dilution", "coagulation", "wall_loss", "sequence", "condensation"]
)
def test_native_cpu_disk_continuation_is_exact(tmp_path, name, monkeypatch):
    """Native CPU continuation retains random streams without any Warp launch."""
    import warp as wp

    def forbidden(*args, **kwargs):
        raise AssertionError("native CPU must not launch Warp kernels")

    monkeypatch.setattr(wp, "launch", forbidden)
    initial, process, dt = scenario(name)
    original = CPUSimulation(initial, process, time_step=dt, root_seed=171)
    run_until(original, 3 * dt)
    saved = original.checkpoint()
    path = tmp_path / "native.pcr"
    save_cpu_checkpoint(saved, path)
    loaded = load_cpu_checkpoint(path)
    restored = CPUSimulation.resume(loaded, process)
    assert restored.checkpoint().rng == saved.rng
    equal_state(original.snapshot(), restored.snapshot())
    # Identical subsequent schedules, including a shortened final cycle.
    run_until(original, 5.5 * dt)
    run_until(restored, 5.5 * dt)
    equal_state(original.snapshot(), restored.snapshot())
    assert original.checkpoint().rng == restored.checkpoint().rng
    assert original.time == restored.time


@pytest.mark.parametrize("method", ["rk23", "ros2"])
def test_adaptive_restart_preserves_controller_and_future_bitwise(
    tmp_path, method
):
    """The saved proposed step/error history resumes the exact same accepted grid."""
    initial, strategy = case()
    original = CondensationODE(
        initial, strategy, method=method, rtol=1e-5, dynamic_viscosity=1.8e-5
    )
    run_until(original, 0.3)
    saved = checkpoint_condensation(original)
    path = tmp_path / "adaptive.pcr"
    save_cpu_checkpoint(saved, path)
    restored = resume_condensation(load_cpu_checkpoint(path), strategy)
    assert restored.solver.next_step == original.solver.next_step
    assert restored.solver.statistics == original.solver.statistics
    while original.time < 0.7:
        original.step(0.7)
        restored.step(0.7)
        assert original.time == restored.time
        npt.assert_array_equal(original.solver.state, restored.solver.state)
    equal_state(original.snapshot(), restored.snapshot())


def test_native_failed_cycle_rolls_back_rng_and_all_state(monkeypatch):
    """Later failure cannot expose earlier stochastic updates in the cycle."""
    initial, process, dt = scenario("sequence")
    owner = CPUSimulation(initial, process, time_step=dt, root_seed=17)
    saved = owner.checkpoint()

    # Inject failure into the last private component after coagulation ran.
    def fail(*args, **kwargs):
        raise ValueError("later scientific failure")

    monkeypatch.setattr(owner._processes[-1], "execute", fail)
    with pytest.raises(ValueError, match="later scientific failure"):
        owner.step(dt)
    equal_state(initial, owner.snapshot())
    assert owner.checkpoint().rng == saved.rng
    assert owner.time == owner.completed_steps == 0


def test_native_recording_does_not_consume_random_streams():
    """Passive observation at identical boundaries leaves exact RNG continuation."""
    initial, process, dt = scenario("coagulation")
    plain = CPUSimulation(initial, process, time_step=dt, root_seed=17)
    recorded = CPUSimulation(initial, process, time_step=dt, root_seed=17)
    outputs = np.arange(5) * dt
    run_until(plain, 4 * dt, output_times=outputs)
    run_until(recorded, 4 * dt, output_times=outputs, recorder=Recorder())
    equal_state(plain.snapshot(), recorded.snapshot())
    assert plain.checkpoint().rng == recorded.checkpoint().rng


def test_changed_adaptive_physics_and_callable_forcing_reject():
    """Restart identity cannot be inferred from arbitrary callable forcing."""
    initial, strategy = case()
    owner = CondensationODE(initial, strategy)
    saved = checkpoint_condensation(owner)
    strategy.diffusion_coefficient = 1e-5
    with pytest.raises(ValueError, match="physics differs"):
        resume_condensation(saved, strategy)
    forced = CondensationODE(
        initial, strategy, forcing=lambda t: (298.15, 101325)
    )
    with pytest.raises(ValueError, match="forcing"):
        checkpoint_condensation(forced)


def test_auxiliary_fields_and_inventory_ledgers_survive_disk(tmp_path):
    """Restart retains additive tracer values, removals and field descriptors."""
    from particula import FieldSpec

    initial, process, dt = scenario("dilution")
    values = np.where(initial.particles.concentration > 0, 2e-20, 0.0)
    initial.fields.add(
        FieldSpec(
            "lab.tracer", ("box", "particle"), "kg", particle_policy="additive"
        ),
        values,
    )
    owner = CPUSimulation(initial, process, time_step=dt)
    run_until(owner, 3 * dt)
    path = tmp_path / "fields.pcr"
    save_cpu_checkpoint(owner.checkpoint(), path)
    saved = load_cpu_checkpoint(path)
    restored = CPUSimulation.resume(saved, process)
    equal_state(owner.snapshot(), restored.snapshot())
    before = owner.snapshot().fields.inventory_changes("lab.tracer")
    assert np.all(before["removal"] > 0)
    run_until(owner, 6 * dt)
    run_until(restored, 6 * dt)
    equal_state(owner.snapshot(), restored.snapshot())
    # Neither disk inspection nor snapshots alias the live continuation.
    saved.state[3][0][1][:] = 0
    assert np.any(restored.snapshot().fields["lab.tracer"] > 0)


def test_latent_energy_diagnostics_survive_restart(tmp_path):
    """Last-step energy is retained separately from physical configuration."""
    initial, process, dt = scenario("condensation")
    owner = CPUSimulation(initial, process, time_step=dt)
    owner.step(dt)
    saved = owner.checkpoint()
    path = tmp_path / "latent.pcr"
    save_cpu_checkpoint(saved, path)
    restored = CPUSimulation.resume(load_cpu_checkpoint(path), process)
    for left, right in zip(
        saved.auxiliary, restored.checkpoint().auxiliary, strict=True
    ):
        npt.assert_array_equal(left[0], right[0])
        assert left[1] == right[1]
    npt.assert_array_equal(
        restored._processes[0].last_latent_heat_energy,
        owner._processes[0].last_latent_heat_energy,
    )


def test_native_staggered_rng_continuation_and_cpu_controls(tmp_path):
    """CPU staggered random stages resume without GPU planner restrictions."""
    from particula.dynamics import MassCondensation
    from particula.dynamics.condensation.condensation_strategies import (
        CondensationIsothermalStaggered,
    )

    initial, strategy = case()
    staggered = CondensationIsothermalStaggered(
        **vars(strategy), theta_mode="random", shuffle_each_step=True
    )
    process = MassCondensation(staggered)
    owner = CPUSimulation(initial, process, time_step=0.001, root_seed=172)
    run_until(owner, 0.003)
    path = tmp_path / "staggered.pcr"
    save_cpu_checkpoint(owner.checkpoint(), path)
    saved = load_cpu_checkpoint(path)
    restored = CPUSimulation.resume(saved, process)
    run_until(owner, 0.007)
    run_until(restored, 0.007)
    equal_state(owner.snapshot(), restored.snapshot())
    assert owner.checkpoint().rng == restored.checkpoint().rng
    assert b"condensation" in owner.checkpoint().rng
    staggered.num_batches += 1
    with pytest.raises(ValueError, match="configuration"):
        CPUSimulation.resume(saved, process)


def test_cpu_sampling_setting_change_rejects():
    """A native sampler option absent from the GPU plan still identifies CPU work."""
    initial, process, dt = scenario("coagulation")
    saved = CPUSimulation(initial, process, time_step=dt).checkpoint()
    process.coagulation_strategy.particle_resolved_bins_per_decade += 1
    with pytest.raises(ValueError, match="configuration"):
        CPUSimulation.resume(saved, process)


@pytest.mark.parametrize(
    "field,value", [("time", -1), ("completed_steps", True)]
)
def test_corrupt_native_controller_rejects(field, value):
    """Invalid clock/counter metadata fails before any continuation."""
    from dataclasses import replace

    initial, process, dt = scenario("dilution")
    saved = CPUSimulation(initial, process, time_step=dt).checkpoint()
    with pytest.raises(ValueError):
        CPUSimulation.resume(replace(saved, **{field: value}), process)


def test_native_nucleation_disk_continuation_before_integral_pulse(tmp_path):
    """A saved source configuration produces the same qualified integral pulse."""
    initial, process, dt = scenario("nucleation")
    original = CPUSimulation(initial, process, time_step=dt, root_seed=13)
    path = tmp_path / "source.pcr"
    save_cpu_checkpoint(original.checkpoint(), path)
    resumed = CPUSimulation.resume(load_cpu_checkpoint(path), process)
    original.step(dt)
    resumed.step(dt)
    equal_state(original.snapshot(), resumed.snapshot())
    assert original.checkpoint().rng == resumed.checkpoint().rng
