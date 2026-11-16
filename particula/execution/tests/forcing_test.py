"""Environmental knots, one-sided physics, accepted outputs and exact restart."""

from dataclasses import FrozenInstanceError

import numpy as np
import numpy.testing as npt
import pytest
import warp as wp
from scipy.integrate import solve_ivp

from particula.execution.condensation_ode import CondensationODE
from particula.execution.cpu_restart import (
    checkpoint_condensation,
    load_cpu_checkpoint,
    resume_condensation,
    save_cpu_checkpoint,
)
from particula.execution.forcing import EnvironmentForcing
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.gpu_restart import (
    checkpoint_gpu_condensation,
    load_gpu_checkpoint,
    resume_gpu_condensation,
    save_gpu_checkpoint,
)
from particula.execution.recording import Recorder, run_until
from particula.execution.split_restart import checkpoint_split, resume_split
from particula.execution.splitting import SplitSimulation, condensation_flow
from particula.execution.tests.condensation_ode_test import case


@pytest.fixture(params=["rk23", "ros2", "ros3", "warp-cpu", "cuda:0"])
def backend(request):
    """Separate native CPU solvers, Warp CPU and actual CUDA evidence."""
    if request.param == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    return request.param


def owner_for(backend, initial, strategy, table, **controls):
    """Construct the explicitly requested numerical backend."""
    if backend in ("rk23", "ros2", "ros3"):
        return CondensationODE(
            initial, strategy, method=backend, forcing=table, **controls
        )
    return GPUCondensationODE(
        initial,
        strategy,
        device="cpu" if backend == "warp-cpu" else backend,
        forcing=table,
        **controls,
    )


def table_for(mode):
    """Declare abrupt or smooth, independently specified environmental change."""
    return EnvironmentForcing(
        (0, 0.2, 0.5),
        (298.15, 330.0, 285.0),
        (101325.0, 90000.0, 105000.0),
        mode,
    )


def radius_reference(mode):
    """Integrate an independent radius/closed-reservoir equation by segments."""
    temperatures, pressures = (
        [298.15, 330.0, 285.0],
        [101325.0, 90000.0, 105000.0],
    )
    knots = [0, 0.2, 0.5, 1.0]
    answers = []
    for method in ("DOP853", "Radau"):
        radius = 1.0
        values = [radius]
        for index in range(3):

            def rate(time, state, index=index):
                fraction = (
                    (time - knots[index]) / (knots[index + 1] - knots[index])
                    if mode == "linear" and index < 2
                    else 0
                )
                temperature = temperatures[index] + fraction * (
                    temperatures[min(index + 1, 2)] - temperatures[index]
                )
                pressure = pressures[index] + fraction * (
                    pressures[min(index + 1, 2)] - pressures[index]
                )
                viscosity = (
                    1.716e-5
                    * (temperature / 273.15) ** 1.5
                    * 383.55
                    / (temperature + 110.4)
                )
                mean_free_path = (
                    2
                    * viscosity
                    / pressure
                    / np.sqrt(
                        8 * 0.018 / (np.pi * 8.31446261815324 * temperature)
                    )
                )
                x = state[0]
                kn = mean_free_path / (1e-6 * x)
                correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
                vapor = 1e-5 - 1e9 * (4 * np.pi * 1000 * 1e-18 / 3) * (x**3 - 1)
                return [2e-5 * correction * vapor / (1000 * 1e-12 * x)]

            solved = solve_ivp(
                rate,
                knots[index : index + 2],
                [radius],
                method=method,
                rtol=1e-12,
                atol=1e-14,
            )
            assert solved.success
            radius = solved.y[0, -1]
            values.append(radius)
        answers.append(values)
    npt.assert_allclose(answers[0], answers[1], rtol=0, atol=1e-10)
    return np.asarray(answers[0])


@pytest.mark.parametrize("mode", ["step", "linear"])
def test_forcing_hits_knots_and_matches_independent_reference(backend, mode):
    """Trials cannot cross knots; accepted rows use right-hand environment."""
    initial, strategy = case()
    table = table_for(mode)
    owner = owner_for(
        backend,
        initial,
        strategy,
        table,
        rtol=1e-7,
        particle_atol=1e-27,
        gas_atol=1e-17,
        first_step=0.5,
        max_step=2,
    )
    recorder = Recorder(
        ("particle_masses", "gas_concentration", "temperature", "pressure")
    )
    run_until(owner, 1.0, output_times=[0, 0.2, 0.5], recorder=recorder)
    arrays = recorder.arrays()
    npt.assert_array_equal(arrays["time"], [0, 0.2, 0.5, 1.0])
    npt.assert_array_equal(
        arrays["temperature"][:, 0], [298.15, 330.0, 285.0, 285.0]
    )
    reference = radius_reference(mode)
    mass0 = initial.particles.masses[0, 0, 0]
    radii = (arrays["particle_masses"][:, 0, 0, 0] / mass0) ** (1 / 3)
    npt.assert_allclose(radii, reference, rtol=1e-7)
    npt.assert_allclose(
        arrays["gas_concentration"][:, 0, 1],
        1e-5 - 1e9 * mass0 * (reference**3 - 1),
        rtol=1e-7,
    )
    assert owner._forcing is table
    plain = owner_for(
        backend,
        initial,
        strategy,
        table,
        rtol=1e-7,
        particle_atol=1e-27,
        gas_atol=1e-17,
        first_step=0.5,
        max_step=2,
    )
    run_until(plain, 1.0)
    npt.assert_array_equal(
        plain.snapshot().particles.masses, owner.snapshot().particles.masses
    )


def test_table_is_immutable_broadcasts_and_uses_explicit_sides():
    """Caller arrays and returned rows cannot change later forcing or restart."""
    temperatures = np.array([[280.0, 290.0], [300.0, 310.0]])
    table = EnvironmentForcing((0, 1), temperatures, 101325.0, "step")
    temperatures[:] = 1
    npt.assert_array_equal(table(1)[0], [300, 310])
    npt.assert_array_equal(table.segment(0)(1)[0], [280, 290])
    table(0)[0][:] = 999
    npt.assert_array_equal(table(0)[0], [280, 290])
    with pytest.raises(FrozenInstanceError):
        table.mode = "linear"
    for times in ((), (1, 2), (0, 0), (0, np.nan)):
        with pytest.raises(ValueError):
            EnvironmentForcing(times, 298.0, 101325.0)


def test_rejected_trials_restore_declared_forcing(backend):
    """Failure keeps the accepted clock and the full table for future retries."""
    initial, strategy = case()
    table = table_for("step")
    owner = owner_for(
        backend,
        initial,
        strategy,
        table,
        rtol=1e-14,
        particle_atol=1e-35,
        gas_atol=1e-25,
        first_step=100,
        max_step=100,
        max_trials=1,
    )
    with pytest.raises(RuntimeError, match="trial count"):
        owner.step(1)
    assert owner.time == 0
    assert owner._forcing is table
    npt.assert_array_equal(
        owner.snapshot().particles.masses, initial.particles.masses
    )


def test_forcing_disk_restart_at_jump_preserves_future(backend, tmp_path):
    """The table plus accepted absolute time determines the exact next segment."""
    initial, strategy = case()
    owner = owner_for(backend, initial, strategy, table_for("step"), rtol=1e-6)
    recorder = Recorder(("temperature", "gas_concentration"))
    run_until(owner, 0.2, output_times=[0], recorder=recorder)
    path = tmp_path / "forced.pcr"
    if backend in ("rk23", "ros2", "ros3"):
        save_cpu_checkpoint(checkpoint_condensation(owner), path)
        restored = resume_condensation(load_cpu_checkpoint(path), strategy)
    else:
        save_gpu_checkpoint(checkpoint_gpu_condensation(owner), path)
        restored = resume_gpu_condensation(
            load_gpu_checkpoint(path),
            strategy,
            device="cpu" if backend == "warp-cpu" else backend,
        )
    run_until(owner, 1, output_times=[0.2, 0.5])
    run_until(restored, 1, output_times=[0, 0.2, 0.5], recorder=recorder)
    npt.assert_array_equal(
        owner.snapshot().particles.masses, restored.snapshot().particles.masses
    )
    npt.assert_array_equal(
        owner.snapshot().gas.concentration,
        restored.snapshot().gas.concentration,
    )
    npt.assert_array_equal(recorder.arrays()["time"], [0, 0.2, 0.5, 1])
    if backend in ("rk23", "ros2", "ros3"):
        assert owner.solver.next_step == restored.solver.next_step
        assert owner.solver.statistics == restored.solver.statistics
    else:
        assert owner.next_step == restored.next_step
        assert owner.statistics == restored.statistics


def test_split_cycle_lands_on_forcing_events_and_resumes(tmp_path):
    """Outer cycles are bounded by known knots even without output requests."""
    initial, strategy = case()
    flow = condensation_flow(
        "growth", strategy, forcing=table_for("step"), rtol=1e-7
    )
    owner = SplitSimulation(initial, flow, time_step=2)
    assert owner.step(1) == 0.2
    path = tmp_path / "forced-split.pcr"
    save_cpu_checkpoint(checkpoint_split(owner), path)
    restored = resume_split(load_cpu_checkpoint(path), flow)
    assert owner.step(1) == restored.step(1) == 0.5
    run_until(owner, 1)
    run_until(restored, 1)
    npt.assert_array_equal(
        owner.snapshot().particles.masses, restored.snapshot().particles.masses
    )
