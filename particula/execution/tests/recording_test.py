"""Physical output timing, detachment, provenance and bounded recording."""

import numpy as np
import numpy.testing as npt
import pytest

from particula import _cpu_batch
from particula.aerosol import Aerosol
from particula.execution.recording import Recorder, run_until
from particula.gas import EnvironmentData, GasData
from particula.particles import create_resolved_particle_data


def state():
    """Create a native observation with non-unit resolved volume."""
    return Aerosol(
        particles=create_resolved_particle_data(
            masses=np.array([[1e-15]]),
            density=np.array([1000.0]),
            concentration=100.0,
            volume=0.5,
        ),
        gas=GasData(
            name=["vapor"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e-5]]),
            partitioning=np.array([True]),
        ),
        environment=EnvironmentData(
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.zeros((1, 1)),
        ),
    )


class Owner:
    """Minimal accepted-step owner with an exact gas-decay trajectory."""

    def __init__(self):
        """Retain initial native state and a zero clock."""
        self.time = 0.0
        self.state = state()
        self.calls = 0

    def step(self, bound):
        """Accept a bounded exact step."""
        dt = min(0.1, bound - self.time)
        self.state.gas.concentration *= np.exp(-dt)
        self.time = bound if dt == bound - self.time else self.time + dt
        self.calls += 1

    def snapshot(self):
        """Return detached state without changing accepted values."""
        return _cpu_batch.detached(self.state)


def test_output_times_final_and_resume_without_duplicate_rows():
    """Requested physical times are exact and a resumed join is observed once."""
    owner, recorder = Owner(), Recorder()
    run_until(owner, 0.25, output_times=[0, 0.15], recorder=recorder)
    run_until(owner, 0.4, output_times=[0, 0.15, 0.25, 0.3], recorder=recorder)
    arrays = recorder.arrays()
    npt.assert_array_equal(arrays["time"], [0, 0.15, 0.25, 0.3, 0.4])
    npt.assert_allclose(
        arrays["gas_concentration"][:, 0, 0],
        1e-5 * np.exp(-arrays["time"]),
        rtol=1e-14,
    )
    npt.assert_array_equal(arrays["volume"], np.full((5, 1), 0.5))


def test_passive_observation_does_not_change_fixed_schedule():
    """Equal step/output schedules retain exactly the same accepted trajectory."""
    plain, observed = Owner(), Owner()
    run_until(plain, 0.3, output_times=[0.1, 0.2])
    run_until(observed, 0.3, output_times=[0.1, 0.2], recorder=Recorder())
    npt.assert_array_equal(
        plain.state.gas.concentration, observed.state.gas.concentration
    )
    assert plain.calls == observed.calls


def test_detached_arrays_and_native_population_provenance():
    """Returned arrays, native input and metadata do not alias retained rows."""
    native, recorder = state(), Recorder(("particle_concentration",))
    recorder.record(0, native)
    native.particles.concentration[:] = 0
    arrays = recorder.arrays()
    arrays["particle_concentration"][:] = 999
    assert recorder.arrays()["particle_concentration"][0, 0, 0] == 100
    metadata = recorder.metadata
    assert metadata["observables"]["particle_concentration"]["unit"] == "count"
    metadata["gas_names"][0] = "wrong"
    assert recorder.metadata["gas_names"] == ["vapor"]


def test_limit_and_species_change_reject_without_partial_append():
    """A failed observation leaves all previous arrays and metadata usable."""
    native, recorder = state(), Recorder(max_records=1)
    recorder.record(0, native)
    with pytest.raises(ValueError, match="storage limit"):
        recorder.record(1, native)
    assert recorder.last_time == 0
    other = Recorder()
    other.record(0, native)
    native.gas.name[0] = "different"
    with pytest.raises(ValueError, match="provenance"):
        other.record(1, native)
    assert other.last_time == 0


def test_invalid_schedule_is_rejected_before_advance():
    """No accepted work occurs for malformed or out-of-range output times."""
    owner = Owner()
    for times in ([0.2, 0.1], [0.4], [np.nan], [-1]):
        with pytest.raises(ValueError):
            run_until(owner, 0.3, output_times=times)
    assert owner.time == owner.calls == 0


def test_adaptive_rejections_never_emit_trial_observations():
    """Only requested accepted times are visible despite rejected trial work."""
    from particula.execution.condensation_ode import CondensationODE
    from particula.execution.tests.condensation_ode_test import case

    initial, strategy = case()
    options = dict(rtol=1e-8, first_step=100, max_step=100)
    plain = CondensationODE(initial, strategy, **options)
    observed = CondensationODE(initial, strategy, **options)
    recorder = Recorder()
    times = [0.0, 0.1, 0.3]
    run_until(plain, 0.3, output_times=times)
    run_until(observed, 0.3, output_times=times, recorder=recorder)
    assert observed.solver.statistics.rejected > 0
    npt.assert_array_equal(recorder.arrays()["time"], times)
    npt.assert_array_equal(observed.solver.state, plain.solver.state)
    assert observed.solver.statistics == plain.solver.statistics


def test_exhausted_adaptive_trial_budget_preserves_last_observation():
    """A failed first trial publishes neither state nor an output row."""
    from particula.execution.condensation_ode import CondensationODE
    from particula.execution.tests.condensation_ode_test import case

    initial, strategy = case()
    owner = CondensationODE(
        initial, strategy, first_step=100, max_step=100, max_trials=1
    )
    recorder = Recorder()
    with pytest.raises(RuntimeError):
        run_until(owner, 100, output_times=[0], recorder=recorder)
    assert owner.time == recorder.last_time == 0
    npt.assert_array_equal(recorder.arrays()["time"], [0])
    npt.assert_array_equal(
        owner.snapshot().gas.concentration, initial.gas.concentration
    )


def test_shortened_output_steps_meet_independent_radius_reference():
    """Two output schedules meet the same independent growth accuracy target."""
    from particula.execution.condensation_ode import CondensationODE
    from particula.execution.tests.condensation_ode_test import (
        case,
        reference_duration,
    )

    initial, strategy = case()
    duration = reference_duration()
    radii = []
    for outputs in ((), (0, duration / 3, 2 * duration / 3)):
        owner = CondensationODE(
            initial,
            strategy,
            rtol=1e-5,
            particle_atol=1e-23,
            gas_atol=1e-13,
            dynamic_viscosity=1.8e-5,
        )
        run_until(owner, duration, output_times=outputs, recorder=Recorder())
        radii.append(owner.snapshot().particles.radii[0, 0] / 1e-6)
    npt.assert_allclose(radii, [1.2, 1.2], rtol=2e-5)


def test_split_observations_follow_complete_species_budget():
    """Strang half steps never appear as rows; complete cycles close the budget."""
    from particula.execution.splitting import (
        SplitGroup,
        SplitSimulation,
        condensation_flow,
        dilution_flow,
    )
    from particula.execution.tests.condensation_ode_test import case

    initial, strategy = case()
    group = SplitGroup(
        (
            dilution_flow("removal", 0.2),
            condensation_flow("growth", strategy, rtol=1e-7),
        )
    )
    owner = SplitSimulation(initial, group, time_step=0.05)
    recorder = Recorder()
    times = np.array([0, 0.05, 0.15, 0.2])
    run_until(owner, 0.2, output_times=times, recorder=recorder)
    arrays = recorder.arrays()
    npt.assert_array_equal(arrays["time"], times)
    total = (
        arrays["particle_mass_density"][:, 0, 0]
        + arrays["gas_concentration"][:, 0, 1]
    )
    npt.assert_allclose(total, total[0] * np.exp(-0.2 * times), rtol=3e-13)
    npt.assert_allclose(
        arrays["number_density"][:, 0], 1e9 * np.exp(-0.2 * times), rtol=3e-13
    )
