"""Contract tests for the supported CPU nucleation runnable."""

# ruff: noqa: D100, D101, D102, D103

import numpy as np
import numpy.testing as npt
import particula.dynamics
import particula.dynamics.particle_process as particle_process
import pytest
from particula.aerosol import Aerosol
from particula.dynamics.nucleation import (
    ActivationNucleationStrategy,
    ClosedInterval,
    FormationMetadata,
    InjectionComposition,
    NucleationSourceConfig,
    NucleationValidityDomain,
)
from particula.dynamics.particle_process import (
    Nucleation,
    NucleationCommitConfig,
)
from particula.gas.atmosphere import Atmosphere
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import MassBasedMovingBin
from particula.particles.exhaustion import ExhaustionControls
from particula.particles.particle_data import ParticleData
from particula.particles.representation import ParticleRepresentation
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.runnable import RunnableABC
from particula.util.constants import AVOGADRO_NUMBER


def _source_config(coefficient: float = 1.0e-2) -> NucleationSourceConfig:
    """Build a source configuration spanning the deterministic fixture."""
    return NucleationSourceConfig(
        strategy=ActivationNucleationStrategy(
            coefficient=coefficient,
            validity_domain=NucleationValidityDomain(
                ClosedInterval(1.0, 1.0e30),
                ClosedInterval(200.0, 400.0),
            ),
            injection_composition=InjectionComposition((1,)),
            formation_metadata=FormationMetadata(1.0e-9),
        ),
        precursor_index=0,
    )


def _commit_config() -> NucleationCommitConfig:
    """Build valid one-box commit controls."""
    return NucleationCommitConfig(
        maximum_slot_weight=1.0e20,
        source_charge=0.0,
        exhaustion_controls=ExhaustionControls(),
        requested_scale=np.array([1.0], dtype=np.float64),
        minimum_scale=np.array([1.0], dtype=np.float64),
        minimum_volume=np.array([1.0], dtype=np.float64),
    )


def _aerosol() -> tuple[Aerosol, ParticleData, GasData, GasData]:
    """Build facades over deterministic writable backing containers."""
    particles = ParticleData(
        masses=np.zeros((1, 3, 1), dtype=np.float64),
        concentration=np.zeros((1, 3), dtype=np.float64),
        charge=np.zeros((1, 3), dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([1.0], dtype=np.float64),
    )
    gas = GasData(
        name=["precursor"],
        molar_mass=np.array([0.1], dtype=np.float64),
        concentration=np.array([[1.0e-12]], dtype=np.float64),
        partitioning=np.array([True]),
    )
    gas_only = GasData(
        name=["inert"],
        molar_mass=np.array([0.04], dtype=np.float64),
        concentration=np.array([[2.0e-6]], dtype=np.float64),
        partitioning=np.array([False]),
    )
    particle_facade = ParticleRepresentation.from_data(
        particles,
        strategy=MassBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.zeros(3, dtype=np.float64),
    )
    atmosphere = Atmosphere(
        temperature=298.15,
        total_pressure=101325.0,
        partitioning_species=GasSpecies.from_data(
            gas,
            ConstantVaporPressureStrategy(0.0),
        ),
        gas_only_species=GasSpecies.from_data(
            gas_only,
            ConstantVaporPressureStrategy(0.0),
        ),
    )
    return Aerosol(atmosphere, particle_facade), particles, gas, gas_only


def _runnable(coefficient: float = 1.0e-2) -> Nucleation:
    """Build a runnable with a single-box environment."""
    return Nucleation(
        _source_config(coefficient),
        _commit_config(),
        EnvironmentData(
            temperature=np.array([298.15], dtype=np.float64),
            pressure=np.array([101325.0], dtype=np.float64),
            saturation_ratio=np.array([[1.0]], dtype=np.float64),
        ),
    )


def test_public_surface_exports_only_supported_p5_types():
    """Dynamics exports P5 types without exposing P2/P3 transaction records."""
    assert particula.dynamics.Nucleation is Nucleation
    assert particula.dynamics.NucleationCommitConfig is NucleationCommitConfig
    assert not hasattr(particula.dynamics, "ParticleSourceCommitConfig")
    assert not hasattr(particula.dynamics.nucleation, "commit_particle_source")
    assert not hasattr(particle_process, "ParticleSourceCommitConfig")
    assert not hasattr(particle_process, "PotentialEventData")
    assert not hasattr(particle_process, "commit_particle_source")
    assert not hasattr(particle_process, "finalize_particle_source")


def test_commit_config_owns_immutable_float64_sidecars():
    """Public controls copy caller sidecars and preserve P3 scalar rules."""
    requested = np.array([1.0], dtype=np.float32)
    config = NucleationCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=np.float64(0.0),
        exhaustion_controls=ExhaustionControls(),
        requested_scale=requested,
        minimum_scale=np.array([1.0]),
        minimum_volume=np.array([1.0]),
    )

    assert config.requested_scale.dtype == np.float64
    assert not config.requested_scale.flags.writeable
    assert not np.shares_memory(config.requested_scale, requested)
    with pytest.raises(ValueError):
        config.requested_scale[0] = 2.0
    with pytest.raises(TypeError, match="exact Python float"):
        NucleationCommitConfig(
            1.0, 0.0, ExhaustionControls(), [1.0], [1.0], [1.0], np.float64(1)
        )


def test_rate_uses_backing_gas_and_omits_saturation_outside_domain():
    """The P1 adapter supplies gas lane zero and None saturation as required."""
    aerosol, _, gas, _ = _aerosol()
    runnable = _runnable()
    expected = runnable.source_config.strategy.potential_rate(
        float(gas.concentration[0, 0]),
        float(gas.molar_mass[0]),
        298.15,
        None,
    )

    assert runnable.rate(aerosol) == expected


def test_rate_uses_environment_saturation_for_saturation_domain():
    """The P1 adapter supplies the selected environment saturation lane."""
    aerosol, _, gas, _ = _aerosol()
    strategy = ActivationNucleationStrategy(
        coefficient=1.0e-2,
        validity_domain=NucleationValidityDomain(
            ClosedInterval(1.0, 1.0e30),
            ClosedInterval(200.0, 400.0),
            ClosedInterval(1.4, 1.6),
        ),
        injection_composition=InjectionComposition((1,)),
        formation_metadata=FormationMetadata(1.0e-9),
    )
    runnable = Nucleation(
        NucleationSourceConfig(strategy=strategy, precursor_index=0),
        _commit_config(),
        EnvironmentData(
            temperature=np.array([298.15], dtype=np.float64),
            pressure=np.array([101325.0], dtype=np.float64),
            saturation_ratio=np.array([[1.5]], dtype=np.float64),
        ),
    )
    expected = strategy.potential_rate(
        float(gas.concentration[0, 0]),
        float(gas.molar_mass[0]),
        298.15,
        1.5,
    )

    assert runnable.rate(aerosol) == expected


@pytest.mark.parametrize(
    ("source_config", "commit_config", "environment", "message"),
    [
        (None, _commit_config, EnvironmentData, "source_config"),
        (_source_config, None, EnvironmentData, "commit_config"),
        (_source_config, _commit_config, None, "environment"),
    ],
)
def test_constructor_rejects_wrong_top_level_configuration_types(
    source_config, commit_config, environment, message
):
    """Construction reports the invalid public configuration boundary."""
    source = source_config() if callable(source_config) else source_config
    commit = commit_config() if callable(commit_config) else commit_config
    environment = (
        environment(
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.array([[1.0]]),
        )
        if callable(environment)
        else environment
    )
    with pytest.raises(TypeError, match=message):
        Nucleation(source, commit, environment)


def test_rate_rejects_invalid_topology_before_p1_and_preserves_state():
    """Topology mismatch raises before P1 and leaves facade storage untouched."""
    aerosol, particles, gas, gas_only = _aerosol()
    particles.masses = np.zeros((1, 3, 2), dtype=np.float64)
    gas_before = gas.concentration.copy()
    gas_only_before = gas_only.concentration.copy()

    with pytest.raises(ValueError, match="species widths"):
        _runnable().rate(aerosol)
    npt.assert_array_equal(gas.concentration, gas_before)
    npt.assert_array_equal(gas_only.concentration, gas_only_before)


def test_execute_commits_each_positive_substep_and_preserves_identities(
    monkeypatch: pytest.MonkeyPatch,
):
    """Positive substeps finalize and commit with the original backing objects."""
    aerosol, particles, gas, gas_only = _aerosol()
    runnable = _runnable()
    calls: list[tuple[float, ParticleData, GasData]] = []

    def finalize(events, composition, source_gas):
        assert source_gas is gas
        return object(), object()

    def commit(demand, diagnostics, source_particles, source_gas, config):
        calls.append((config.maximum_slot_weight, source_particles, source_gas))

    monkeypatch.setattr(particle_process, "_finalize_particle_source", finalize)
    monkeypatch.setattr(particle_process, "_commit_particle_source", commit)
    assert runnable.execute(aerosol, 2.0, sub_steps=2) is aerosol
    assert len(calls) == 2
    assert all(entry[1] is particles and entry[2] is gas for entry in calls)
    assert aerosol.particles.data is particles
    assert aerosol.atmosphere.partitioning_species.data is gas
    npt.assert_array_equal(gas_only.concentration, [[2.0e-6]])


def test_real_execution_depletes_gas_and_never_touches_gas_only():
    """P3 transfer changes only the supported particle and partitioning state."""
    aerosol, particles, gas, gas_only = _aerosol()
    initial_gas = gas.concentration.copy()
    initial_gas_only = gas_only.concentration.copy()

    assert _runnable().execute(aerosol, 1.0, sub_steps=2) is aerosol
    assert gas.concentration[0, 0] < initial_gas[0, 0]
    assert np.any(particles.concentration > 0.0)
    npt.assert_array_equal(gas_only.concentration, initial_gas_only)


@pytest.mark.parametrize("sub_steps", [0, -1, True, 1.0, "1", None])
def test_invalid_substeps_fail_before_facade_access(sub_steps):
    """Invalid substep counts reject before inspecting the aerosol facade."""
    with pytest.raises(
        ValueError, match="sub_steps must be a positive integer"
    ):
        _runnable().execute(object(), 1.0, sub_steps=sub_steps)


def test_excessive_substeps_fail_before_facade_access_or_source_work():
    """The bounded substep count rejects before topology or P1 evaluation."""
    with pytest.raises(ValueError, match="must not exceed"):
        _runnable().execute(
            object(),
            1.0,
            sub_steps=particle_process._MAX_NUCLEATION_SUB_STEPS + 1,
        )


def test_maximum_substeps_are_accepted(monkeypatch: pytest.MonkeyPatch):
    """The documented maximum executes valid zero-rate substeps."""
    aerosol, _, _, _ = _aerosol()
    runnable = _runnable()
    calls = 0

    def zero_rate(_: GasData) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    monkeypatch.setattr(runnable, "_rate_from_gas", zero_rate)

    assert (
        runnable.execute(
            aerosol, 1.0, particle_process._MAX_NUCLEATION_SUB_STEPS
        )
        is aerosol
    )
    assert calls == particle_process._MAX_NUCLEATION_SUB_STEPS


def test_malformed_facade_fails_before_source_work(monkeypatch):
    """Missing legacy facades reject before any P1 source evaluation."""
    runnable = _runnable()
    monkeypatch.setattr(
        runnable,
        "_rate_from_gas",
        lambda _: pytest.fail("P1 must not run for malformed facades"),
    )

    with pytest.raises(TypeError, match="facades"):
        runnable.execute(object(), 1.0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda particles, gas, runnable: setattr(
            particles, "masses", np.zeros((2, 3, 1), dtype=np.float64)
        ),
        lambda particles, gas, runnable: setattr(
            gas, "concentration", np.zeros((2, 1), dtype=np.float64)
        ),
        lambda particles, gas, runnable: setattr(
            gas, "molar_mass", np.ones((1, 1), dtype=np.float64)
        ),
        lambda particles, gas, runnable: setattr(
            particles, "masses", np.zeros((1, 3, 2), dtype=np.float64)
        ),
        lambda particles, gas, runnable: setattr(
            runnable.environment,
            "temperature",
            np.array([298.15, 298.15], dtype=np.float64),
        ),
        lambda particles, gas, runnable: setattr(
            runnable.environment,
            "saturation_ratio",
            np.ones((1, 2), dtype=np.float64),
        ),
        lambda particles, gas, runnable: setattr(
            runnable,
            "source_config",
            NucleationSourceConfig(
                strategy=runnable.source_config.strategy,
                precursor_index=1,
            ),
        ),
    ],
)
def test_invalid_topology_is_write_free_and_skips_source_work(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
):
    """Unsupported facade topology is rejected before P1, P2, or P3 work."""
    aerosol, particles, gas, gas_only = _aerosol()
    runnable = _runnable()
    mutate(particles, gas, runnable)
    arrays = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.concentration,
        gas.partitioning,
        gas_only.molar_mass,
        gas_only.concentration,
        gas_only.partitioning,
    )
    snapshots = tuple(
        (
            array.copy(),
            id(array),
            array.shape,
            array.dtype,
            array.flags.c_contiguous,
            array.flags.writeable,
        )
        for array in arrays
    )
    facade_identities = (
        id(aerosol),
        id(aerosol.particles),
        id(aerosol.particles.data),
        id(aerosol.atmosphere.partitioning_species),
        id(aerosol.atmosphere.partitioning_species.data),
    )
    monkeypatch.setattr(
        runnable,
        "_rate_from_gas",
        lambda _: pytest.fail("P1 must not run for invalid topology"),
    )
    monkeypatch.setattr(
        particle_process,
        "_finalize_particle_source",
        lambda *args: pytest.fail("P2 must not run for invalid topology"),
    )
    monkeypatch.setattr(
        particle_process,
        "_commit_particle_source",
        lambda *args: pytest.fail("P3 must not run for invalid topology"),
    )

    with pytest.raises((TypeError, ValueError)):
        runnable.execute(aerosol, 1.0)
    for array, snapshot in zip(arrays, snapshots, strict=True):
        values, identity, shape, dtype, contiguous, writable = snapshot
        npt.assert_array_equal(array, values)
        assert (
            id(array),
            array.shape,
            array.dtype,
            array.flags.c_contiguous,
            array.flags.writeable,
        ) == (identity, shape, dtype, contiguous, writable)
    assert facade_identities == (
        id(aerosol),
        id(aerosol.particles),
        id(aerosol.particles.data),
        id(aerosol.atmosphere.partitioning_species),
        id(aerosol.atmosphere.partitioning_species.data),
    )


def test_zero_duration_validates_topology_without_source_work(monkeypatch):
    """A valid zero duration is a structural, exact write-free no-op."""
    aerosol, particles, gas, gas_only = _aerosol()
    particle_before = particles.masses.copy()
    gas_before = gas.concentration.copy()
    gas_only_before = gas_only.concentration.copy()
    monkeypatch.setattr(
        particle_process,
        "_finalize_particle_source",
        lambda *args: pytest.fail("P2 must not run for zero duration"),
    )

    assert _runnable().execute(aerosol, 0.0) is aerosol
    npt.assert_array_equal(particles.masses, particle_before)
    npt.assert_array_equal(gas.concentration, gas_before)
    npt.assert_array_equal(gas_only.concentration, gas_only_before)
    assert isinstance(_runnable(), RunnableABC)


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("maximum_slot_weight", 0.0, ValueError),
        ("maximum_slot_weight", True, TypeError),
        ("source_charge", np.inf, ValueError),
        ("exhaustion_controls", None, TypeError),
        ("requested_scale", [[1.0]], ValueError),
        ("minimum_scale", ["bad"], TypeError),
        ("minimum_volume", [True], TypeError),
        ("radius_cubed_relative_error", -1.0, ValueError),
    ],
)
def test_commit_config_rejects_invalid_controls(field, value, exception):
    """Commit controls enforce P3-compatible scalar and sidecar validation."""
    controls = {
        "maximum_slot_weight": 1.0,
        "source_charge": 0.0,
        "exhaustion_controls": ExhaustionControls(),
        "requested_scale": [1.0],
        "minimum_scale": [1.0],
        "minimum_volume": [1.0],
        "radius_cubed_relative_error": 1.0,
    }
    controls[field] = value

    with pytest.raises(exception):
        NucleationCommitConfig(**controls)


@pytest.mark.parametrize(
    ("duration", "exception"),
    [
        (None, TypeError),
        (True, TypeError),
        ("one", TypeError),
        (-1.0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (np.array([1.0]), ValueError),
    ],
)
def test_invalid_duration_fails_before_topology_or_source_work(
    duration, exception
):
    """Invalid durations reject before facade lookup and source transactions."""
    with pytest.raises(exception):
        _runnable().execute(object(), duration)


def test_execute_skips_transactions_for_zero_rate(monkeypatch):
    """An exact zero P1 rate is a write-free skip for every substep."""
    aerosol, particles, gas, gas_only = _aerosol()
    snapshots = [
        particles.masses.copy(),
        particles.concentration.copy(),
        gas.concentration.copy(),
        gas_only.concentration.copy(),
    ]
    runnable = _runnable()
    monkeypatch.setattr(runnable, "_rate_from_gas", lambda _: 0.0)
    monkeypatch.setattr(
        particle_process,
        "_finalize_particle_source",
        lambda *args: pytest.fail("P2 must not run for a zero rate"),
    )
    monkeypatch.setattr(
        particle_process,
        "_commit_particle_source",
        lambda *args: pytest.fail("P3 must not run for a zero rate"),
    )

    assert runnable.execute(aerosol, 1.0, sub_steps=2) is aerosol
    for actual, expected in zip(
        (
            particles.masses,
            particles.concentration,
            gas.concentration,
            gas_only.concentration,
        ),
        snapshots,
        strict=True,
    ):
        npt.assert_array_equal(actual, expected)


def test_execute_uses_equal_duration_and_fresh_config_per_positive_substep(
    monkeypatch: pytest.MonkeyPatch,
):
    """Positive rates build one P2 record and private P3 config per substep."""
    aerosol, _, gas, _ = _aerosol()
    runnable = _runnable()
    rates = iter((2.0, 3.0))
    durations: list[float] = []
    configs = []
    monkeypatch.setattr(runnable, "_rate_from_gas", lambda _: next(rates))

    def finalize(events, composition, source_gas):
        assert source_gas is gas
        assert (
            composition is runnable.source_config.strategy.injection_composition
        )
        durations.append(events.duration)
        return object(), object()

    def commit(demand, diagnostics, particles, source_gas, config):
        assert source_gas is gas
        configs.append(config)

    monkeypatch.setattr(particle_process, "_finalize_particle_source", finalize)
    monkeypatch.setattr(particle_process, "_commit_particle_source", commit)

    assert runnable.execute(aerosol, 4.0, sub_steps=2) is aerosol
    assert durations == [2.0, 2.0]
    assert len(configs) == 2
    assert configs[0] is not configs[1]


def test_later_substep_rate_uses_gas_depleted_by_prior_commit(
    monkeypatch: pytest.MonkeyPatch,
):
    """Each P1 rate observes the live precursor gas after the preceding P3."""
    aerosol, _, gas, _ = _aerosol()
    runnable = _runnable()
    observed_rates: list[float] = []
    rate_from_gas = runnable._rate_from_gas

    def record_rate(source_gas: GasData) -> float:
        rate = rate_from_gas(source_gas)
        observed_rates.append(rate)
        return rate

    def commit(*args) -> None:
        if len(observed_rates) == 1:
            gas.concentration[0, 0] *= 0.5

    monkeypatch.setattr(runnable, "_rate_from_gas", record_rate)
    monkeypatch.setattr(
        particle_process,
        "_finalize_particle_source",
        lambda *args: (object(), object()),
    )
    monkeypatch.setattr(particle_process, "_commit_particle_source", commit)

    assert runnable.execute(aerosol, 2.0, sub_steps=2) is aerosol
    assert len(observed_rates) == 2
    assert observed_rates[1] == observed_rates[0] * 0.5
    npt.assert_array_equal(gas.concentration, [[0.5e-12]])


def test_real_substeps_read_current_gas_and_match_independent_transfer_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-box P5 path recomputes deterministic source demand per step."""
    aerosol, particles, gas, _ = _aerosol()
    runnable = _runnable()
    observed_gas: list[float] = []
    real_rate = runnable._rate_from_gas

    def spy_rate(source_gas: GasData) -> float:
        observed_gas.append(float(source_gas.concentration[0, 0]))
        return real_rate(source_gas)

    monkeypatch.setattr(runnable, "_rate_from_gas", spy_rate)
    initial_gas = float(gas.concentration[0, 0])
    molar_mass = float(gas.molar_mass[0])
    coefficient = 1.0e-2
    duration = 0.5
    per_event_mass = molar_mass / AVOGADRO_NUMBER
    expected_gas = initial_gas
    for _ in range(2):
        rate = coefficient * expected_gas * AVOGADRO_NUMBER / molar_mass
        potential = rate * duration
        admitted = min(potential, expected_gas / per_event_mass)
        expected_gas -= admitted * per_event_mass

    assert runnable.execute(aerosol, 1.0, sub_steps=2) is aerosol
    npt.assert_allclose(observed_gas[0], initial_gas)
    assert observed_gas[1] < observed_gas[0]
    npt.assert_allclose(gas.concentration, [[expected_gas]])
    npt.assert_allclose(
        np.einsum("bn,bns->bs", particles.concentration, particles.masses),
        [[initial_gas - expected_gas]],
    )
    assert aerosol.particles.data is particles
    assert aerosol.atmosphere.partitioning_species.data is gas


def test_source_errors_propagate_and_stop_later_stages(monkeypatch):
    """P1 and P2 errors retain identity and do not invoke later boundaries."""
    aerosol, _, gas, _ = _aerosol()
    runnable = _runnable()
    p1_error = RuntimeError("p1")
    monkeypatch.setattr(
        runnable,
        "_rate_from_gas",
        lambda _: (_ for _ in ()).throw(p1_error),
    )
    with pytest.raises(RuntimeError, match="p1"):
        runnable.execute(aerosol, 1.0)

    monkeypatch.setattr(runnable, "_rate_from_gas", lambda _: 1.0)
    p2_error = RuntimeError("p2")
    monkeypatch.setattr(
        particle_process,
        "_finalize_particle_source",
        lambda *args: (_ for _ in ()).throw(p2_error),
    )
    monkeypatch.setattr(
        particle_process,
        "_commit_particle_source",
        lambda *args: pytest.fail("P3 must not follow a P2 error"),
    )
    gas_before = gas.concentration.copy()
    with pytest.raises(RuntimeError, match="p2"):
        runnable.execute(aerosol, 1.0)
    npt.assert_array_equal(gas.concentration, gas_before)


def test_execute_preflights_topology_once(monkeypatch: pytest.MonkeyPatch):
    """Execution validates facade topology once before its sequential steps."""
    aerosol, _, _, _ = _aerosol()
    runnable = _runnable()
    topology = runnable._topology
    calls = 0

    def counted_topology(source_aerosol):
        nonlocal calls
        calls += 1
        return topology(source_aerosol)

    monkeypatch.setattr(runnable, "_topology", counted_topology)
    monkeypatch.setattr(runnable, "_rate_from_gas", lambda _: 0.0)

    assert runnable.execute(aerosol, 1.0, sub_steps=3) is aerosol
    assert calls == 1


def test_later_commit_failure_preserves_prior_substep(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failing later P3 transaction does not roll back an earlier commit."""
    aerosol, _, gas, _ = _aerosol()
    runnable = _runnable()
    commits = 0

    monkeypatch.setattr(runnable, "_rate_from_gas", lambda _: 1.0)
    monkeypatch.setattr(
        particle_process,
        "_finalize_particle_source",
        lambda *args: (object(), object()),
    )

    def commit(*args):
        nonlocal commits
        commits += 1
        if commits == 1:
            gas.concentration[0, 0] = 0.5e-12
            return
        raise RuntimeError("second P3 transaction")

    monkeypatch.setattr(particle_process, "_commit_particle_source", commit)

    with pytest.raises(RuntimeError, match="second P3 transaction"):
        runnable.execute(aerosol, 2.0, sub_steps=2)
    assert commits == 2
    npt.assert_array_equal(gas.concentration, [[0.5e-12]])


def test_runnable_sequence_preserves_process_order(
    monkeypatch: pytest.MonkeyPatch,
):
    """Nucleation composes through ``|`` with one inner substep per cycle."""
    aerosol, _, _, _ = _aerosol()
    runnable = _runnable()
    calls: list[tuple[str, int]] = []

    def execute_nucleation(source_aerosol, time_step, sub_steps=1):
        calls.append(("nucleation", sub_steps))
        return source_aerosol

    class Marker(RunnableABC):
        """Record execution after nucleation."""

        def rate(self, aerosol):
            """Return a marker rate."""
            return 0.0

        def execute(self, source_aerosol, time_step, sub_steps=1):
            """Record the delegated substep count."""
            calls.append(("marker", sub_steps))
            return source_aerosol

    monkeypatch.setattr(runnable, "execute", execute_nucleation)
    sequence = runnable | Marker()

    assert sequence.execute(aerosol, 2.0, sub_steps=2) is aerosol
    assert calls == [
        ("nucleation", 1),
        ("marker", 1),
        ("nucleation", 1),
        ("marker", 1),
    ]
