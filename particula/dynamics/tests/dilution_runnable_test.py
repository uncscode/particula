"""Regression tests for the dilution strategy and runnable."""

# ruff: noqa: D100, D101, D102, D103, D107

import numpy as np
import numpy.testing as npt
import pytest
from particula.aerosol import Aerosol
from particula.dynamics.dilution import DilutionStrategy, get_dilution_rate
from particula.dynamics.particle_process import Dilution
from particula.gas.atmosphere import Atmosphere
from particula.gas.species import GasSpecies
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import MassBasedMovingBin
from particula.particles.representation import ParticleRepresentation
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.runnable import RunnableABC, RunnableSequence


def _make_aerosol() -> Aerosol:
    """Build a deterministic aerosol with particle and gas concentrations."""
    particles = ParticleRepresentation(
        strategy=MassBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.array([1e-18, 2e-18], dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        concentration=np.array([4.0, 8.0], dtype=np.float64),
        charge=np.array([1.0, -1.0], dtype=np.float64),
        volume=2.0,
    )
    partitioning = GasSpecies(
        name="partitioning",
        molar_mass=0.1,
        concentration=3.0,
        partitioning=True,
    )
    gas_only = GasSpecies(
        name=np.array(["gas_a", "gas_b"]),
        molar_mass=np.array([0.02, 0.03], dtype=np.float64),
        concentration=np.array([5.0, 7.0], dtype=np.float64),
        partitioning=False,
    )
    atmosphere = Atmosphere(
        temperature=298.15,
        total_pressure=101325.0,
        partitioning_species=partitioning,
        gas_only_species=gas_only,
    )
    return Aerosol(atmosphere=atmosphere, particles=particles)


def _concentrations(
    aerosol: Aerosol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy all concentration domains from an aerosol."""
    return (
        aerosol.particles.get_concentration().copy(),
        np.asarray(
            aerosol.atmosphere.partitioning_species.get_concentration()
        ).copy(),
        aerosol.atmosphere.gas_only_species.get_concentration().copy(),
    )


@pytest.mark.parametrize(
    ("coefficient", "exception", "message"),
    [
        (None, TypeError, "coefficient.*numeric"),
        ("invalid", TypeError, "coefficient.*numeric"),
        (object(), TypeError, "coefficient.*numeric"),
        (-1.0, ValueError, "coefficient.*nonnegative"),
        (np.nan, ValueError, "coefficient.*finite"),
        (np.inf, ValueError, "coefficient.*finite"),
        (np.array([1.0]), ValueError, "coefficient.*scalar"),
    ],
)
def test_dilution_strategy_rejects_invalid_coefficients(
    coefficient, exception, message
):
    """Strategy construction preserves the scalar coefficient contract."""
    with pytest.raises(exception, match=message):
        DilutionStrategy(coefficient)


@pytest.mark.parametrize(
    "coefficient",
    [0.0, 0.25, np.float64(0.5), np.array(0.75)],
)
def test_dilution_strategy_retains_valid_coefficients(coefficient):
    """Python and NumPy scalar coefficients become finite nonnegative values."""
    strategy = DilutionStrategy(coefficient)

    assert np.isfinite(strategy.coefficient)
    assert strategy.coefficient >= 0.0
    assert strategy.coefficient == coefficient


def test_dilution_strategy_rate_and_step_follow_primitive_contract():
    """Rate uses physical concentration and step decays all domains in place."""
    aerosol = _make_aerosol()
    strategy = DilutionStrategy(0.25)
    sources = _concentrations(aerosol)

    npt.assert_allclose(
        strategy.rate(aerosol),
        get_dilution_rate(0.25, sources[0]),
    )
    assert strategy.step(aerosol, 4.0) is aerosol

    decay = np.exp(-0.25 * 4.0)
    for result, source in zip(_concentrations(aerosol), sources, strict=True):
        npt.assert_allclose(result, source * decay, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("time_step", "exception", "message"),
    [
        (None, TypeError, "time_step.*numeric"),
        ("invalid", TypeError, "time_step.*numeric"),
        (-1.0, ValueError, "time_step.*nonnegative"),
        (np.nan, ValueError, "time_step.*finite"),
        (np.inf, ValueError, "time_step.*finite"),
        (np.array([1.0]), ValueError, "time_step.*scalar"),
    ],
)
def test_dilution_strategy_step_propagates_validation_without_mutation(
    time_step, exception, message
):
    """Direct steps retain P2 validation and atomicity unchanged."""
    aerosol = _make_aerosol()
    sources = _concentrations(aerosol)

    with pytest.raises(exception, match=message):
        DilutionStrategy(0.25).step(aerosol, time_step)

    for result, source in zip(_concentrations(aerosol), sources, strict=True):
        npt.assert_array_equal(result, source)


@pytest.mark.parametrize("sub_steps", [1, 2, np.int64(4)])
def test_dilution_execute_substeps_matches_exact_decay(sub_steps):
    """Repeated equal steps yield the expected finite exponential decay."""
    aerosol = _make_aerosol()
    sources = _concentrations(aerosol)
    coefficient = 0.25
    total_time_step = 4.0

    assert (
        Dilution(DilutionStrategy(coefficient)).execute(
            aerosol,
            total_time_step,
            sub_steps,
        )
        is aerosol
    )

    decay = np.exp(-coefficient * total_time_step)
    for result, source in zip(_concentrations(aerosol), sources, strict=True):
        npt.assert_allclose(result, source * decay, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("coefficient, time_step", [(0.0, 2.0), (0.5, 0.0)])
def test_dilution_execute_no_ops_are_exact(coefficient, time_step):
    """Zero coefficient or duration leaves every concentration unchanged."""
    aerosol = _make_aerosol()
    sources = _concentrations(aerosol)

    Dilution(DilutionStrategy(coefficient)).execute(aerosol, time_step, 2)

    for result, source in zip(_concentrations(aerosol), sources, strict=True):
        npt.assert_array_equal(result, source)


@pytest.mark.parametrize(
    "coefficient, time_step", [(1e100, 1e100), (1e300, 2.0)]
)
def test_dilution_execute_large_finite_decay_remains_physical(
    coefficient, time_step
):
    """Large finite products underflow cleanly without negative concentrations."""
    aerosol = _make_aerosol()

    Dilution(DilutionStrategy(coefficient)).execute(aerosol, time_step)

    for concentration in _concentrations(aerosol):
        assert np.all(np.isfinite(concentration))
        assert np.all(concentration >= 0.0)


class SpyStrategy:
    """Record delegation calls without changing an aerosol."""

    def __init__(self, rate_result):
        self.rate_result = rate_result
        self.calls: list[tuple[Aerosol, float]] = []

    def rate(self, aerosol: Aerosol):
        return self.rate_result

    def step(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        self.calls.append((aerosol, time_step))
        return aerosol


def test_dilution_delegates_rate_and_equal_substeps():
    """Runnable returns the exact rate object and calls the strategy per step."""
    aerosol = _make_aerosol()
    rate_result = np.array([-1.0, -2.0])
    strategy = SpyStrategy(rate_result)
    dilution = Dilution(strategy)  # type: ignore[arg-type]

    assert dilution.rate(aerosol) is rate_result
    assert dilution.execute(aerosol, 3.0, 3) is aerosol
    assert [call[0] for call in strategy.calls] == [aerosol] * 3
    npt.assert_allclose([call[1] for call in strategy.calls], [1.0, 1.0, 1.0])


@pytest.mark.parametrize("sub_steps", [0, -1, True, False, 1.0, "two", None])
def test_dilution_rejects_invalid_substeps_before_delegation(sub_steps):
    """Invalid substep counts fail before strategy calls or mutation."""
    aerosol = _make_aerosol()
    sources = _concentrations(aerosol)
    strategy = SpyStrategy(0.0)

    with pytest.raises(ValueError, match="sub_steps.*positive integer"):
        Dilution(strategy).execute(aerosol, 1.0, sub_steps)  # type: ignore[arg-type]

    assert strategy.calls == []
    for result, source in zip(_concentrations(aerosol), sources, strict=True):
        npt.assert_array_equal(result, source)


@pytest.mark.parametrize(
    ("time_step", "exception", "message"),
    [
        (None, TypeError, "time_step.*numeric"),
        ("invalid", TypeError, "time_step.*numeric"),
        (-1.0, ValueError, "time_step.*nonnegative"),
        (np.nan, ValueError, "time_step.*finite"),
        (np.inf, ValueError, "time_step.*finite"),
        (np.array([1.0]), ValueError, "time_step.*scalar"),
    ],
)
def test_dilution_rejects_invalid_duration_before_delegation(
    time_step, exception, message
):
    """Invalid total durations fail before a strategy call or mutation."""
    aerosol = _make_aerosol()
    sources = _concentrations(aerosol)
    strategy = SpyStrategy(0.0)

    with pytest.raises(exception, match=message):
        Dilution(strategy).execute(aerosol, time_step)  # type: ignore[arg-type]

    assert strategy.calls == []
    for result, source in zip(_concentrations(aerosol), sources, strict=True):
        npt.assert_array_equal(result, source)


class NonconformingStrategy(SpyStrategy):
    """Return another aerosol to verify runnable identity ownership."""

    def __init__(self, alternate: Aerosol):
        super().__init__(0.0)
        self.alternate = alternate

    def step(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        self.calls.append((aerosol, time_step))
        return self.alternate


def test_dilution_ignores_nonconforming_strategy_return_value():
    """Runnable always retains the input aerosol across strategy calls."""
    aerosol = _make_aerosol()
    alternate = _make_aerosol()
    strategy = NonconformingStrategy(alternate)

    assert Dilution(strategy).execute(aerosol, 2.0, 2) is aerosol
    assert all(call_aerosol is aerosol for call_aerosol, _ in strategy.calls)
    assert alternate is not aerosol


class OrderedSpyStrategy(SpyStrategy):
    """Record dilution execution order in addition to substep calls."""

    def __init__(self, events: list[str]):
        super().__init__(0.0)
        self.events = events

    def step(self, aerosol: Aerosol, time_step: float) -> Aerosol:
        self.events.append("dilution")
        return super().step(aerosol, time_step)


class Marker(RunnableABC):
    """Record runnable ordering while retaining aerosol identity."""

    def __init__(self, calls: list[str]):
        self.calls = calls

    def rate(self, aerosol: Aerosol) -> float:
        return 0.0

    def execute(
        self,
        aerosol: Aerosol,
        time_step: float,
        sub_steps: int = 1,
    ) -> Aerosol:
        self.calls.append("marker")
        return aerosol


def test_dilution_composes_in_runnable_sequence_order():
    """Sequences interleave dilution and following processes per outer step."""
    aerosol = _make_aerosol()
    calls: list[str] = []
    strategy = OrderedSpyStrategy(calls)
    marker = Marker(calls)
    dilution = Dilution(strategy)  # type: ignore[arg-type]

    assert (dilution | marker).execute(aerosol, 1.0) is aerosol
    assert calls == ["dilution", "marker"]

    calls.clear()
    strategy.calls.clear()
    sequence = RunnableSequence()
    sequence.add_process(dilution)
    sequence.add_process(marker)
    assert sequence.execute(aerosol, 4.0, sub_steps=2) is aerosol

    assert calls == ["dilution", "marker", "dilution", "marker"]
    assert len(strategy.calls) == 2
    npt.assert_allclose([call[1] for call in strategy.calls], [2.0, 2.0])
