"""Tests for immutable, inventory-limited nucleation source records."""

from dataclasses import FrozenInstanceError

import numpy as np
import numpy.testing as npt
import particula.dynamics
import particula.dynamics.nucleation as nucleation
import pytest
from particula.dynamics.nucleation import particle_source
from particula.dynamics.nucleation.nucleation_strategies import (
    InjectionComposition,
)
from particula.dynamics.nucleation.particle_source import (
    PotentialEventData,
    SourceDemandData,
    SourceDiagnostics,
    finalize_particle_source,
)
from particula.gas.gas_data import GasData
from particula.util.constants import AVOGADRO_NUMBER


def _gas(concentration: np.ndarray) -> GasData:
    """Build three-species gas data with supplied mass concentrations."""
    return GasData(
        name=["a", "b", "unused"],
        molar_mass=np.array([0.1, 0.2, 0.3]),
        concentration=concentration,
        partitioning=np.array([True, True, False]),
    )


def test_finalization_uses_shared_count_and_per_box_limiting_species() -> None:
    """Every demand lane uses one inventory-admitted event count per box."""
    composition = InjectionComposition((1, 2, 0))
    per_event = np.array([0.1, 0.4, 0.0]) / AVOGADRO_NUMBER
    counts = np.array([10.0, 20.0])
    gas = _gas(
        np.array(
            [
                [50.0 * per_event[0], 10.0 * per_event[1], 1.0],
                [20.0 * per_event[0], 60.0 * per_event[1], 1.0],
            ]
        )
    )
    snapshot = gas.concentration.copy()

    demand, diagnostics = finalize_particle_source(
        PotentialEventData(counts, 1.0), composition, gas
    )

    expected_admitted = np.array([10.0, 20.0])
    npt.assert_allclose(demand.per_event_mass, per_event, rtol=1e-14)
    npt.assert_allclose(diagnostics.potential_event_count, counts)
    npt.assert_allclose(diagnostics.gas_admitted_event_count, expected_admitted)
    npt.assert_allclose(
        demand.gas_mass_removed,
        expected_admitted[:, None] * per_event[None, :],
        rtol=1e-14,
    )
    npt.assert_array_equal(diagnostics.gas_limited_event_count, [0.0, 0.0])
    npt.assert_array_equal(diagnostics.limiting_species_index, [-1, -1])
    npt.assert_array_equal(gas.concentration, snapshot)


def test_inventory_limit_ties_and_exact_depletion_are_deterministic() -> None:
    """Lowest original participating index wins an equal inventory tie."""
    composition = InjectionComposition((1, 2, 0))
    per_event = np.array([0.1, 0.4, 0.0]) / AVOGADRO_NUMBER
    gas = _gas(np.array([[3.0 * per_event[0], 3.0 * per_event[1], 0.0]]))

    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([7.0]), 1.0), composition, gas
    )

    npt.assert_allclose(diagnostics.gas_admitted_event_count, [3.0])
    npt.assert_allclose(diagnostics.gas_limited_event_count, [4.0])
    npt.assert_array_equal(diagnostics.limiting_species_index, [0])
    npt.assert_allclose(
        demand.gas_mass_removed[0, :2], gas.concentration[0, :2]
    )


def test_inventory_limits_each_box_by_its_tightest_participating_species() -> (
    None
):
    """Different boxes use one common count from their own limiting lane."""
    composition = InjectionComposition((1, 2, 0))
    per_event = np.array([0.1, 0.4, 0.0]) / AVOGADRO_NUMBER
    gas = _gas(
        np.array(
            [
                [9.0 * per_event[0], 20.0 * per_event[1], 3.0],
                [30.0 * per_event[0], 7.0 * per_event[1], 3.0],
            ]
        )
    )

    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([20.0, 20.0]), 1.0), composition, gas
    )

    expected_admitted = np.array([9.0, 7.0])
    npt.assert_allclose(diagnostics.gas_admitted_event_count, expected_admitted)
    npt.assert_allclose(
        demand.gas_mass_removed,
        expected_admitted[:, None] * per_event[None, :],
        rtol=1e-14,
    )
    npt.assert_array_equal(diagnostics.limiting_species_index, [0, 1])
    npt.assert_allclose(
        diagnostics.potential_event_count,
        diagnostics.gas_admitted_event_count
        + diagnostics.gas_limited_event_count,
    )
    assert np.all(demand.gas_mass_removed[:, :2] <= gas.concentration[:, :2])


@pytest.mark.parametrize(
    "rate,duration,concentration",
    [
        (np.array([0.0]), 1.0, np.array([[1.0, 1.0, 1.0]])),
        (np.array([2.0]), 0.0, np.array([[1.0, 1.0, 1.0]])),
        (np.array([2.0]), 1.0, np.array([[0.0, 1.0, 1.0]])),
    ],
)
def test_zero_paths_preserve_shapes_and_sentinel(
    rate: np.ndarray,
    duration: float,
    concentration: np.ndarray,
) -> None:
    """Zero potential or admitted counts use exact zeros and the sentinel."""
    demand, diagnostics = finalize_particle_source(
        PotentialEventData(rate, duration),
        InjectionComposition((1, 2, 0)),
        _gas(concentration),
    )

    assert demand.per_event_mass.shape == (3,)
    assert demand.gas_mass_removed.shape == (1, 3)
    npt.assert_array_equal(diagnostics.gas_admitted_event_count, [0.0])
    npt.assert_array_equal(demand.gas_mass_removed, np.zeros((1, 3)))
    npt.assert_array_equal(diagnostics.limiting_species_index, [-1])


def test_zero_participating_inventory_preserves_potential_count() -> None:
    """Zero inventory in every participating lane admits no events."""
    gas = _gas(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]]))

    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([1.5, 0.0]), 2.0),
        InjectionComposition((1, 2, 0)),
        gas,
    )

    npt.assert_allclose(diagnostics.potential_event_count, [3.0, 0.0])
    npt.assert_allclose(diagnostics.gas_admitted_event_count, [0.0, 0.0])
    npt.assert_array_equal(diagnostics.gas_limited_event_count, [3.0, 0.0])
    npt.assert_array_equal(diagnostics.limiting_species_index, [-1, -1])
    npt.assert_allclose(
        demand.per_event_mass,
        np.array([0.1, 0.4, 0.0]) / AVOGADRO_NUMBER,
    )
    npt.assert_array_equal(demand.gas_mass_removed, np.zeros((2, 3)))


def test_zero_box_input_returns_typed_readonly_empty_records() -> None:
    """The pure planning boundary supports the canonical empty batch shape."""
    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([], dtype=np.float64), 1.0),
        InjectionComposition((1, 0)),
        GasData(
            name=["a", "b"],
            molar_mass=np.array([0.1, 0.2]),
            concentration=np.empty((0, 2)),
            partitioning=np.array([True, False]),
        ),
    )

    assert demand.gas_mass_removed.shape == (0, 2)
    assert diagnostics.limiting_species_index.dtype == np.int32
    assert diagnostics.limiting_species_index.shape == (0,)
    assert not demand.gas_mass_removed.flags.writeable
    assert not diagnostics.potential_event_count.flags.writeable


@pytest.mark.parametrize(
    "record",
    [
        lambda: SourceDemandData(np.ones((1, 1)), np.ones((1, 1))),
        lambda: SourceDemandData(np.ones(1), np.ones(1)),
        lambda: SourceDiagnostics(
            np.ones((1, 1)), np.ones(1), np.ones(1), np.zeros(1, dtype=np.int32)
        ),
    ],
)
def test_output_records_require_documented_array_ranks(record: object) -> None:
    """Output records preserve their declared vector and matrix schemas."""
    with pytest.raises(ValueError, match="rank"):
        record()  # type: ignore[operator]


def test_records_defensively_copy_payloads_and_are_frozen() -> None:
    """Frozen records own read-only arrays rather than caller payloads."""
    rate = np.array([2.0])
    events = PotentialEventData(rate, 1.0)
    per_event = np.array([1.0])
    demand = SourceDemandData(per_event, np.array([[2.0]]))
    diagnostics = SourceDiagnostics(
        np.array([2.0]), np.array([1.0]), np.array([1.0]), np.array([0])
    )
    rate[0] = 3.0
    per_event[0] = 4.0

    npt.assert_array_equal(events.potential_rate, [2.0])
    npt.assert_array_equal(demand.per_event_mass, [1.0])
    for payload in (
        events.potential_rate,
        demand.gas_mass_removed,
        diagnostics.limiting_species_index,
    ):
        assert not payload.flags.writeable
    with pytest.raises(ValueError):
        events.potential_rate[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        events.duration = 2.0  # type: ignore[misc]


def test_private_array_and_scalar_validators_reject_invalid_payloads() -> None:
    """Private record helpers reject nonnumeric arrays and Boolean scalars."""
    with pytest.raises(TypeError, match="payload"):
        particle_source._readonly_copy(["invalid"], np.float64, "payload")
    with pytest.raises(ValueError, match="rank 1"):
        particle_source._readonly_copy([[1.0]], np.float64, "payload", ndim=1)

    assert particle_source._is_real_scalar(np.float64(1.0))
    assert not particle_source._is_real_scalar(True)
    assert not particle_source._is_real_scalar(np.array(1.0))


def test_finalization_outputs_do_not_alias_inputs() -> None:
    """Successful finalization returns independent immutable output arrays."""
    rate = np.array([4.0])
    gas = _gas(np.array([[1.0, 1.0, 1.0]]))
    demand, diagnostics = finalize_particle_source(
        PotentialEventData(rate, 1.0), InjectionComposition((1, 1, 0)), gas
    )

    for output in (
        demand.per_event_mass,
        demand.gas_mass_removed,
        diagnostics.potential_event_count,
        diagnostics.gas_admitted_event_count,
        diagnostics.gas_limited_event_count,
        diagnostics.limiting_species_index,
    ):
        assert not output.flags.writeable
        assert not np.shares_memory(output, rate)
        assert not np.shares_memory(output, gas.concentration)
    assert not np.shares_memory(demand.per_event_mass, gas.molar_mass)
    assert diagnostics.limiting_species_index.dtype == np.int32


@pytest.mark.parametrize(
    "factory,error,match",
    [
        (
            lambda: PotentialEventData(np.array([[1.0]]), 1.0),
            ValueError,
            "rank",
        ),
        (
            lambda: PotentialEventData(np.array([np.nan]), 1.0),
            ValueError,
            "finite",
        ),
        (
            lambda: PotentialEventData(np.array([-1.0]), 1.0),
            ValueError,
            "nonnegative",
        ),
        (
            lambda: PotentialEventData(np.array([1.0]), True),
            TypeError,
            "duration",
        ),
        (
            lambda: PotentialEventData(np.array([1.0]), np.inf),
            ValueError,
            "duration",
        ),
    ],
)
def test_potential_event_record_rejects_invalid_inputs(
    factory: object,
    error: type[Exception],
    match: str,
) -> None:
    """Potential event data validates array and scalar physical inputs."""
    with pytest.raises(error, match=match):
        factory()  # type: ignore[operator]


def test_finalization_rejects_invalid_gas_without_mutation() -> None:
    """Full concentration validation includes unused lanes before calculation."""
    gas = _gas(np.array([[1.0, 1.0, np.nan]]))
    snapshot = gas.concentration.copy()

    with pytest.raises(ValueError, match="concentration must be finite"):
        finalize_particle_source(
            PotentialEventData(np.array([1.0]), 1.0),
            InjectionComposition((1, 1, 0)),
            gas,
        )
    assert np.array_equal(gas.concentration, snapshot, equal_nan=True)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda gas: setattr(gas, "name", ["a"]), "composition width"),
        (
            lambda gas: setattr(gas, "molar_mass", np.array([[0.1, 0.2]])),
            "molar_mass must have shape",
        ),
        (
            lambda gas: setattr(gas, "concentration", np.array([1.0, 2.0])),
            "concentration shape",
        ),
        (
            lambda gas: setattr(gas, "partitioning", np.array([True])),
            "partitioning must have shape",
        ),
    ],
)
def test_finalization_rejects_mutated_gas_schemas_without_writing(
    mutate: object,
    match: str,
) -> None:
    """P2 revalidates mutable gas schemas before any source calculation."""
    gas = _gas(np.array([[1.0, 1.0, 1.0]]))
    mutate(gas)  # type: ignore[operator]
    concentration = gas.concentration.copy()

    with pytest.raises(ValueError, match=match):
        finalize_particle_source(
            PotentialEventData(np.array([1.0]), 1.0),
            InjectionComposition((1, 1, 0)),
            gas,
        )
    npt.assert_array_equal(gas.concentration, concentration)


def test_finalization_rejects_invalid_types_mass_and_derived_count() -> None:
    """P2 rejects invalid boundary objects and nonphysical derived inputs."""
    gas = _gas(np.array([[1.0, 1.0, 1.0]]))
    composition = InjectionComposition((1, 1, 0))
    events = PotentialEventData(np.array([1.0]), 1.0)

    with pytest.raises(TypeError, match="potential_events"):
        finalize_particle_source(
            None,  # type: ignore[arg-type]
            composition,
            gas,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="composition"):
        finalize_particle_source(events, None, gas)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gas"):
        finalize_particle_source(
            events,
            composition,
            None,  # type: ignore[arg-type]
        )

    gas.molar_mass[0] = 0.0
    with pytest.raises(ValueError, match="participating gas molar_mass"):
        finalize_particle_source(events, composition, gas)

    overflow_events = PotentialEventData(np.array([np.finfo(float).max]), 2.0)
    with pytest.raises(ValueError, match="potential_event_count"):
        finalize_particle_source(
            overflow_events,
            composition,
            _gas(np.ones((1, 3))),
        )


@pytest.mark.parametrize(
    "duration,error,match",
    [
        (None, TypeError, "duration"),
        ("one", TypeError, "duration"),
        (-1.0, ValueError, "nonnegative"),
    ],
)
def test_potential_event_record_rejects_other_invalid_durations(
    duration: object,
    error: type[Exception],
    match: str,
) -> None:
    """Duration accepts only finite, nonnegative non-Boolean real scalars."""
    with pytest.raises(error, match=match):
        PotentialEventData(np.array([1.0]), duration)  # type: ignore[arg-type]


def test_finalization_rejects_rounding_correction_that_cannot_reduce_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded correction raises rather than admitting an inventory overshoot."""
    composition = InjectionComposition((1, 0, 0))
    per_event = 0.1 / AVOGADRO_NUMBER
    gas = _gas(np.array([[0.1, 1.0, 1.0]]))
    events = PotentialEventData(np.array([2.0 / per_event]), 1.0)
    snapshot = gas.concentration.copy()
    original_minimum = particle_source.np.minimum

    def potential_count_only(
        potential: np.ndarray, inventory: np.ndarray
    ) -> np.ndarray:
        """Force the public finalizer through its bounded correction path."""
        del inventory
        return potential

    potential_count_only.reduce = original_minimum.reduce  # type: ignore[attr-defined]

    monkeypatch.setattr(
        particle_source.np,
        "minimum",
        potential_count_only,
    )
    monkeypatch.setattr(
        particle_source.np,
        "nextafter",
        lambda values, direction: values,
    )

    with pytest.raises(ValueError, match="remains out of inventory"):
        finalize_particle_source(events, composition, gas)
    npt.assert_array_equal(gas.concentration, snapshot)


def test_concrete_p2_names_are_not_package_exports() -> None:
    """P2 remains an intentionally concrete module-only boundary."""
    for module in (nucleation, particula.dynamics):
        assert not hasattr(module, "PotentialEventData")
        assert not hasattr(module, "finalize_particle_source")
