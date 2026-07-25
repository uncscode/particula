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
    FinalizedSourceDiagnostics,
    ParticleSourceCommitConfig,
    PotentialEventData,
    SourceDemandData,
    SourceDiagnostics,
    commit_particle_source,
    finalize_particle_source,
)
from particula.gas.gas_data import GasData
from particula.particles.exhaustion import (
    POLICY_SCALE_DEFERRED,
    ExhaustionControls,
)
from particula.particles.particle_data import ParticleData
from particula.util.constants import AVOGADRO_NUMBER

CONSERVATION_RTOL = 1e-12
CONSERVATION_ATOL = 1e-30


def _inventory(particles: ParticleData) -> np.ndarray:
    """Return concentration-weighted particle inventory by box and species."""
    return np.einsum(
        "bn,bns->bs",
        particles.concentration,
        particles.masses,
        dtype=np.float64,
        optimize=True,
    )


def _oracle_finalize(
    rate: np.ndarray,
    duration: float,
    molecule_counts: tuple[int, ...],
    molar_mass: np.ndarray,
    concentration: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Independently calculate the documented P2 float64 admission result."""
    counts = np.asarray(molecule_counts, dtype=np.float64)
    per_event = (
        counts * np.asarray(molar_mass, dtype=np.float64) / AVOGADRO_NUMBER
    )
    potential = np.asarray(rate, dtype=np.float64) * duration
    participating = np.flatnonzero(counts > 0.0)
    admitted = potential.copy()
    limiting = np.full(potential.shape, -1, dtype=np.int32)
    if participating.size:
        ratios = concentration[:, participating] / per_event[participating]
        candidate = np.min(ratios, axis=1)
        admitted = np.minimum(potential, candidate)
        limited = potential > admitted
        limiting[limited & (admitted > 0.0)] = participating[
            np.argmin(ratios, axis=1)[limited & (admitted > 0.0)]
        ]
        for _ in range(4):
            removal = admitted[:, None] * per_event[None, :]
            overshot = np.any(
                removal[:, participating] > concentration[:, participating],
                axis=1,
            )
            if not np.any(overshot):
                break
            admitted[overshot] = np.nextafter(admitted[overshot], -np.inf)
    removal = admitted[:, None] * per_event[None, :]
    return (
        per_event,
        potential,
        admitted,
        potential - admitted,
        limiting,
        removal,
    )


def _snapshot_arrays(*arrays: np.ndarray) -> tuple[tuple[object, ...], ...]:
    """Capture values and caller-visible array identity and metadata."""
    return tuple(
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


def _assert_snapshot(
    arrays: tuple[np.ndarray, ...], snapshots: tuple[tuple[object, ...], ...]
) -> None:
    """Assert arrays retain exact values, identity, and storage metadata."""
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


def _all_mutable_arrays(
    particles: ParticleData, gas: GasData
) -> tuple[np.ndarray, ...]:
    """Return every caller-owned mutable P3 particle and gas array."""
    return (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.concentration,
    )


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


def test_inventory_tie_uses_original_noncontiguous_species_index() -> None:
    """Tie diagnostics retain the original gas-species index."""
    composition = InjectionComposition((0, 1, 0, 2))
    per_event = np.array([0.0, 0.2, 0.0, 0.8]) / AVOGADRO_NUMBER
    gas = GasData(
        name=["unused_0", "a", "unused_2", "b"],
        molar_mass=np.array([0.1, 0.2, 0.3, 0.4]),
        concentration=np.array(
            [[1.0, 3.0 * per_event[1], 1.0, 3.0 * per_event[3]]]
        ),
        partitioning=np.array([False, True, False, True]),
    )

    _, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([7.0]), 1.0), composition, gas
    )

    npt.assert_array_equal(diagnostics.limiting_species_index, [1])


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
        (lambda gas: setattr(gas, "name", ["a"]), "gas name"),
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


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda gas: setattr(gas, "name", None), "gas name"),
        (lambda gas: setattr(gas, "name", ["a", 1, "unused"]), "gas name"),
        (
            lambda gas: setattr(gas, "partitioning", np.array([1, 1, 0])),
            "boolean array",
        ),
    ],
)
def test_finalization_rejects_malformed_gas_metadata_without_writing(
    mutate: object,
    match: str,
) -> None:
    """Malformed name and partitioning metadata fail deterministically."""
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


def test_finalization_recomputes_demand_after_final_ulp_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fourth bounded correction returns demand from its final admission."""
    composition = InjectionComposition((1, 0, 0))
    per_event = 0.1 / AVOGADRO_NUMBER
    gas = _gas(np.array([[10.0 * per_event, 1.0, 1.0]]))
    original_minimum = particle_source.np.minimum

    def four_ulps_above(
        potential: np.ndarray, inventory: np.ndarray
    ) -> np.ndarray:
        """Force the inventory correction to consume all four ULP steps."""
        del potential
        corrected = inventory.copy()
        for _ in range(4):
            corrected = np.nextafter(corrected, np.inf)
        return corrected

    four_ulps_above.reduce = original_minimum.reduce  # type: ignore[attr-defined]
    monkeypatch.setattr(particle_source.np, "minimum", four_ulps_above)

    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([20.0]), 1.0), composition, gas
    )

    npt.assert_allclose(
        diagnostics.gas_admitted_event_count, [10.0], rtol=1e-15
    )
    npt.assert_array_equal(
        demand.gas_mass_removed,
        diagnostics.gas_admitted_event_count[:, None]
        * demand.per_event_mass[None, :],
    )
    assert np.all(demand.gas_mass_removed <= gas.concentration)


@pytest.mark.parametrize(
    "composition,mutate,match",
    [
        (
            InjectionComposition((2**53 + 1, 0, 0)),
            lambda gas: None,
            "molecule_counts must be representable",
        ),
        (
            InjectionComposition((2**53, 0, 0)),
            lambda gas: gas.molar_mass.__setitem__(0, np.finfo(float).max),
            "per_event_mass must be finite",
        ),
    ],
)
def test_finalization_rejects_unrepresentable_composition_without_writing(
    composition: InjectionComposition,
    mutate: object,
    match: str,
) -> None:
    """Numeric composition failures are normalized and preserve gas state."""
    gas = _gas(np.array([[1.0, 1.0, 1.0]]))
    mutate(gas)  # type: ignore[operator]
    concentration = gas.concentration.copy()

    with pytest.raises(ValueError, match=match):
        finalize_particle_source(
            PotentialEventData(np.array([1.0]), 1.0), composition, gas
        )
    npt.assert_array_equal(gas.concentration, concentration)


def test_concrete_p2_names_are_not_package_exports() -> None:
    """P2 remains an intentionally concrete module-only boundary."""
    for module in (nucleation, particula.dynamics):
        assert not hasattr(module, "PotentialEventData")
        assert not hasattr(module, "finalize_particle_source")


def _source_particles(boxes: int = 1, capacity: int = 3) -> ParticleData:
    """Build valid fixed-slot particle storage with all slots free."""
    return ParticleData(
        masses=np.zeros((boxes, capacity, 1)),
        concentration=np.zeros((boxes, capacity)),
        charge=np.zeros((boxes, capacity)),
        density=np.array([1000.0]),
        volume=np.ones(boxes),
    )


def _commit_config(
    boxes: int = 1, scale: float = 1.0
) -> ParticleSourceCommitConfig:
    """Build permissive immutable P3 controls for deterministic tests."""
    return ParticleSourceCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(),
        requested_scale=np.full(boxes, scale),
        minimum_scale=np.full(boxes, scale),
        minimum_volume=np.full(boxes, 0.1),
    )


def _p2_records(
    events: np.ndarray,
) -> tuple[SourceDemandData, SourceDiagnostics]:
    """Build internally consistent one-species P2 records for P3 tests."""
    per_event = np.array([0.25])
    return (
        SourceDemandData(per_event, events[:, None] * per_event[None, :]),
        SourceDiagnostics(
            events,
            events,
            np.zeros_like(events),
            np.full(events.size, -1, dtype=np.int32),
        ),
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return a detached read-only payload for P2 forgery checks."""
    copied = values.copy()
    copied.setflags(write=False)
    return copied


def test_commit_particle_source_activates_ascending_slots_and_conserves() -> (
    None
):
    """P3 packages source demand equally and applies one atomic transfer."""
    particles = _source_particles(capacity=4)
    particles.masses[0, 1] = 0.5
    particles.concentration[0, 1] = 3.0
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([3.0]))
    initial = np.einsum("bn,bns->bs", particles.concentration, particles.masses)
    initial_gas = gas.concentration.copy()

    result = commit_particle_source(
        demand, diagnostics, particles, gas, _commit_config()
    )

    npt.assert_array_equal(result.requested_slot_count, [2])
    npt.assert_array_equal(result.activated_slot_count, [2])
    npt.assert_allclose(particles.masses[0, [0, 2], 0], [0.25, 0.25])
    npt.assert_allclose(particles.concentration[0, [0, 2]], [1.5, 1.5])
    npt.assert_allclose(particles.charge[0, [0, 2]], [1.0, 1.0])
    particle_mass = np.einsum(
        "bn,bns->bs", particles.concentration, particles.masses
    )
    npt.assert_allclose(
        particle_mass + gas.concentration, initial + initial_gas
    )
    npt.assert_allclose(result.conservation_residual, 0.0, atol=1e-30)
    assert not result.gas_mass_removed.flags.writeable


def test_commit_particle_source_zero_row_is_exact_noop() -> None:
    """Zero admitted demand neither activates slots nor changes gas."""
    particles = _source_particles()
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([0.0]))
    before = (
        particles.masses.copy(),
        particles.concentration.copy(),
        gas.concentration.copy(),
    )

    result = commit_particle_source(
        demand, diagnostics, particles, gas, _commit_config()
    )

    npt.assert_array_equal(result.requested_slot_count, [0])
    npt.assert_array_equal(result.activated_slot_count, [0])
    npt.assert_array_equal(particles.masses, before[0])
    npt.assert_array_equal(particles.concentration, before[1])
    npt.assert_array_equal(gas.concentration, before[2])


@pytest.mark.parametrize(
    ("event_count", "expected_slots"),
    [
        (0.0, 0),
        (2.0, 1),
        (4.0, 2),
        (4.01, 3),
    ],
)
def test_commit_particle_source_uses_ceiling_slot_counts(
    event_count: float,
    expected_slots: int,
) -> None:
    """P3 uses equal-weight prefixes at zero and ceiling-count boundaries."""
    particles = _source_particles(capacity=3)
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([event_count]))

    result = commit_particle_source(
        demand, diagnostics, particles, gas, _commit_config()
    )

    npt.assert_array_equal(result.requested_slot_count, [expected_slots])
    npt.assert_array_equal(result.activated_slot_count, [expected_slots])
    if expected_slots:
        npt.assert_allclose(
            particles.concentration[0, :expected_slots],
            event_count / expected_slots,
        )
        npt.assert_allclose(
            particles.masses[0, :expected_slots, 0],
            0.25,
        )
        npt.assert_allclose(particles.charge[0, :expected_slots], 1.0)
    npt.assert_array_equal(
        particles.concentration[0, expected_slots:],
        np.zeros(3 - expected_slots),
    )


def test_commit_particle_source_rejects_late_gas_underflow_atomically() -> None:
    """A final gas rejection leaves all caller-visible storage unchanged."""
    particles = _source_particles(boxes=2)
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0], [0.1]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([2.0, 2.0]))
    particle_snapshot = tuple(
        field.copy()
        for field in (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
        )
    )
    gas_snapshot = gas.concentration.copy()

    with pytest.raises(ValueError, match="final gas concentration"):
        commit_particle_source(
            demand, diagnostics, particles, gas, _commit_config(boxes=2)
        )

    for field, snapshot in zip(
        (
            particles.masses,
            particles.concentration,
            particles.charge,
            particles.density,
            particles.volume,
        ),
        particle_snapshot,
        strict=True,
    ):
        npt.assert_array_equal(field, snapshot)
    npt.assert_array_equal(gas.concentration, gas_snapshot)


def test_commit_particle_source_scales_gas_and_particle_inventory() -> None:
    """Scaling fallback preserves the scaled-domain particle-plus-gas balance."""
    particles = _source_particles(capacity=2)
    particles.masses[0, 0, 0] = 0.5
    particles.concentration[0, 0] = 4.0
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([3.0]))
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(
            resampling=False,
            representative_volume_scaling=True,
        ),
        requested_scale=np.array([0.5]),
        minimum_scale=np.array([0.5]),
        minimum_volume=np.array([0.1]),
    )

    result = commit_particle_source(demand, diagnostics, particles, gas, config)

    npt.assert_array_equal(
        result.exhaustion_policy_code, [POLICY_SCALE_DEFERRED]
    )
    npt.assert_allclose(result.representative_volume_scale, [0.5])
    npt.assert_allclose(result.represented_event_count, [1.5])
    npt.assert_allclose(result.representation_reduction_event_count, [1.5])
    npt.assert_array_equal(result.requested_slot_count, [1])
    npt.assert_array_equal(result.activated_slot_count, [1])
    npt.assert_allclose(particles.volume, [0.5])
    npt.assert_allclose(particles.concentration, [[2.0, 1.5]])
    npt.assert_allclose(gas.concentration, [[2.125]])
    particle_mass = np.einsum(
        "bn,bns->bs", particles.concentration, particles.masses
    )
    npt.assert_allclose(
        particle_mass + gas.concentration,
        [[3.5]],
        rtol=1e-12,
        atol=1e-30,
    )


def test_commit_config_and_finalized_diagnostics_own_payloads() -> None:
    """P3 immutable records detach caller vectors and arrays."""
    requested = np.array([1.0])
    minimum = np.array([1.0])
    minimum_volume = np.array([0.1])
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(),
        requested_scale=requested,
        minimum_scale=minimum,
        minimum_volume=minimum_volume,
    )
    record = FinalizedSourceDiagnostics(
        potential_event_count=np.array([1.0]),
        gas_admitted_event_count=np.array([1.0]),
        represented_event_count=np.array([1.0]),
        gas_limited_event_count=np.array([1.0]),
        representation_reduction_event_count=np.array([1.0]),
        residual_event_count=np.array([1.0]),
        limiting_species_index=np.array([-1], dtype=np.int32),
        gas_mass_removed=np.array([[1.0]]),
        requested_slot_count=np.array([1], dtype=np.int32),
        activated_slot_count=np.array([1], dtype=np.int32),
        released_slot_count=np.array([0], dtype=np.int32),
        exhaustion_policy_code=np.array([0], dtype=np.int32),
        representative_volume_scale=requested,
        conservation_residual=np.array([[0.0]]),
    )
    requested[0] = 0.5
    minimum[0] = 0.5
    minimum_volume[0] = 0.5

    npt.assert_array_equal(config.requested_scale, [1.0])
    npt.assert_array_equal(config.minimum_scale, [1.0])
    npt.assert_array_equal(config.minimum_volume, [0.1])
    npt.assert_array_equal(record.representative_volume_scale, [1.0])
    with pytest.raises(ValueError):
        record.represented_event_count[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        record.residual_event_count = np.array([0.0])  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda demand, diagnostics: object.__setattr__(
                demand,
                "per_event_mass",
                _readonly(np.array([0.25], dtype=np.float32)),
            ),
            "documented dtypes",
        ),
        (
            lambda demand, diagnostics: object.__setattr__(
                diagnostics,
                "limiting_species_index",
                _readonly(np.array([1], dtype=np.int32)),
            ),
            "limiting species indices",
        ),
    ],
)
def test_commit_particle_source_rejects_forged_p2_records_atomically(
    mutate: object,
    match: str,
) -> None:
    """Forged immutable P2 payloads fail before staging can reach callers."""
    particles = _source_particles()
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([2.0]))
    particle_snapshot = particles.copy()
    gas_snapshot = gas.concentration.copy()
    mutate(demand, diagnostics)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        commit_particle_source(
            demand,
            diagnostics,
            particles,
            gas,
            _commit_config(),
        )

    npt.assert_array_equal(particles.masses, particle_snapshot.masses)
    npt.assert_array_equal(
        particles.concentration, particle_snapshot.concentration
    )
    npt.assert_array_equal(particles.charge, particle_snapshot.charge)
    npt.assert_array_equal(particles.volume, particle_snapshot.volume)
    npt.assert_array_equal(gas.concentration, gas_snapshot)


def test_concrete_p3_names_are_not_package_exports() -> None:
    """P3 remains intentionally concrete to particle_source."""
    for module in (nucleation, particula.dynamics):
        assert not hasattr(module, "ParticleSourceCommitConfig")
        assert not hasattr(module, "FinalizedSourceDiagnostics")
        assert not hasattr(module, "commit_particle_source")


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("maximum_slot_weight", True, TypeError, "real scalar"),
        ("maximum_slot_weight", 0.0, ValueError, "positive"),
        ("source_charge", np.inf, ValueError, "finite"),
        ("radius_cubed_relative_error", 1, TypeError, "exact Python float"),
        ("minimum_scale", np.array([[1.0]]), ValueError, "rank 1"),
        ("minimum_volume", ["bad"], TypeError, "float64-compatible"),
    ],
)
def test_commit_config_rejects_invalid_controls_and_sidecars(
    field: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    """Commit configuration validates scalar and owned vector payloads."""
    arguments: dict[str, object] = dict(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(),
        requested_scale=np.array([1.0]),
        minimum_scale=np.array([1.0]),
        minimum_volume=np.array([0.1]),
    )
    arguments[field] = value

    with pytest.raises(error, match=match):
        ParticleSourceCommitConfig(**arguments)  # type: ignore[arg-type]


def test_private_commit_helpers_validate_counts_schema_and_inventory() -> None:
    """P3 helpers reject impossible counts and calculate weighted inventory."""
    particles = _source_particles(capacity=2)
    particles.masses[0, 0, 0] = 2.0
    particles.concentration[0, 0] = 3.0

    npt.assert_array_equal(
        particle_source._request_counts(np.array([0.0, 2.1]), 2.0, 2),
        np.array([0, 2], dtype=np.int32),
    )
    npt.assert_allclose(
        particle_source._weighted_particle_mass(particles), [[6.0]]
    )
    with pytest.raises(ValueError, match="exceeds particle capacity"):
        particle_source._request_counts(np.array([4.1]), 2.0, 2)

    particles.masses = particles.masses.astype(np.float32)
    with pytest.raises(ValueError, match="dtype float64"):
        particle_source._validate_commit_particle_schema(particles)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda particles, gas: setattr(
            gas.concentration.flags, "writeable", False
        ),
        lambda particles, gas: setattr(
            particles.concentration.flags, "writeable", False
        ),
    ],
)
def test_commit_rejects_nonwritable_or_noncontiguous_storage_without_writes(
    mutation: object,
) -> None:
    """P3 validates caller-owned writable storage before detached staging."""
    particles = _source_particles()
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([1.0]))
    snapshot = (
        particles.masses.copy(),
        particles.concentration.copy(),
        gas.concentration.copy(),
    )
    mutation(particles, gas)  # type: ignore[operator]

    with pytest.raises(ValueError, match="writable|contiguous"):
        commit_particle_source(
            demand, diagnostics, particles, gas, _commit_config()
        )

    npt.assert_array_equal(particles.masses, snapshot[0])
    npt.assert_array_equal(particles.concentration, snapshot[1])
    npt.assert_array_equal(gas.concentration, snapshot[2])


def test_commit_rejects_inconsistent_records_and_invalid_scaling_vectors() -> (
    None
):
    """P3 rejects forged P2 relationships and invalid P4 relationships early."""
    particles = _source_particles()
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([1.0]))
    object.__setattr__(
        diagnostics,
        "gas_limited_event_count",
        _readonly(np.array([1.0])),
    )
    with pytest.raises(ValueError, match="mutually inconsistent"):
        commit_particle_source(
            demand, diagnostics, particles, gas, _commit_config()
        )

    demand, diagnostics = _p2_records(np.array([1.0]))
    config = _commit_config()
    object.__setattr__(config, "requested_scale", _readonly(np.array([1.1])))
    with pytest.raises(ValueError, match="invalid representative"):
        commit_particle_source(demand, diagnostics, particles, gas, config)


def test_commit_resampling_precedes_scaling_for_full_capacity() -> None:
    """A releasable full row chooses resampling before scaling fallback."""
    particles = _source_particles(capacity=3)
    particles.masses[0, :, 0] = [0.2, 0.3, 0.4]
    particles.concentration[0] = [1.0, 1.0, 1.0]
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([2.0]))
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(True, True),
        requested_scale=np.array([0.5]),
        minimum_scale=np.array([0.5]),
        minimum_volume=np.array([0.1]),
    )

    result = commit_particle_source(demand, diagnostics, particles, gas, config)

    assert result.exhaustion_policy_code[0] != POLICY_SCALE_DEFERRED
    npt.assert_array_equal(result.representative_volume_scale, [1.0])
    npt.assert_array_equal(result.released_slot_count, [1])
    npt.assert_array_equal(result.activated_slot_count, [1])


def test_commit_scaling_reduces_oversized_provisional_request() -> None:
    """Scale fallback occurs before final fixed-capacity request validation."""
    particles = _source_particles(capacity=3)
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[10.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([10.0]))
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(False, True),
        requested_scale=np.array([0.5]),
        minimum_scale=np.array([0.5]),
        minimum_volume=np.array([0.1]),
    )

    result = commit_particle_source(demand, diagnostics, particles, gas, config)

    npt.assert_array_equal(
        result.exhaustion_policy_code, [POLICY_SCALE_DEFERRED]
    )
    npt.assert_allclose(result.represented_event_count, [5.0])
    npt.assert_array_equal(result.requested_slot_count, [3])


def test_commit_scaling_cannot_activate_a_full_store_atomically() -> None:
    """Scaling source demand does not create a free slot in a full store."""
    particles = _source_particles(capacity=3)
    particles.masses[:, :, 0] = 0.25
    particles.concentration[:] = 1.0
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([3.0]))
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(False, True),
        requested_scale=np.array([0.5]),
        minimum_scale=np.array([0.5]),
        minimum_volume=np.array([0.1]),
    )
    particle_snapshot = particles.copy()
    gas_snapshot = gas.concentration.copy()

    with pytest.raises(ValueError, match="free slot capacity"):
        commit_particle_source(demand, diagnostics, particles, gas, config)

    for name in ("masses", "concentration", "charge", "density", "volume"):
        npt.assert_array_equal(
            getattr(particles, name), getattr(particle_snapshot, name)
        )
    npt.assert_array_equal(gas.concentration, gas_snapshot)


def test_commit_full_capacity_with_policies_off_preserves_all_storage() -> None:
    """Policies-disabled capacity rejection preserves every mutable field."""
    particles = _source_particles(capacity=2)
    particles.masses[:, :, 0] = 0.25
    particles.concentration[:] = 1.0
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([1.0]))
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=2.0,
        source_charge=1.0,
        exhaustion_controls=ExhaustionControls(False, False),
        requested_scale=np.array([1.0]),
        minimum_scale=np.array([1.0]),
        minimum_volume=np.array([0.1]),
    )
    fields = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.concentration,
    )
    snapshots = tuple(field.copy() for field in fields)
    metadata = tuple(
        (
            id(field),
            field.shape,
            field.dtype,
            field.flags.c_contiguous,
            field.flags.writeable,
        )
        for field in fields
    )

    with pytest.raises(ValueError, match="cannot represent"):
        commit_particle_source(demand, diagnostics, particles, gas, config)

    for field, snapshot, expected in zip(
        fields, snapshots, metadata, strict=True
    ):
        assert (
            id(field),
            field.shape,
            field.dtype,
            field.flags.c_contiguous,
            field.flags.writeable,
        ) == expected
        npt.assert_array_equal(field, snapshot)


def test_commit_multibox_multispecies_conserves_each_box_and_species() -> None:
    """P3 places independent requests and removes matching multi-species gas."""
    particles = ParticleData(
        masses=np.zeros((2, 3, 2)),
        concentration=np.zeros((2, 3)),
        charge=np.zeros((2, 3)),
        density=np.array([1000.0, 1200.0]),
        volume=np.ones(2),
    )
    gas = GasData(
        name=["a", "b"],
        molar_mass=np.array([0.1, 0.2]),
        concentration=np.array([[5.0, 6.0], [7.0, 8.0]]),
        partitioning=np.array([True, True]),
    )
    events = np.array([2.0, 3.0])
    per_event = np.array([0.25, 0.5])
    demand = SourceDemandData(per_event, events[:, None] * per_event)
    diagnostics = SourceDiagnostics(
        events, events, np.zeros(2), np.full(2, -1, dtype=np.int32)
    )
    initial_gas = gas.concentration.copy()

    result = commit_particle_source(
        demand, diagnostics, particles, gas, _commit_config(boxes=2)
    )

    npt.assert_array_equal(result.requested_slot_count, [1, 2])
    npt.assert_allclose(
        gas.concentration, initial_gas - events[:, None] * per_event
    )
    npt.assert_allclose(
        np.einsum("bn,bns->bs", particles.concentration, particles.masses)
        + gas.concentration,
        initial_gas,
    )


def test_commit_rejects_tiny_forged_gas_demand_atomically() -> None:
    """A tolerance-sized forged P2 mass demand cannot activate a source slot."""
    particles = _source_particles()
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0]]),
        partitioning=np.array([True]),
    )
    demand, diagnostics = _p2_records(np.array([0.0]))
    object.__setattr__(
        diagnostics, "gas_admitted_event_count", _readonly(np.array([1e-31]))
    )
    particle_snapshot = particles.copy()
    gas_snapshot = gas.concentration.copy()

    with pytest.raises(ValueError, match="mutually inconsistent"):
        commit_particle_source(
            demand, diagnostics, particles, gas, _commit_config()
        )

    npt.assert_array_equal(particles.masses, particle_snapshot.masses)
    npt.assert_array_equal(
        particles.concentration, particle_snapshot.concentration
    )
    npt.assert_array_equal(gas.concentration, gas_snapshot)


@pytest.mark.parametrize("target", ["particle", "gas"])
def test_commit_rejects_overlapping_mutable_storage_atomically(
    target: str,
) -> None:
    """Zero-stride mutable views reject before the transaction stages writes."""
    particles = _source_particles(boxes=2)
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[5.0], [5.0]]),
        partitioning=np.array([True]),
    )
    if target == "particle":
        particles.concentration = np.lib.stride_tricks.as_strided(
            np.zeros(1), shape=(2, 3), strides=(0, 0)
        )
    else:
        gas.concentration = np.lib.stride_tricks.as_strided(
            np.full(1, 5.0), shape=(2, 1), strides=(0, 0)
        )
    demand, diagnostics = _p2_records(np.array([1.0, 1.0]))
    particle_snapshot = particles.copy()
    gas_snapshot = gas.concentration.copy()

    with pytest.raises(ValueError, match="contiguous"):
        commit_particle_source(
            demand, diagnostics, particles, gas, _commit_config(boxes=2)
        )

    npt.assert_array_equal(particles.masses, particle_snapshot.masses)
    npt.assert_array_equal(
        particles.concentration, particle_snapshot.concentration
    )
    npt.assert_array_equal(gas.concentration, gas_snapshot)


@pytest.mark.parametrize("limiting_lane", [0, 1, 2])
def test_finalization_matches_independent_oracle_for_every_limiting_lane(
    limiting_lane: int,
) -> None:
    """P2 admission selects each participating multi-species lane in turn."""
    molecule_counts = (1, 2, 3, 0)
    molar_mass = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    per_event = np.asarray(molecule_counts, dtype=np.float64) * molar_mass
    per_event /= AVOGADRO_NUMBER
    concentration = np.full((1, 4), 100.0 * per_event, dtype=np.float64)
    concentration[0, limiting_lane] = 3.0 * per_event[limiting_lane]
    concentration[0, 3] = 7.0
    gas = GasData(
        name=["a", "b", "c", "inert"],
        molar_mass=molar_mass,
        concentration=concentration.copy(),
        partitioning=np.array([True, True, True, False]),
    )
    expected = _oracle_finalize(
        np.array([10.0]), 1.0, molecule_counts, molar_mass, concentration
    )

    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([10.0]), 1.0),
        InjectionComposition(molecule_counts),
        gas,
    )

    per_event_expected, potential, admitted, limited, limiting, removal = (
        expected
    )
    npt.assert_allclose(demand.per_event_mass, per_event_expected)
    npt.assert_allclose(demand.gas_mass_removed, removal)
    npt.assert_allclose(diagnostics.potential_event_count, potential)
    npt.assert_allclose(diagnostics.gas_admitted_event_count, admitted)
    npt.assert_allclose(diagnostics.gas_limited_event_count, limited)
    npt.assert_array_equal(diagnostics.limiting_species_index, limiting)


def test_finalization_oracle_covers_inert_and_no_participating_rows() -> None:
    """The standalone oracle retains potential when no lane participates."""
    concentration = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    expected = _oracle_finalize(
        np.array([2.0, 3.0]), 2.0, (0, 0), np.array([0.1, 0.2]), concentration
    )
    npt.assert_allclose(expected[0], [0.0, 0.0])
    npt.assert_allclose(expected[-1], 0.0)
    npt.assert_allclose(expected[2], expected[1])
    npt.assert_array_equal(expected[4], [-1, -1])


def test_commit_multibox_oracle_proves_per_species_source_transfer() -> None:
    """P2-to-P3 coupling conserves each box/species source transfer separately."""
    particles = ParticleData(
        masses=np.zeros((2, 4, 2), dtype=np.float64),
        concentration=np.zeros((2, 4), dtype=np.float64),
        charge=np.zeros((2, 4), dtype=np.float64),
        density=np.array([1000.0, 1200.0]),
        volume=np.ones(2),
    )
    counts = (1, 2)
    molar_mass = np.array([0.1, 0.2])
    per_event = np.asarray(counts) * molar_mass / AVOGADRO_NUMBER
    concentration = np.array(
        [
            [4.0 * per_event[0], 9.0 * per_event[1]],
            [8.0 * per_event[0], 3.0 * per_event[1]],
        ],
        dtype=np.float64,
    )
    gas = GasData(
        name=["a", "b"],
        molar_mass=molar_mass,
        concentration=concentration.copy(),
        partitioning=np.array([True, True]),
    )
    expected = _oracle_finalize(
        np.array([10.0, 10.0]), 1.0, counts, molar_mass, concentration
    )
    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([10.0, 10.0]), 1.0),
        InjectionComposition(counts),
        gas,
    )
    particle_before = _inventory(particles)
    gas_before = gas.concentration.copy()

    result = commit_particle_source(
        demand, diagnostics, particles, gas, _commit_config(boxes=2)
    )

    npt.assert_allclose(result.gas_mass_removed, expected[-1])
    npt.assert_array_equal(result.limiting_species_index, expected[4])
    particle_gain = _inventory(particles) - particle_before
    gas_removal = gas_before - gas.concentration
    npt.assert_allclose(
        particle_gain,
        gas_removal,
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_ATOL,
    )
    npt.assert_allclose(
        result.conservation_residual, 0.0, atol=CONSERVATION_ATOL
    )


def test_p2_and_p3_rejections_preserve_all_accessible_metadata() -> None:
    """Invalid P2/P3 input is read-only; P3 has no caller diagnostic buffer."""
    gas = _gas(np.array([[1.0, 1.0, 1.0]], dtype=np.float64))
    rate = np.array([1.0])
    rate_snapshot = _snapshot_arrays(rate)
    gas.concentration[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        finalize_particle_source(
            PotentialEventData(rate, 1.0), InjectionComposition((1, 2, 0)), gas
        )
    _assert_snapshot((rate,), rate_snapshot)
    assert not hasattr(commit_particle_source, "diagnostic_buffer")
    gas.concentration[0, 2] = 1.0
    particles = _source_particles()
    gas = GasData(
        name=["a"],
        molar_mass=np.array([0.1]),
        concentration=np.array([[1.0]]),
        partitioning=np.array([True]),
    )
    snapshots = _snapshot_arrays(*_all_mutable_arrays(particles, gas))
    demand, diagnostics = _p2_records(np.array([1.0]))
    object.__setattr__(
        config := _commit_config(),
        "requested_scale",
        _readonly(np.array([1.1])),
    )
    with pytest.raises(ValueError, match="invalid representative"):
        commit_particle_source(demand, diagnostics, particles, gas, config)
    _assert_snapshot(_all_mutable_arrays(particles, gas), snapshots)
