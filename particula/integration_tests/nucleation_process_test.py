"""Deterministic CPU P2/P3 nucleation validation; no replay or performance claim."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics.nucleation.nucleation_strategies import (
    InjectionComposition,
)
from particula.dynamics.nucleation.particle_source import (
    ParticleSourceCommitConfig,
    PotentialEventData,
    commit_particle_source,
    finalize_particle_source,
)
from particula.gas.gas_data import GasData
from particula.particles.exhaustion import (
    POLICY_ACTIVATE,
    POLICY_SCALE_DEFERRED,
    ExhaustionControls,
)
from particula.particles.particle_data import ParticleData
from particula.util.constants import AVOGADRO_NUMBER

CONSERVATION_RTOL = 1e-12
CONSERVATION_ATOL = 1e-30


def _oracle(
    rate: np.ndarray,
    duration: float,
    counts: tuple[int, ...],
    molar_mass: np.ndarray,
    concentration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate independent float64 P2 potential, admission, limiter, demand."""
    per_event = np.asarray(counts, dtype=np.float64) * molar_mass
    per_event /= AVOGADRO_NUMBER
    potential = rate * duration
    participating = np.flatnonzero(np.asarray(counts) > 0)
    ratios = concentration[:, participating] / per_event[participating]
    admitted = np.minimum(potential, np.min(ratios, axis=1))
    limiting = np.full(rate.shape, -1, dtype=np.int32)
    reduced = potential > admitted
    limiting[reduced & (admitted > 0.0)] = participating[
        np.argmin(ratios, axis=1)[reduced & (admitted > 0.0)]
    ]
    for _ in range(4):
        removal = admitted[:, None] * per_event[None, :]
        overshot = np.any(
            removal[:, participating] > concentration[:, participating], axis=1
        )
        admitted[overshot] = np.nextafter(admitted[overshot], -np.inf)
    return per_event, potential, admitted, limiting


def _inventory(particles: ParticleData) -> np.ndarray:
    """Calculate concentration-weighted particle inventory."""
    return np.einsum(
        "bn,bns->bs",
        particles.concentration,
        particles.masses,
        dtype=np.float64,
        optimize=True,
    )


def _snapshot(*arrays: np.ndarray) -> tuple[tuple[object, ...], ...]:
    """Capture exact values and caller-visible array metadata."""
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


def _assert_unchanged(
    arrays: tuple[np.ndarray, ...], snapshots: tuple[tuple[object, ...], ...]
) -> None:
    """Assert that a rejected direct transaction changed no caller state."""
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


def _mutable_arrays(
    particles: ParticleData, gas: GasData
) -> tuple[np.ndarray, ...]:
    """Return all caller-owned writable arrays at the P3 boundary."""
    return (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.concentration,
    )


def _fixture() -> tuple[ParticleData, GasData, ParticleSourceCommitConfig]:
    """Build a bounded multi-box, multi-species direct P2/P3 fixture."""
    particles = ParticleData(
        masses=np.zeros((2, 4, 2), dtype=np.float64),
        concentration=np.zeros((2, 4), dtype=np.float64),
        charge=np.zeros((2, 4), dtype=np.float64),
        density=np.array([1000.0, 1200.0]),
        volume=np.ones(2),
    )
    gas = GasData(
        name=["a", "b"],
        molar_mass=np.array([0.1, 0.2]),
        concentration=np.zeros((2, 2), dtype=np.float64),
        partitioning=np.array([True, True]),
    )
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=10.0,
        source_charge=0.0,
        exhaustion_controls=ExhaustionControls(False, False),
        requested_scale=np.ones(2),
        minimum_scale=np.ones(2),
        minimum_volume=np.full(2, 0.1),
    )
    return particles, gas, config


def test_direct_multibox_cycles_match_oracle_and_current_gas_coupling() -> None:
    """Two direct cycles use current gas and conserve every box/species lane."""
    particles, gas, config = _fixture()
    counts = (1, 2)
    per_event = np.asarray(counts) * gas.molar_mass / AVOGADRO_NUMBER
    gas.concentration[:] = [
        [4.0 * per_event[0], 10.0 * per_event[1]],
        [10.0 * per_event[0], 3.0 * per_event[1]],
    ]
    rates = np.array([6.0, 6.0])
    for _ in range(2):
        concentration_before = gas.concentration.copy()
        particle_before = _inventory(particles)
        expected = _oracle(
            rates, 1.0, counts, gas.molar_mass, concentration_before
        )
        demand, diagnostics = finalize_particle_source(
            PotentialEventData(rates, 1.0), InjectionComposition(counts), gas
        )
        result = commit_particle_source(
            demand, diagnostics, particles, gas, config
        )

        expected_per_event, potential, admitted, limiting = expected
        npt.assert_allclose(demand.per_event_mass, expected_per_event)
        npt.assert_allclose(diagnostics.potential_event_count, potential)
        npt.assert_allclose(diagnostics.gas_admitted_event_count, admitted)
        npt.assert_array_equal(diagnostics.limiting_species_index, limiting)
        npt.assert_allclose(result.represented_event_count, admitted)
        npt.assert_allclose(result.representation_reduction_event_count, 0.0)
        npt.assert_allclose(result.residual_event_count, 0.0)
        expected_slots = np.ceil(admitted / 10.0).astype(np.int32)
        npt.assert_array_equal(result.requested_slot_count, expected_slots)
        npt.assert_array_equal(result.activated_slot_count, expected_slots)
        removal = concentration_before - gas.concentration
        npt.assert_allclose(
            _inventory(particles) - particle_before,
            removal,
            rtol=CONSERVATION_RTOL,
            atol=CONSERVATION_ATOL,
        )
        npt.assert_allclose(
            result.gas_mass_removed,
            admitted[:, None] * expected_per_event[None, :],
            rtol=CONSERVATION_RTOL,
            atol=CONSERVATION_ATOL,
        )


def test_direct_p3_policy_failure_is_write_free() -> None:
    """A no-policy full-capacity P3 rejection preserves all direct inputs."""
    particles, gas, config = _fixture()
    particles.masses[:] = 0.25
    particles.concentration[:] = 1.0
    gas.concentration[:] = 1.0
    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([1.0, 1.0]), 1.0),
        InjectionComposition((1, 1)),
        gas,
    )
    arrays = _mutable_arrays(particles, gas)
    snapshots = _snapshot(*arrays)

    with pytest.raises(ValueError, match="cannot represent"):
        commit_particle_source(demand, diagnostics, particles, gas, config)

    _assert_unchanged(arrays, snapshots)


def test_direct_scaling_fallback_preserves_per_lane_scaled_balance() -> None:
    """P3 scales selected rows before adding their finalized source transfer."""
    particles, gas, _ = _fixture()
    particles.masses[0, :3] = [0.25, 0.5]
    particles.concentration[0, :3] = 2.0
    counts = (1, 2)
    per_event = np.asarray(counts, dtype=np.float64) * gas.molar_mass
    per_event /= AVOGADRO_NUMBER
    gas.concentration[:] = [
        [100.0 * per_event[0], 100.0 * per_event[1]],
        [0.0, 0.0],
    ]
    config = ParticleSourceCommitConfig(
        maximum_slot_weight=10.0,
        source_charge=0.0,
        exhaustion_controls=ExhaustionControls(False, True),
        requested_scale=np.array([0.5, 1.0]),
        minimum_scale=np.array([0.5, 1.0]),
        minimum_volume=np.array([0.1, 0.1]),
    )
    particle_before = _inventory(particles)
    gas_before = gas.concentration.copy()

    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([15.0, 0.0]), 1.0),
        InjectionComposition(counts),
        gas,
    )
    result = commit_particle_source(demand, diagnostics, particles, gas, config)

    expected_scale = np.array([0.5, 1.0])
    expected_represented = np.array([7.5, 0.0])
    expected_removal = expected_represented[:, None] * per_event[None, :]
    npt.assert_array_equal(
        result.exhaustion_policy_code,
        [POLICY_SCALE_DEFERRED, POLICY_ACTIVATE],
    )
    npt.assert_allclose(result.representative_volume_scale, expected_scale)
    npt.assert_allclose(result.represented_event_count, expected_represented)
    npt.assert_allclose(result.representation_reduction_event_count, [7.5, 0.0])
    npt.assert_array_equal(result.requested_slot_count, [1, 0])
    npt.assert_allclose(result.gas_mass_removed, expected_removal)
    npt.assert_allclose(
        _inventory(particles) + gas.concentration,
        expected_scale[:, None] * (particle_before + gas_before),
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_ATOL,
    )


def test_direct_invalid_p2_and_p3_preflight_preserve_all_inputs() -> None:
    """P2/P3 preflight failures preserve arrays, records, and config sidecars."""
    particles, gas, config = _fixture()
    gas.concentration[:] = 1.0
    gas.concentration[1, 1] = np.nan
    p2_inputs = (gas.concentration, gas.molar_mass, gas.partitioning)
    p2_snapshot = _snapshot(*p2_inputs)

    with pytest.raises(ValueError, match="finite"):
        finalize_particle_source(
            PotentialEventData(np.array([1.0, 1.0]), 1.0),
            InjectionComposition((1, 1)),
            gas,
        )

    _assert_unchanged(p2_inputs, p2_snapshot)
    gas.concentration[1, 1] = 1.0
    demand, diagnostics = finalize_particle_source(
        PotentialEventData(np.array([1.0, 1.0]), 1.0),
        InjectionComposition((1, 1)),
        gas,
    )
    object.__setattr__(config, "requested_scale", np.array([1.1, 1.1]))
    p3_arrays = _mutable_arrays(particles, gas)
    p3_snapshot = _snapshot(*p3_arrays)
    record_arrays = (
        demand.per_event_mass,
        demand.gas_mass_removed,
        diagnostics.potential_event_count,
        diagnostics.gas_admitted_event_count,
        diagnostics.gas_limited_event_count,
        diagnostics.limiting_species_index,
        config.requested_scale,
        config.minimum_scale,
        config.minimum_volume,
    )
    record_snapshot = _snapshot(*record_arrays)

    with pytest.raises(ValueError, match="invalid representative"):
        commit_particle_source(demand, diagnostics, particles, gas, config)

    _assert_unchanged(p3_arrays, p3_snapshot)
    _assert_unchanged(record_arrays, record_snapshot)
    # P3 owns diagnostics; it does not accept a caller-owned output buffer.
    assert "buffer" not in commit_particle_source.__annotations__
