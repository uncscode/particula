"""Integration evidence for concrete-only Brownian coagulation adapters.

CPU and Warp stochastic trajectories are deliberately not compared seed by
seed. These tests instead check local CPU rates, caller-owned resident-resource
identity, and physical invariants after explicit caller synchronization of each
Warp dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

import particula as par
from particula.execution import ExecutionResult
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    CPUCoagulationExecutionAdapter,
    CPUCoagulationExecutionState,
    CPUCoagulationResult,
    CPUCoagulationState,
    WarpBrownianCoagulationExecutionAdapter,
    WarpBrownianCoagulationExecutionState,
    WarpBrownianCoagulationResult,
    WarpBrownianCoagulationState,
)
from particula.gas import EnvironmentData
from particula.particles import ParticleData
from particula.util.constants import (
    BOLTZMANN_CONSTANT,
    GAS_CONSTANT,
    MOLECULAR_WEIGHT_AIR,
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)


@dataclass(frozen=True)
class _Fixture:
    """Store explicit fixed-shape Brownian adapter test data."""

    masses: np.ndarray
    concentration: np.ndarray
    charge: np.ndarray
    density: np.ndarray
    volume: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    active: tuple[tuple[int, ...], ...]


def _fixture(boxes: int = 1, active_count: int = 3) -> _Fixture:
    """Build a small fp64 fixture with a protected inactive sentinel."""
    slots = 4
    masses = np.zeros((boxes, slots, 1), dtype=np.float64)
    concentration = np.zeros((boxes, slots), dtype=np.float64)
    charge = np.zeros((boxes, slots), dtype=np.float64)
    for box in range(boxes):
        for slot in range(active_count):
            masses[box, slot, 0] = (slot + 1) * 1.0e-18
            concentration[box, slot] = 1.0
            charge[box, slot] = float(slot - 1)
        # This deliberately invalid-as-active slot must remain untouched.
        masses[box, -1, 0] = 7.0e-31
        charge[box, -1] = 29.0
    return _Fixture(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=np.array([1000.0], dtype=np.float64),
        volume=np.full(boxes, 1.0e-6, dtype=np.float64),
        temperature=np.linspace(298.15, 303.15, boxes, dtype=np.float64),
        pressure=np.full(boxes, 101325.0, dtype=np.float64),
        active=tuple(tuple(range(active_count)) for _ in range(boxes)),
    )


def _inventory(fixture: _Fixture) -> tuple[np.ndarray, np.ndarray]:
    """Return concentration-weighted mass and active signed charge budgets."""
    mass = np.sum(fixture.masses * fixture.concentration[..., None], axis=1)
    charge = np.sum(fixture.charge * (fixture.concentration > 0.0), axis=1)
    return mass, charge


def _make_cpu_aerosol() -> tuple[Any, Any]:
    """Build a real particle-resolved CPU aerosol and Brownian runnable."""
    mass = np.array([1.0e-18, 2.0e-18, 3.0e-18], dtype=np.float64)
    particles = (
        par.particles.ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(
            par.particles.ParticleResolvedSpeciatedMass()
        )
        .set_activity_strategy(par.particles.ActivityIdealMass())
        .set_surface_strategy(par.particles.SurfaceStrategyVolume())
        .set_mass(mass, "kg")
        .set_density(np.array([1000.0]), "kg/m^3")
        .set_charge(np.array([0.0, 1.0, -1.0]))
        .set_volume(1.0e-6, "m^3")
        .build()
    )
    atmosphere = (
        par.gas.AtmosphereBuilder()
        .set_temperature(298.15, temperature_units="K")
        .set_pressure(101325.0, pressure_units="Pa")
        .build()
    )
    aerosol = par.Aerosol(atmosphere=atmosphere, particles=particles)
    runnable = par.dynamics.Coagulation(
        par.dynamics.BrownianCoagulationStrategy("particle_resolved")
    )
    return aerosol, runnable


def _local_brownian_reference(
    radii: np.ndarray, masses: np.ndarray, temperature: float, pressure: float
) -> np.ndarray:
    """Calculate the Fuchs Brownian pair kernel from local equations.

    This intentionally duplicates the small fixed-fixture reference rather than
    calling production Brownian helpers, so it remains independent evidence.
    """
    viscosity = (
        REF_VISCOSITY_AIR_STP
        * (temperature / REF_TEMPERATURE_STP) ** 1.5
        * (REF_TEMPERATURE_STP + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )
    air_mean_free_path = (2.0 * viscosity / pressure) / np.sqrt(
        8.0 * MOLECULAR_WEIGHT_AIR / (np.pi * GAS_CONSTANT * temperature)
    )
    knudsen_number = air_mean_free_path / radii
    slip_correction = 1.0 + knudsen_number * (
        1.257 + 0.4 * np.exp(-1.1 / knudsen_number)
    )
    mobility = slip_correction / (6.0 * np.pi * viscosity * radii)
    diffusivity = BOLTZMANN_CONSTANT * temperature * mobility
    thermal_speed = np.sqrt(
        8.0 * BOLTZMANN_CONSTANT * temperature / (np.pi * masses)
    )
    particle_mean_free_path = 8.0 * diffusivity / (np.pi * thermal_speed)
    collection_term = (
        (2.0 * radii + particle_mean_free_path) ** 3
        - (4.0 * radii**2 + particle_mean_free_path**2) ** 1.5
    ) / (6.0 * radii * particle_mean_free_path) - 2.0 * radii

    sum_diffusivity = diffusivity[:, None] + diffusivity[None, :]
    sum_radius = radii[:, None] + radii[None, :]
    sum_collection_term = np.hypot(
        collection_term[:, None], collection_term[None, :]
    )
    sum_thermal_speed = np.hypot(thermal_speed[:, None], thermal_speed[None, :])
    return (4.0 * np.pi * sum_diffusivity * sum_radius) / (
        sum_radius / (sum_radius + sum_collection_term)
        + 4.0 * sum_diffusivity / (sum_radius * sum_thermal_speed)
    )


def _warp_bundle(fixture: _Fixture, device: str = "cpu") -> Any:
    """Convert detached CPU data once and allocate caller-owned sidecars."""
    wp = pytest.importorskip("warp")
    from particula.gpu import to_warp_environment_data, to_warp_particle_data

    particle_data = ParticleData(
        fixture.masses.copy(),
        fixture.concentration.copy(),
        fixture.charge.copy(),
        fixture.density.copy(),
        fixture.volume.copy(),
    )
    environment_data = EnvironmentData(
        fixture.temperature.copy(),
        fixture.pressure.copy(),
        np.ones((fixture.masses.shape[0], 1), dtype=np.float64),
    )
    boxes, slots, _ = fixture.masses.shape
    return type(
        "Bundle",
        (),
        {
            "particles": to_warp_particle_data(particle_data, device=device),
            "environment": to_warp_environment_data(
                environment_data, device=device
            ),
            "pairs": wp.full(
                (boxes, max(1, slots // 2), 2),
                -1,
                dtype=wp.int32,
                device=device,
            ),
            "counts": wp.zeros(boxes, dtype=wp.int32, device=device),
            "rng": wp.zeros(boxes, dtype=wp.uint32, device=device),
        },
    )()


def _dispatch(
    bundle: Any,
    *,
    time_step: float,
    initialize_rng: bool,
    rng_seed: int = 41,
) -> Any:
    """Dispatch one selected resident call without adapter-side synchronization.

    The caller retains every resident resource and must explicitly synchronize
    before observing device state.
    """
    state = WarpBrownianCoagulationState(
        BrownianCoagulationConfig(),
        bundle.particles,
        None,
        None,
        time_step,
        collision_pairs=bundle.pairs,
        n_collisions=bundle.counts,
        rng_states=bundle.rng,
        rng_seed=rng_seed,
        initialize_rng=initialize_rng,
        environment=bundle.environment,
    )
    return WarpBrownianCoagulationExecutionAdapter().execute(
        WarpBrownianCoagulationExecutionState(state)
    )


def _assert_warp_invariants(bundle: Any, before: _Fixture) -> None:
    """Assert conservation, valid pairs, and inactive-sentinel protection."""
    masses = bundle.particles.masses.numpy()
    concentration = bundle.particles.concentration.numpy()
    charge = bundle.particles.charge.numpy()
    pairs = bundle.pairs.numpy()
    counts = bundle.counts.numpy()
    after = _Fixture(
        masses,
        concentration,
        charge,
        before.density,
        before.volume,
        before.temperature,
        before.pressure,
        before.active,
    )
    initial_mass, initial_charge = _inventory(before)
    final_mass, final_charge = _inventory(after)
    npt.assert_allclose(final_mass, initial_mass, rtol=1e-12, atol=1e-30)
    npt.assert_allclose(final_charge, initial_charge, rtol=1e-12, atol=1e-30)
    assert np.all(np.isfinite(masses)) and np.all(masses >= 0.0)
    assert np.all(np.isfinite(concentration)) and np.all(concentration >= 0.0)
    for box, active in enumerate(before.active):
        npt.assert_array_equal(masses[box, -1], before.masses[box, -1])
        npt.assert_array_equal(charge[box, -1], before.charge[box, -1])
        assert concentration[box, -1] == 0.0
        used: set[int] = set()
        for pair in pairs[box, : counts[box]]:
            first, second = map(int, pair)
            assert first < second and first in active and second in active
            assert first not in used and second not in used
            used.update((first, second))


def _assert_warp_result_identities(
    result: ExecutionResult, bundle: Any
) -> None:
    """Assert a resident dispatch returns every supplied resource by identity."""
    assert result.backend_result is not None
    value = result.backend_result.value
    assert isinstance(value, WarpBrownianCoagulationResult)
    assert value.particles is bundle.particles
    assert value.collision_pairs is bundle.pairs
    assert value.n_collisions is bundle.counts


def test_cpu_adapter_matches_local_brownian_kernel_reference_and_preserves_identity() -> (
    None
):
    """The real CPU strategy rate matches a local reference before dispatch."""
    aerosol, runnable = _make_cpu_aerosol()
    strategy = runnable.coagulation_strategy
    radii = aerosol.particles.get_radius()
    masses = aerosol.particles.get_mass()
    observed = strategy.kernel(aerosol.particles, 298.15, 101325.0)
    expected = _local_brownian_reference(radii, masses, 298.15, 101325.0)
    npt.assert_allclose(observed, expected, rtol=1e-7, atol=0.0)
    request = CPUCoagulationExecutionState(
        CPUCoagulationState(BrownianCoagulationConfig(), aerosol),
        0.0,
        1,
        runnable,
    )
    result = CPUCoagulationExecutionAdapter().execute(request)
    assert isinstance(result, ExecutionResult)
    assert result.backend_result is not None
    assert isinstance(result.backend_result.value, CPUCoagulationResult)
    assert result.backend_result.value.aerosol is aerosol


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("boxes", (1, 2))
def test_warp_adapter_preserves_resident_resources_and_conservation(
    boxes: int,
) -> None:
    """Resident Warp calls retain identities and conserve each box budget."""
    wp = pytest.importorskip("warp")
    fixture = _fixture(boxes)
    bundle = _warp_bundle(fixture)
    result = _dispatch(bundle, time_step=1.0e10, initialize_rng=True)
    wp.synchronize()
    _assert_warp_result_identities(result, bundle)
    _assert_warp_invariants(bundle, fixture)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_warp_adapter_keeps_boxes_isolated() -> None:
    """Changing one box cannot affect another resident box's result."""
    wp = pytest.importorskip("warp")
    fixture = _fixture(boxes=2)
    altered = _Fixture(
        masses=fixture.masses.copy(),
        concentration=fixture.concentration.copy(),
        charge=fixture.charge.copy(),
        density=fixture.density.copy(),
        volume=fixture.volume.copy(),
        temperature=fixture.temperature.copy(),
        pressure=fixture.pressure.copy(),
        active=fixture.active,
    )
    altered.masses[0, :3, 0] *= 10.0
    baseline_bundle = _warp_bundle(fixture)
    altered_bundle = _warp_bundle(altered)
    baseline_result = _dispatch(
        baseline_bundle, time_step=1.0e10, initialize_rng=True
    )
    wp.synchronize()
    _assert_warp_result_identities(baseline_result, baseline_bundle)
    altered_result = _dispatch(
        altered_bundle, time_step=1.0e10, initialize_rng=True
    )
    wp.synchronize()
    _assert_warp_result_identities(altered_result, altered_bundle)
    for baseline, changed in zip(
        (
            baseline_bundle.particles.masses.numpy()[1],
            baseline_bundle.particles.concentration.numpy()[1],
            baseline_bundle.particles.charge.numpy()[1],
            baseline_bundle.pairs.numpy()[1],
            baseline_bundle.counts.numpy()[1],
            baseline_bundle.rng.numpy()[1],
        ),
        (
            altered_bundle.particles.masses.numpy()[1],
            altered_bundle.particles.concentration.numpy()[1],
            altered_bundle.particles.charge.numpy()[1],
            altered_bundle.pairs.numpy()[1],
            altered_bundle.counts.numpy()[1],
            altered_bundle.rng.numpy()[1],
        ),
        strict=True,
    ):
        npt.assert_array_equal(baseline, changed)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("active_count", (0, 1))
def test_warp_adapter_zero_or_one_active_slots_are_exact_no_merge_cases(
    active_count: int,
) -> None:
    """Sparse resident slots are write-free no-merge cases."""
    wp = pytest.importorskip("warp")
    fixture = _fixture(active_count=active_count)
    bundle = _warp_bundle(fixture)
    result = _dispatch(bundle, time_step=1.0, initialize_rng=True)
    wp.synchronize()
    _assert_warp_result_identities(result, bundle)
    npt.assert_array_equal(bundle.counts.numpy(), [0])
    npt.assert_array_equal(
        bundle.pairs.numpy(), -np.ones_like(bundle.pairs.numpy())
    )
    _assert_warp_invariants(bundle, fixture)


@pytest.mark.warp
@pytest.mark.stochastic
def test_warp_brownian_acceptance_matches_fixed_trial_three_sigma_bound() -> (
    None
):
    """One hundred fresh two-active trials meet a bounded aggregate check."""
    wp = pytest.importorskip("warp")
    trials = 100
    observed = 0
    reference = _fixture(active_count=2)
    radii = np.cbrt(
        3.0 * reference.masses[0, :2, 0] / (4.0 * np.pi * reference.density[0])
    )
    kernel = _local_brownian_reference(
        radii,
        reference.masses[0, :2, 0],
        float(reference.temperature[0]),
        float(reference.pressure[0]),
    )[0, 1]
    time_step = 0.5 * reference.volume[0] / kernel
    for seed in range(trials):
        fixture = _fixture(active_count=2)
        bundle = _warp_bundle(fixture)
        _dispatch(
            bundle,
            time_step=time_step,
            initialize_rng=True,
            rng_seed=41 + seed,
        )
        wp.synchronize()
        _assert_warp_invariants(bundle, fixture)
        observed += int(bundle.counts.numpy()[0])
    expected = kernel * time_step * trials / reference.volume[0]
    assert abs(observed - expected) <= 3.0 * np.sqrt(expected)


@pytest.mark.warp
def test_warp_persistent_rng_advances_across_nonterminal_dispatches() -> None:
    """A supplied stream advances across two calls with eligible slots left."""
    wp = pytest.importorskip("warp")
    fixture = _fixture(active_count=3)
    bundle = _warp_bundle(fixture)
    first_result = _dispatch(bundle, time_step=1.0, initialize_rng=True)
    wp.synchronize()
    _assert_warp_result_identities(first_result, bundle)
    first = bundle.rng.numpy().copy()
    assert np.count_nonzero(bundle.particles.concentration.numpy()[0]) >= 2
    second_result = _dispatch(bundle, time_step=1.0, initialize_rng=False)
    wp.synchronize()
    _assert_warp_result_identities(second_result, bundle)
    second = bundle.rng.numpy().copy()
    assert np.count_nonzero(bundle.particles.concentration.numpy()[0]) >= 2
    assert not np.array_equal(first, np.zeros_like(first))
    assert not np.array_equal(second, first)


@pytest.mark.warp
def test_warp_explicit_rng_reset_replays_first_dispatch_exactly() -> None:
    """Resetting one supplied stream replays a pristine equivalent dispatch."""
    wp = pytest.importorskip("warp")
    fixture = _fixture(active_count=3)
    first = _warp_bundle(fixture)
    _dispatch(first, time_step=1.0e10, initialize_rng=True)
    wp.synchronize()
    expected = tuple(
        value.numpy().copy()
        for value in (
            first.particles.masses,
            first.particles.concentration,
            first.particles.charge,
            first.pairs,
            first.counts,
            first.rng,
        )
    )
    second = _warp_bundle(fixture)
    second.rng = first.rng
    _dispatch(second, time_step=1.0e10, initialize_rng=True)
    wp.synchronize()
    actual = tuple(
        value.numpy()
        for value in (
            second.particles.masses,
            second.particles.concentration,
            second.particles.charge,
            second.pairs,
            second.counts,
            second.rng,
        )
    )
    for observed, reference in zip(actual, expected, strict=True):
        npt.assert_array_equal(observed, reference)


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize("boxes", (1, 2))
def test_warp_cuda_adapter_invariants_when_available(boxes: int) -> None:
    """Optional CUDA rows retain the same resident invariant-only contract."""
    wp = pytest.importorskip("warp")
    from particula.gpu.tests.cuda_availability import cuda_available

    if not cuda_available(wp):
        pytest.skip("CUDA not available")
    fixture = _fixture(boxes)
    bundle = _warp_bundle(fixture, device="cuda")
    result = _dispatch(bundle, time_step=1.0e10, initialize_rng=True)
    wp.synchronize()
    _assert_warp_result_identities(result, bundle)
    _assert_warp_invariants(bundle, fixture)
