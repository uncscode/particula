"""Multi-box resident lifecycle, logical-ID, and wall-loss regressions.

The rows deliberately use complete CPU carriers and the real resident resource
boundary.  Snapshots synchronize only at assertion boundaries; no test retains
a resident binding after its case completes.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import Backend, Device
from particula.execution.gpu_resources import GPUResourceRegistry
from particula.execution.gpu_session import (
    ResidentStepGuard,
    setup_resident_session,
)
from particula.execution.process_adapters import (
    ResidentWallLossAdapter,
    ResidentWallLossRequest,
)
from particula.gas import EnvironmentData, GasData
from particula.gpu.tests.cuda_availability import CUDA_SKIP_REASON, warp_devices
from particula.particles import ParticleData

PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-30
INVENTORY_RTOL = 1e-12
INVENTORY_ATOL = 1e-30


def _require_device(device: str) -> Any:
    """Lazily require Warp and skip an unavailable optional CUDA device."""
    wp = pytest.importorskip("warp")
    if device not in warp_devices(wp):
        pytest.skip(CUDA_SKIP_REASON)
    return wp


def _cpu_carriers(
    manifest: tuple[tuple[str, int], ...],
) -> tuple[ParticleData, GasData, EnvironmentData]:
    """Build independent 16-slot, two-species rows at manifest lanes."""
    n_boxes = len(manifest)
    masses = np.zeros((n_boxes, 16, 2), dtype=np.float64)
    concentration = np.zeros((n_boxes, 16), dtype=np.float64)
    charge = np.zeros((n_boxes, 16), dtype=np.float64)
    gas_concentration = np.zeros((n_boxes, 2), dtype=np.float64)
    temperature = np.zeros(n_boxes, dtype=np.float64)
    pressure = np.zeros(n_boxes, dtype=np.float64)
    volume = np.zeros(n_boxes, dtype=np.float64)
    for logical_id, lane in manifest:
        # ``box-3`` and ``free`` are valid no-work rows in every arrangement.
        active_slots = (
            0
            if logical_id in {"box-3", "free"}
            else 1 + (sum(ord(character) for character in logical_id) % 2)
        )
        value = float(sum(ord(character) for character in logical_id))
        masses[lane, :active_slots] = np.array(
            [1.0e-20 * value, 2.0e-20 * value],
            dtype=np.float64,
        )
        concentration[lane, :active_slots] = 1.0 + value / 1000.0
        gas_concentration[lane] = np.array(
            [1.0e-11 * value, 2.0e-11 * value],
            dtype=np.float64,
        )
        temperature[lane] = 295.0 + value / 100.0
        pressure[lane] = 101325.0 - value
        volume[lane] = 1.0e-8 * value
    particles = ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=np.array([1000.0, 1200.0], dtype=np.float64),
        volume=volume,
    )
    gas = GasData(
        name=["species-a", "species-b"],
        molar_mass=np.array([0.018, 0.098], dtype=np.float64),
        concentration=gas_concentration,
        partitioning=np.array([False, False]),
    )
    environment = EnvironmentData(
        temperature=temperature,
        pressure=pressure,
        saturation_ratio=np.ones((n_boxes, 2), dtype=np.float64),
    )
    assert particles.masses.shape == (n_boxes, 16, 2)
    assert particles.concentration.shape == (n_boxes, 16)
    assert gas.concentration.shape == (n_boxes, 2)
    return particles, gas, environment


def _binding(
    device: str,
    manifest: tuple[tuple[str, int], ...],
    root_seed: int = 1531,
) -> tuple[Any, GPUResourceRegistry, ResidentStepGuard]:
    """Upload one manifest fixture and bind its exact registry and guard."""
    wp = _require_device(device)
    particles, gas, environment = _cpu_carriers(manifest)
    ids, lanes = zip(*manifest, strict=True)
    session = setup_resident_session(
        particles,
        gas,
        environment,
        Device(Backend.WARP, str(wp.get_device(device))),
        root_seed=root_seed,
        logical_box_ids=ids,
        lanes=lanes,
    )
    registry = GPUResourceRegistry(session)
    return session, registry, ResidentStepGuard(session, registry)


@pytest.fixture
def resident_factory() -> Generator[Any, None, None]:
    """Close each created binding in reverse order, including failed cases."""
    bindings: list[tuple[Any, GPUResourceRegistry, ResidentStepGuard]] = []

    def create(*args: Any, **kwargs: Any) -> tuple[Any, GPUResourceRegistry]:
        session, registry, guard = _binding(*args, **kwargs)
        bindings.append((session, registry, guard))
        return session, registry

    yield create
    for session, registry, guard in reversed(bindings):
        session.close(registry, guard)


def _lane(session: Any, logical_id: str) -> int:
    """Resolve a physical row through immutable logical-ID stream metadata."""
    stream = session.metadata.stream
    return stream.lanes[stream.logical_box_ids.index(logical_id)]


def _snapshot(session: Any) -> tuple[np.ndarray, ...]:
    """Synchronize once and copy mutable resident state for comparison."""
    wp = pytest.importorskip("warp")
    wp.synchronize_device(session.particles.masses.device)
    return (
        session.particles.masses.numpy().copy(),
        session.particles.concentration.numpy().copy(),
        session.particles.charge.numpy().copy(),
        session.gas.concentration.numpy().copy(),
        session.gas.vapor_pressure.numpy().copy(),
        session.environment.temperature.numpy().copy(),
        session.environment.pressure.numpy().copy(),
        session.environment.saturation_ratio.numpy().copy(),
    )


def _inventory(session: Any) -> np.ndarray:
    """Return independent concentration-weighted particle-plus-gas inventory."""
    state = _snapshot(session)
    masses, concentration, _, gas, *_ = state
    volume = session.particles.volume.numpy()
    return volume[:, None] * (
        np.sum(masses * concentration[:, :, None], axis=1) + gas
    )


def _wall_loss(
    session: Any,
    registry: GPUResourceRegistry,
    duration: float,
    selected: tuple[int, ...] | None = None,
) -> Any:
    """Execute real neutral wall loss with the registry-owned RNG sidecar."""
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    resources = registry.acquire_wall_loss()
    request = ResidentWallLossRequest(
        session,
        registry,
        resources,
        NeutralWallLossConfig("spherical", 1.0, chamber_radius=1.0),
        duration,
        enabled_box_indices=selected,
    )
    assert ResidentWallLossAdapter().execute(request) is session.particles
    return resources


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_multi_box_zero_duration_matches_independent_sessions_and_conserves_inventory(
    resident_factory: Any,
) -> None:
    """Zero-duration selected dispatch preserves four logical rows and inventory."""
    manifest = tuple((f"box-{index}", index) for index in range(4))
    multi, multi_registry = resident_factory("cpu", manifest)
    initial_inventory = _inventory(multi)
    before = _snapshot(multi)
    resources = _wall_loss(multi, multi_registry, 0.0, (0, 1, 2, 3))
    assert resources.rng_states is multi_registry.acquire_wall_loss().rng_states
    after = _snapshot(multi)
    for actual, expected in zip(after, before, strict=True):
        npt.assert_allclose(
            actual, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL
        )
    npt.assert_allclose(
        _inventory(multi),
        initial_inventory,
        rtol=INVENTORY_RTOL,
        atol=INVENTORY_ATOL,
    )
    for logical_id, _ in manifest:
        single, single_registry = resident_factory("cpu", ((logical_id, 0),))
        _wall_loss(single, single_registry, 0.0, (0,))
        lane = _lane(multi, logical_id)
        for actual, expected in zip(
            _snapshot(multi), _snapshot(single), strict=True
        ):
            npt.assert_allclose(
                actual[lane], expected[0], rtol=PARITY_RTOL, atol=PARITY_ATOL
            )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_multi_box_logical_id_permutation_addition_and_wall_loss_selection_are_isolated(
    resident_factory: Any,
) -> None:
    """Logical rows survive permutation, unrelated addition, and empty selection."""
    reference, reference_registry = resident_factory(
        "cpu", (("box-a", 0), ("box-b", 1), ("box-c", 2), ("box-d", 3))
    )
    candidate, candidate_registry = resident_factory(
        "cpu",
        (("box-c", 2), ("extra", 4), ("box-a", 0), ("box-d", 3), ("box-b", 1)),
    )
    _wall_loss(reference, reference_registry, 0.0, ())
    before = _snapshot(candidate)
    rng = candidate_registry.acquire_wall_loss().rng_states.numpy().copy()
    _wall_loss(candidate, candidate_registry, 0.0, ())
    npt.assert_equal(_snapshot(candidate), before)
    npt.assert_equal(
        candidate_registry.acquire_wall_loss().rng_states.numpy(), rng
    )
    for logical_id in ("box-a", "box-b", "box-c", "box-d"):
        for actual, expected in zip(
            _snapshot(candidate), _snapshot(reference), strict=True
        ):
            npt.assert_allclose(
                actual[_lane(candidate, logical_id)],
                expected[_lane(reference, logical_id)],
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )


@pytest.mark.warp
@pytest.mark.stochastic
def test_resident_scheduler_streams_continue_per_logical_box_without_no_work_consumption(
    resident_factory: Any,
) -> None:
    """Selected active rows advance their wall-loss stream; free rows do not."""
    session, registry = resident_factory(
        "cpu", (("active", 0), ("free", 1), ("other", 2), ("one", 3))
    )
    resources = registry.acquire_wall_loss()
    before = resources.rng_states.numpy().copy()
    _wall_loss(session, registry, 1.0, (0, 1))
    after = resources.rng_states.numpy().copy()
    assert after[0] != before[0]
    assert after[1] == before[1]
    assert resources.rng_states is registry.acquire_wall_loss().rng_states


@pytest.mark.warp
@pytest.mark.stochastic
def test_resident_wall_loss_removal_matches_cpu_binomial_aggregate(
    resident_factory: Any,
) -> None:
    """Fresh Warp CPU streams produce finite bounded aggregate removal evidence."""
    from particula.dynamics.properties.wall_loss_coefficient import (
        get_spherical_wall_loss_coefficient_via_system_state,
    )

    time_step = 1.0
    density = np.array([1000.0, 1200.0], dtype=np.float64)
    masses = np.array([3.29e-18, 6.58e-18], dtype=np.float64)
    particle_volume = np.sum(masses / density)
    particle_radius = (3.0 * particle_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    coefficient = get_spherical_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=1.0,
        particle_radius=particle_radius,
        particle_density=float(np.sum(masses) / particle_volume),
        temperature=298.29,
        pressure=100996.0,
        chamber_radius=1.0,
    )
    removal_probability = 1.0 - np.exp(-float(coefficient) * time_step)
    removed = 0
    trials = 0
    for seed in range(100):
        session, registry = resident_factory(
            "cpu", (("box", 0),), root_seed=seed
        )
        before = _snapshot(session)[1]
        _wall_loss(session, registry, time_step, (0,))
        after = _snapshot(session)[1]
        removed += int(np.count_nonzero((before > 0.0) & (after == 0.0)))
        trials += int(np.count_nonzero(before > 0.0))
    expected = trials * removal_probability
    bound = max(
        3.0
        * np.sqrt(trials * removal_probability * (1.0 - removal_probability)),
        1.0,
    )
    assert abs(removed - expected) <= bound


@pytest.mark.warp
@pytest.mark.cuda
def test_resident_wall_loss_cuda_smoke_has_finite_bounded_removal(
    resident_factory: Any,
) -> None:
    """Optional CUDA lifecycle smoke keeps removal counts finite and bounded."""
    removed = 0
    trials = 0
    for seed in range(12):
        session, registry = resident_factory(
            "cuda", (("box", 0),), root_seed=seed
        )
        before = _snapshot(session)[1]
        _wall_loss(session, registry, 1.0, (0,))
        after = _snapshot(session)[1]
        assert np.all(np.isfinite(after))
        removed += int(np.count_nonzero((before > 0.0) & (after == 0.0)))
        trials += int(np.count_nonzero(before > 0.0))
    assert 0 <= removed <= trials
