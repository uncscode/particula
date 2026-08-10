"""Resident stochastic-stream invariance regressions.

The tests deliberately compare only executions on the same Warp device.  They
exercise the resident adapters so stream ownership is verified at the real
logical-ID to physical-lane boundary rather than through a direct kernel call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from particula.execution import Backend, Device
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    ResidentBrownianCoagulationExecutionAdapter,
    ResidentBrownianCoagulationExecutionState,
    WarpBrownianCoagulationExecutionState,
    WarpBrownianCoagulationState,
)
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


def _require_device(device: str) -> Any:
    """Return Warp after skipping an unavailable optional CUDA row."""
    wp = pytest.importorskip("warp")
    if device not in warp_devices(wp):
        pytest.skip(CUDA_SKIP_REASON)
    return wp


def _make_session(
    device: str,
    logical_box_ids: tuple[str, ...],
    lanes: tuple[int, ...],
    *,
    unrelated_active: bool = True,
) -> tuple[Any, GPUResourceRegistry, ResidentStepGuard]:
    """Create a mapped resident fixture with ``box-a`` at its metadata lane.

    CPU rows are populated via the manifest lane permutation before setup.  This
    makes a permuted manifest a real physical-row permutation, not a relabelled
    comparison of mismatched resident state.
    """
    wp = _require_device(device)
    boxes = len(logical_box_ids)
    masses = np.zeros((boxes, 2, 1), dtype=np.float64)
    concentration = np.zeros((boxes, 2), dtype=np.float64)
    charge = np.zeros((boxes, 2), dtype=np.float64)
    for index, logical_id in enumerate(logical_box_ids):
        lane = lanes[index]
        if logical_id == "box-a" or unrelated_active:
            masses[lane, :, 0] = (1.0e-18, 1.5e-18)
            concentration[lane] = 1.0
    particles = ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=np.array([1000.0], dtype=np.float64),
        volume=np.ones(boxes, dtype=np.float64),
    )
    gas = GasData(
        name=["species"],
        molar_mass=np.array([0.1], dtype=np.float64),
        concentration=np.zeros((boxes, 1), dtype=np.float64),
        partitioning=np.array([False]),
    )
    environment = EnvironmentData(
        temperature=np.full(boxes, 298.15, dtype=np.float64),
        pressure=np.full(boxes, 101325.0, dtype=np.float64),
        saturation_ratio=np.ones((boxes, 1), dtype=np.float64),
    )
    session = setup_resident_session(
        particles,
        gas,
        environment,
        Device(Backend.WARP, str(wp.get_device(device))),
        root_seed=734,
        logical_box_ids=logical_box_ids,
        lanes=lanes,
    )
    registry = GPUResourceRegistry(session)
    return session, registry, ResidentStepGuard(session, registry)


@pytest.fixture
def resident_factory() -> Callable[..., tuple[Any, GPUResourceRegistry]]:
    """Create resident sessions and close every created binding at teardown."""
    bindings: list[tuple[Any, GPUResourceRegistry, ResidentStepGuard]] = []

    def create(*args: Any, **kwargs: Any) -> tuple[Any, GPUResourceRegistry]:
        binding = _make_session(*args, **kwargs)
        bindings.append(binding)
        return binding[0], binding[1]

    yield create

    for session, registry, guard in reversed(bindings):
        session.close(registry, guard)


def _target_lane(session: Any) -> int:
    """Resolve ``box-a`` through immutable stream metadata only."""
    stream = session.metadata.stream
    return stream.lanes[stream.logical_box_ids.index("box-a")]


def _particle_snapshot(session: Any, lane: int) -> tuple[np.ndarray, ...]:
    """Copy the target particle fields after explicit device synchronization."""
    _require_device("cpu").synchronize()
    return (
        session.particles.masses.numpy()[lane].copy(),
        session.particles.concentration.numpy()[lane].copy(),
        session.particles.charge.numpy()[lane].copy(),
    )


def _all_particle_snapshot(session: Any) -> tuple[np.ndarray, ...]:
    """Copy every mutable particle field after explicit synchronization."""
    _require_device("cpu").synchronize()
    return (
        session.particles.masses.numpy().copy(),
        session.particles.concentration.numpy().copy(),
        session.particles.charge.numpy().copy(),
    )


def _rng_snapshot(resources: Any) -> np.ndarray:
    """Copy resident RNG words after explicit device synchronization."""
    _require_device("cpu").synchronize()
    return resources.rng_states.numpy().copy()


def _collision_snapshot(resources: Any) -> np.ndarray:
    """Copy collision-pair sidecar rows after explicit synchronization."""
    _require_device("cpu").synchronize()
    return resources.collision_pairs.numpy().copy()


def _collision_count_snapshot(resources: Any) -> np.ndarray:
    """Copy collision-count sidecar rows after explicit synchronization."""
    _require_device("cpu").synchronize()
    return resources.n_collisions.numpy().copy()


def _dispatch_coagulation(session: Any, registry: GPUResourceRegistry) -> Any:
    """Dispatch one real resident Brownian request with its acquired stream."""
    resources = registry.acquire_coagulation(collision_capacity=1)
    request = WarpBrownianCoagulationExecutionState(
        WarpBrownianCoagulationState(
            BrownianCoagulationConfig(),
            session.particles,
            None,
            None,
            1.0,
            collision_pairs=resources.collision_pairs,
            n_collisions=resources.n_collisions,
            rng_states=resources.rng_states,
            initialize_rng=False,
            environment=session.environment,
        )
    )
    state = ResidentBrownianCoagulationExecutionState(
        request, session, registry, resources
    )
    ResidentBrownianCoagulationExecutionAdapter().execute(state)
    return resources


def _dispatch_wall_loss(
    session: Any,
    registry: GPUResourceRegistry,
    time_step: float,
    enabled_box_indices: tuple[int, ...] | None = None,
) -> Any:
    """Dispatch neutral wall loss using the published resident RNG sidecar."""
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    resources = registry.acquire_wall_loss()
    request = ResidentWallLossRequest(
        session,
        registry,
        resources,
        NeutralWallLossConfig("spherical", 1.0, chamber_radius=1.0),
        time_step,
        enabled_box_indices=enabled_box_indices,
    )
    assert ResidentWallLossAdapter().execute(request) is session.particles
    return resources


@pytest.mark.warp
@pytest.mark.stochastic
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize(
    ("ids", "lanes", "unrelated_active"),
    [
        (("box-a",), (0,), True),
        (("box-a", "box-b"), (0, 1), True),
        (("box-a", "box-b"), (0, 1), False),
        (("box-b", "box-a"), (1, 0), True),
    ],
)
def test_brownian_stream_follows_logical_box_across_arrangements(
    device: str,
    ids: tuple[str, ...],
    lanes: tuple[int, ...],
    unrelated_active: bool,
    resident_factory: Callable[..., tuple[Any, GPUResourceRegistry]],
) -> None:
    """Test target Brownian state is invariant to unrelated resident boxes."""
    _require_device(device)
    reference, reference_registry = resident_factory(device, ("box-a",), (0,))
    candidate, candidate_registry = resident_factory(
        device, ids, lanes, unrelated_active=unrelated_active
    )
    reference_resources = _dispatch_coagulation(reference, reference_registry)
    candidate_initial_rng = _rng_snapshot(
        candidate_registry.acquire_coagulation(1)
    )
    candidate_resources = _dispatch_coagulation(candidate, candidate_registry)
    assert (
        candidate_resources.rng_states
        is candidate_registry.acquire_coagulation(1).rng_states
    )

    reference_lane = _target_lane(reference)
    candidate_lane = _target_lane(candidate)
    for actual, expected in zip(
        _particle_snapshot(candidate, candidate_lane),
        _particle_snapshot(reference, reference_lane),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        _collision_snapshot(candidate_resources)[candidate_lane],
        _collision_snapshot(reference_resources)[reference_lane],
    )
    assert (
        _collision_count_snapshot(candidate_resources)[candidate_lane]
        == _collision_count_snapshot(reference_resources)[reference_lane]
    )
    assert (
        _rng_snapshot(candidate_resources)[candidate_lane]
        == _rng_snapshot(reference_resources)[reference_lane]
    )
    if not unrelated_active and len(ids) > 1:
        other_lane = 1 - candidate_lane
        assert (
            _rng_snapshot(candidate_resources)[other_lane]
            == candidate_initial_rng[other_lane]
        )


@pytest.mark.warp
@pytest.mark.stochastic
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize(
    ("ids", "lanes", "unrelated_active"),
    [
        (("box-a",), (0,), True),
        (("box-a", "box-b"), (0, 1), True),
        (("box-a", "box-b"), (0, 1), False),
        (("box-b", "box-a"), (1, 0), True),
    ],
)
def test_wall_loss_selected_logical_lane_is_stream_invariant(
    device: str,
    ids: tuple[str, ...],
    lanes: tuple[int, ...],
    unrelated_active: bool,
    resident_factory: Callable[..., tuple[Any, GPUResourceRegistry]],
) -> None:
    """Test selected target wall loss ignores active, skipped, and permuted rows."""
    _require_device(device)
    reference, reference_registry = resident_factory(device, ("box-a",), (0,))
    candidate, candidate_registry = resident_factory(
        device, ids, lanes, unrelated_active=unrelated_active
    )
    reference_resources = _dispatch_wall_loss(
        reference, reference_registry, 1.0
    )
    candidate_lane = _target_lane(candidate)
    candidate_initial_rng = _rng_snapshot(
        candidate_registry.acquire_wall_loss()
    )
    candidate_resources = _dispatch_wall_loss(
        candidate, candidate_registry, 1.0, (candidate_lane,)
    )
    assert (
        candidate_resources.rng_states
        is candidate_registry.acquire_wall_loss().rng_states
    )

    for actual, expected in zip(
        _particle_snapshot(candidate, candidate_lane),
        _particle_snapshot(reference, _target_lane(reference)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert (
        _rng_snapshot(candidate_resources)[candidate_lane]
        == _rng_snapshot(reference_resources)[_target_lane(reference)]
    )
    if not unrelated_active and len(ids) > 1:
        other_lane = next(
            lane for lane in range(len(ids)) if lane != candidate_lane
        )
        assert (
            _rng_snapshot(candidate_resources)[other_lane]
            == candidate_initial_rng[other_lane]
        )


@pytest.mark.warp
@pytest.mark.stochastic
def test_wall_loss_noop_and_rejection_preserve_every_stream_word(
    resident_factory: Callable[..., tuple[Any, GPUResourceRegistry]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test zero time, empty selection, and reset rejection are prelaunch safe."""
    _require_device("cpu")
    session, registry = resident_factory("cpu", ("box-a", "box-b"), (1, 0))
    resources = registry.acquire_wall_loss()
    before_rng = _rng_snapshot(resources)
    before_particles = _all_particle_snapshot(session)
    _dispatch_wall_loss(session, registry, 0.0, (_target_lane(session),))
    np.testing.assert_array_equal(_rng_snapshot(resources), before_rng)
    for actual, expected in zip(
        _all_particle_snapshot(session), before_particles, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    import particula.execution.process_adapters as adapters

    monkeypatch.setattr(
        adapters,
        "_get_wall_loss_selected_boxes_step_gpu",
        lambda: pytest.fail("empty selection must not resolve a kernel"),
    )
    _dispatch_wall_loss(session, registry, 1.0, ())
    np.testing.assert_array_equal(_rng_snapshot(resources), before_rng)
    for actual, expected in zip(
        _all_particle_snapshot(session), before_particles, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    monkeypatch.setattr(
        adapters,
        "_get_wall_loss_step_gpu",
        lambda: pytest.fail("rejected request must not resolve a kernel"),
    )
    monkeypatch.setattr(
        adapters,
        "_get_wall_loss_selected_boxes_step_gpu",
        lambda: pytest.fail("rejected request must not resolve a kernel"),
    )
    with pytest.raises(
        ValueError, match="resident wall_loss initialize_rng must be False."
    ):
        ResidentWallLossRequest(
            session,
            registry,
            resources,
            object(),
            1.0,
            initialize_rng=True,
        )
    np.testing.assert_array_equal(_rng_snapshot(resources), before_rng)
    for actual, expected in zip(
        _all_particle_snapshot(session), before_particles, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.warp
def test_resident_processes_publish_distinct_rng_sidecars(
    resident_factory: Callable[..., tuple[Any, GPUResourceRegistry]],
) -> None:
    """Test process namespaces retain distinct registry-owned RNG arrays."""
    _require_device("cpu")
    session, registry = resident_factory("cpu", ("box-a",), (0,))
    assert (
        registry.acquire_coagulation(1).rng_states
        is not registry.acquire_wall_loss().rng_states
    )
    assert session.metadata.stream is not None
