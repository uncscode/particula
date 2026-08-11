"""Resident closed-map transport regression coverage.

The reference calculation in this module intentionally uses NumPy extensive
amounts rather than any GPU transport or dilution helper.
"""

from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import CapabilityRequirements
from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationResourceShape,
    CommunicationShapeKind,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
)
from particula.execution.gpu_resources import GPUResourceRegistry
from particula.execution.process_adapters import (
    ResidentDilutionAdapter,
    ResidentDilutionRequest,
)
from particula.execution.process_graph import (
    DependencyEdge,
    ProcessNode,
    TimestepPlan,
    resolve_timestep_plan,
)
from particula.execution.resident_communication import (
    ResidentCommunicationExecutor,
    ResidentCommunicationRequest,
)


def _barrier_nodes() -> tuple[Any, Any, Any]:
    """Resolve the two canonical transport-barrier nodes."""
    from particula.execution import process_graph

    nodes = tuple(
        ProcessNode(
            schema.node_id,
            schema.kind,
            schema.process,
            CapabilityRequirements(frozenset()),
            schema.resources,
            schema.invalidates,
        )
        for schema in process_graph._NODE_CATALOGUE
        if schema.node_id in {"communication", "volume_evolution"}
    )
    graph = resolve_timestep_plan(
        TimestepPlan(
            nodes, (DependencyEdge("communication", "volume_evolution"),)
        )
    )
    by_id = {node.node_id: node for node in graph.nodes}
    return graph, by_id["communication"], by_id["volume_evolution"]


def _binding(
    rates: np.ndarray,
    *,
    final_volumes: np.ndarray | None,
    source_boxes: np.ndarray | None = None,
    destination_boxes: np.ndarray | None = None,
    enabled: np.ndarray | None = None,
    map_form: CommunicationMapForm = CommunicationMapForm.ONE_DIMENSIONAL,
) -> tuple[Any, Any, Any, Any, Any]:
    """Create one resident GAS-map barrier with explicit float64 state."""
    wp = pytest.importorskip("warp")
    from particula.execution.tests.gpu_resources_test import _session

    session = _session(boxes=3, particle_count=1, species=1)
    particles = cast(Any, session.particles)
    gas = cast(Any, session.gas)
    particles.volume.assign(np.array([1.0, 2.0, 4.0], dtype=np.float64))
    particles.concentration.assign(np.array([[3.0], [2.0], [1.0]]))
    gas.concentration.assign(np.array([[4.0], [3.0], [2.0]]))
    registry = GPUResourceRegistry(session)
    source_boxes = (
        np.array([0, 1], dtype=np.int32)
        if source_boxes is None
        else source_boxes
    )
    destination_boxes = (
        np.array([1, 2], dtype=np.int32)
        if destination_boxes is None
        else destination_boxes
    )
    enabled = (
        np.ones(len(rates), dtype=np.int32) if enabled is None else enabled
    )
    map_data = CommunicationMap(
        map_form,
        CommunicationTransportMode.GAS,
        len(rates),
        wp.array(source_boxes, dtype=wp.int32, device="cpu"),
        wp.array(destination_boxes, dtype=wp.int32, device="cpu"),
        wp.array(enabled, dtype=wp.int32, device="cpu"),
        wp.array(rates, dtype=wp.float64, device="cpu"),
    )
    final = (
        None
        if final_volumes is None
        else wp.array(final_volumes, dtype=wp.float64, device="cpu")
    )
    resources = registry.acquire_communication(
        CommunicationConfiguration(
            map_data,
            PrescribedVolumeUpdate(final),
            (
                CommunicationResourceShape(
                    "edge_rates", wp.float64, CommunicationShapeKind.E
                ),
            ),
        )
    )
    graph, communication, volume = _barrier_nodes()
    request = ResidentCommunicationRequest(
        session, registry, graph, resources, communication, volume, 1.0
    )
    return wp, session, registry, resources, request


def _transport_oracle(
    concentration: np.ndarray,
    particle_concentration: np.ndarray,
    volume: np.ndarray,
    source_boxes: np.ndarray,
    destination_boxes: np.ndarray,
    rates: np.ndarray,
    enabled: np.ndarray,
    dilution: float,
    final_volumes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance independent closed-map amount, dilution, and volume updates."""
    amounts = concentration * volume[:, None]
    deltas = np.zeros_like(amounts)
    for source, destination, rate, is_enabled in zip(
        source_boxes, destination_boxes, rates, enabled, strict=True
    ):
        if not is_enabled:
            continue
        transfer = amounts[source] * rate
        deltas[source] -= transfer
        deltas[destination] += transfer
    factor = np.exp(-dilution)
    gas = (amounts + deltas) * factor / volume[:, None]
    particles = particle_concentration * factor
    if final_volumes is None:
        return gas, particles, volume
    ratio = volume / final_volumes
    return gas * ratio[:, None], particles * ratio[:, None], final_volumes


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("rates", "final_volumes", "dilution"),
    [
        (np.array([0.25, 0.0]), np.array([2.0, 4.0, 8.0]), 0.1),
        (np.array([0.2, 0.3]), None, 0.05),
    ],
    ids=("directed-expansion", "reciprocal-mixing-fixed-volume"),
)
def test_closed_transport_loop_matches_extensive_amount_oracle(
    rates: np.ndarray,
    final_volumes: np.ndarray | None,
    dilution: float,
) -> None:
    """Two real barrier/dilution steps agree with the NumPy amount oracle."""
    wp, session, registry, resources, request = _binding(
        rates, final_volumes=final_volumes
    )
    particles = cast(Any, session.particles)
    gas = cast(Any, session.gas)
    expected_gas = gas.concentration.numpy().copy()
    expected_particles = particles.concentration.numpy().copy()
    expected_volume = particles.volume.numpy().copy()
    identities = (session, registry, resources, resources.buffers)
    executor = ResidentCommunicationExecutor(request)
    dilution_request = ResidentDilutionRequest(session, registry, dilution, 1.0)

    for _ in range(2):
        expected_gas, expected_particles, expected_volume = _transport_oracle(
            expected_gas,
            expected_particles,
            expected_volume,
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            rates,
            np.ones(len(rates), dtype=np.int32),
            dilution,
            final_volumes,
        )
        executor.execute_communication()
        executor.execute_volume_evolution()
        ResidentDilutionAdapter().execute(dilution_request)

    wp.synchronize()
    npt.assert_allclose(
        gas.concentration.numpy(), expected_gas, rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(
        particles.concentration.numpy(),
        expected_particles,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        particles.volume.numpy(), expected_volume, rtol=1e-12, atol=1e-30
    )
    assert identities == (session, registry, resources, resources.buffers)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_sparse_closed_map_matches_oracle_and_conserves_amount() -> None:
    """Disconnected arbitrary pairs use immutable extensive source amounts."""
    source = np.array([0, 2], dtype=np.int32)
    destination = np.array([2, 0], dtype=np.int32)
    rates = np.array([0.25, 0.125], dtype=np.float64)
    enabled = np.ones(2, dtype=np.int32)
    wp, session, registry, resources, request = _binding(
        rates,
        final_volumes=None,
        source_boxes=source,
        destination_boxes=destination,
        enabled=enabled,
        map_form=CommunicationMapForm.ARBITRARY_PAIRS,
    )
    particles = cast(Any, session.particles)
    gas = cast(Any, session.gas)
    before_concentration = gas.concentration.numpy().copy()
    before_volume = particles.volume.numpy().copy()
    expected, _, _ = _transport_oracle(
        before_concentration,
        particles.concentration.numpy(),
        before_volume,
        source,
        destination,
        rates,
        enabled,
        0.0,
        None,
    )
    total_before = np.sum(before_concentration * before_volume[:, None], axis=0)

    ResidentCommunicationExecutor(request).execute_communication()

    wp.synchronize()
    npt.assert_allclose(
        gas.concentration.numpy(), expected, rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(
        np.sum(
            gas.concentration.numpy() * particles.volume.numpy()[:, None],
            axis=0,
        ),
        total_before,
        rtol=1e-12,
        atol=1e-30,
    )
    assert registry._views["communication_gas"] is resources


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_empty_closed_map_is_a_write_free_resident_barrier() -> None:
    """The canonical empty arbitrary-pairs carrier preserves work ledgers."""
    empty_int = np.zeros(0, dtype=np.int32)
    wp, session, registry, resources, request = _binding(
        np.zeros(0, dtype=np.float64),
        final_volumes=None,
        source_boxes=empty_int,
        destination_boxes=empty_int,
        enabled=empty_int,
        map_form=CommunicationMapForm.ARBITRARY_PAIRS,
    )
    gas = cast(Any, session.gas)
    before_gas = gas.concentration.numpy().copy()
    buffers = resources.buffers
    before_ledgers = tuple(
        getattr(buffers, name).numpy().copy()
        for name in ("amounts", "amount_deltas", "outbound_amounts")
    )

    ResidentCommunicationExecutor(request).execute_communication()

    wp.synchronize()
    npt.assert_array_equal(gas.concentration.numpy(), before_gas)
    for name, before in zip(
        ("amounts", "amount_deltas", "outbound_amounts"),
        before_ledgers,
        strict=True,
    ):
        npt.assert_array_equal(getattr(buffers, name).numpy(), before)
    assert registry._views["communication_gas"] is resources


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_disabled_closed_map_leaves_transport_ledgers_unchanged() -> None:
    """An all-disabled closed map is write-free before independent dilution."""
    wp, session, registry, resources, request = _binding(
        np.array([0.25, 0.5]),
        final_volumes=None,
        enabled=np.zeros(2, dtype=np.int32),
    )
    gas = cast(Any, session.gas)
    before_gas = gas.concentration.numpy().copy()
    buffers = resources.buffers
    before_ledgers = {
        name: np.full_like(getattr(buffers, name).numpy(), index + 1.0)
        for index, name in enumerate(
            ("amounts", "amount_deltas", "outbound_amounts")
        )
    }
    for name, sentinel in before_ledgers.items():
        getattr(buffers, name).assign(sentinel)

    ResidentCommunicationExecutor(request).execute_communication()

    wp.synchronize()
    npt.assert_array_equal(gas.concentration.numpy(), before_gas)
    for name, sentinel in before_ledgers.items():
        npt.assert_array_equal(getattr(buffers, name).numpy(), sentinel)
    assert registry._views["communication_gas"] is resources
    assert session.lifecycle.value == "active"
