"""Independent multi-box resident communication parity coverage."""

from __future__ import annotations

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

RTOL = 1e-12
INVENTORY_ATOL = 1e-30


def _request(
    mode: CommunicationTransportMode,
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
    final_volumes: np.ndarray | None,
    *,
    form: CommunicationMapForm = CommunicationMapForm.ARBITRARY_PAIRS,
    enabled: np.ndarray | None = None,
) -> ResidentCommunicationRequest:
    """Build a three-box exact resident request with a closed map."""
    wp = pytest.importorskip("warp")
    from particula.execution import process_graph
    from particula.execution.tests.gpu_resources_test import _session

    session = _session(boxes=3, particle_count=4, species=2)
    session.particles.volume = wp.array(
        [2.0, 1.0, 4.0], dtype=wp.float64, device="cpu"
    )
    session.gas.concentration = wp.array(
        [[10.0, 2.0], [3.0, 4.0], [1.0, 8.0]],
        dtype=wp.float64,
        device="cpu",
    )
    if mode is CommunicationTransportMode.PARTICLES:
        session.particles.masses = wp.array(
            [
                [[2.0, 3.0], [0.0, 0.0], [5.0, 7.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [11.0, 13.0], [0.0, 0.0]],
                [[17.0, 19.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=wp.float64,
            device="cpu",
        )
        session.particles.concentration = wp.array(
            [[4.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
            dtype=wp.float64,
            device="cpu",
        )
        session.particles.charge = wp.array(
            [
                [-2.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [-4.0, 0.0, 0.0, 0.0],
            ],
            dtype=wp.float64,
            device="cpu",
        )
    configuration = CommunicationConfiguration(
        CommunicationMap(
            form,
            mode,
            len(source),
            wp.array(source, dtype=wp.int32, device="cpu"),
            wp.array(destination, dtype=wp.int32, device="cpu"),
            wp.array(
                np.ones(len(source), dtype=np.int32)
                if enabled is None
                else enabled,
                dtype=wp.int32,
                device="cpu",
            ),
            wp.array(rates, dtype=wp.float64, device="cpu"),
        ),
        PrescribedVolumeUpdate(
            None
            if final_volumes is None
            else wp.array(final_volumes, dtype=wp.float64, device="cpu")
        ),
        (
            CommunicationResourceShape(
                "edge_rates", wp.float64, CommunicationShapeKind.E
            ),
        ),
    )
    registry = GPUResourceRegistry(session)
    resources = registry.acquire_communication(configuration)
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
    return ResidentCommunicationRequest(
        session,
        registry,
        graph,
        resources,
        by_id["communication"],
        by_id["volume_evolution"],
        1.0,
    )


def _gas_oracle(
    concentration: np.ndarray,
    volume: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """Calculate one closed explicit-Euler barrier from immutable amounts."""
    amounts = concentration * volume[:, None]
    deltas = np.zeros_like(amounts)
    for left, right, rate in zip(source, destination, rates, strict=True):
        transfer = amounts[left] * rate
        deltas[left] -= transfer
        deltas[right] += transfer
    return (amounts + deltas) / volume[:, None]


def _particle_oracle(  # noqa: C901
    masses: np.ndarray,
    concentration: np.ndarray,
    charge: np.ndarray,
    volume: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently plan immutable-prestate particle transport in NumPy."""
    final_masses = masses.copy()
    final_concentration = concentration.copy()
    final_charge = charge.copy()
    credits = np.zeros_like(concentration)
    debits = np.zeros_like(concentration)
    initial_keys = {
        (box, slot): (*masses[box, slot], charge[box, slot])
        for box in range(concentration.shape[0])
        for slot in range(concentration.shape[1])
        if concentration[box, slot] > 0.0
    }
    occupied = {
        box: {
            key: slot
            for (key_box, slot), key in initial_keys.items()
            if key_box == box
        }
        for box in range(concentration.shape[0])
    }
    reserved: dict[tuple[int, tuple[float, ...]], int] = {}
    used = {box: set() for box in range(concentration.shape[0])}
    incoming: dict[tuple[int, int], tuple[int, int]] = {}
    for _edge, (left, right, rate) in enumerate(
        zip(source, destination, rates, strict=True)
    ):
        for slot in range(concentration.shape[1]):
            request = concentration[left, slot] * rate
            if request <= 0.0:
                continue
            key = initial_keys[left, slot]
            assigned = occupied[right].get(key)
            if assigned is None:
                reservation = (right, key)
                assigned = reserved.get(reservation)
                if assigned is None:
                    assigned = next(
                        free
                        for free in range(concentration.shape[1])
                        if concentration[right, free] == 0.0
                        and free not in used[right]
                    )
                    reserved[reservation] = assigned
                    used[right].add(assigned)
            debits[left, slot] += request
            credits[right, assigned] += request * volume[left] / volume[right]
            incoming.setdefault((right, assigned), (left, slot))
    final_concentration += credits - debits
    for (box, slot), (left, source_slot) in incoming.items():
        if concentration[box, slot] == 0.0:
            final_masses[box, slot] = masses[left, source_slot]
            final_charge[box, slot] = charge[left, source_slot]
    for box in range(concentration.shape[0]):
        for slot in range(concentration.shape[1]):
            if final_concentration[box, slot] == 0.0:
                final_masses[box, slot] = 0.0
                final_charge[box, slot] = 0.0
    return final_masses, final_concentration, final_charge


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    ("source", "destination", "rates", "final_volumes"),
    [
        (
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.25, 0.5], dtype=np.float64),
            np.array([4.0, 2.0, 4.0], dtype=np.float64),
        ),
        (
            np.array([0, 1, 2], dtype=np.int32),
            np.array([1, 0, 0], dtype=np.int32),
            np.array([0.1, 0.2, 0.125], dtype=np.float64),
            np.array([2.0, 1.0, 8.0], dtype=np.float64),
        ),
    ],
    ids=["chain", "reciprocal-isolated"],
)
def test_resident_gas_barriers_match_repeated_numpy_oracle(
    source: np.ndarray,
    destination: np.ndarray,
    rates: np.ndarray,
    final_volumes: np.ndarray,
) -> None:
    """Keep communication-before-volume ordering and gas conservation explicit."""
    wp = pytest.importorskip("warp")
    request = _request(
        CommunicationTransportMode.GAS,
        source,
        destination,
        rates,
        final_volumes,
    )
    executor = ResidentCommunicationExecutor(request)
    gas = cast(Any, request.session.gas)
    particles = cast(Any, request.session.particles)
    initial = gas.concentration.numpy().copy()
    initial_particle_concentration = particles.concentration.numpy().copy()
    old_volume = particles.volume.numpy().copy()
    expected = initial.copy()

    # Two barriers use the current state, with volume evolution after each one.
    for _ in range(2):
        expected = _gas_oracle(expected, old_volume, source, destination, rates)
        expected *= old_volume[:, None] / final_volumes[:, None]
        old_volume = final_volumes.copy()
        executor.execute_communication()
        executor.execute_volume_evolution()
    wp.synchronize()

    npt.assert_allclose(
        gas.concentration.numpy(), expected, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.volume.numpy(), final_volumes, rtol=0.0, atol=0.0
    )
    npt.assert_allclose(
        particles.concentration.numpy(),
        initial_particle_concentration
        * np.array([2.0, 1.0, 4.0])[:, None]
        / final_volumes[:, None],
        rtol=RTOL,
        atol=0.0,
    )
    npt.assert_allclose(
        (
            particles.concentration.numpy() * particles.volume.numpy()[:, None]
        ).sum(),
        (
            initial_particle_concentration * np.array([2.0, 1.0, 4.0])[:, None]
        ).sum(),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )
    npt.assert_allclose(
        (gas.concentration.numpy() * particles.volume.numpy()[:, None]).sum(
            axis=0
        ),
        (initial * np.array([2.0, 1.0, 4.0])[:, None]).sum(axis=0),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_resident_particle_barrier_conserves_multibox_weighted_state() -> None:
    """Exercise sparse closed particle transport through the resident executor."""
    wp = pytest.importorskip("warp")
    source = np.array([0, 1], dtype=np.int32)
    destination = np.array([1, 2], dtype=np.int32)
    request = _request(
        CommunicationTransportMode.PARTICLES,
        source,
        destination,
        np.array([0.25, 0.5], dtype=np.float64),
        None,
    )
    particles = cast(Any, request.session.particles)
    volume = particles.volume.numpy().copy()
    mass = particles.masses.numpy().copy()
    concentration = particles.concentration.numpy().copy()
    charge = particles.charge.numpy().copy()

    ResidentCommunicationExecutor(request).execute_communication()
    wp.synchronize()

    expected_masses, expected_concentration, expected_charge = _particle_oracle(
        mass,
        concentration,
        charge,
        volume,
        source,
        destination,
        np.array([0.25, 0.5], dtype=np.float64),
    )
    npt.assert_allclose(
        particles.concentration.numpy(),
        expected_concentration,
        rtol=RTOL,
        atol=0.0,
    )
    npt.assert_allclose(
        particles.masses.numpy(), expected_masses, rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.charge.numpy(), expected_charge, rtol=RTOL, atol=0.0
    )

    npt.assert_allclose(
        (particles.concentration.numpy() * volume[:, None]).sum(),
        (concentration * volume[:, None]).sum(),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )
    npt.assert_allclose(
        (
            particles.masses.numpy()
            * particles.concentration.numpy()[..., None]
            * volume[:, None, None]
        ).sum(axis=(0, 1)),
        (mass * concentration[..., None] * volume[:, None, None]).sum(
            axis=(0, 1)
        ),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )
    npt.assert_allclose(
        (
            particles.charge.numpy()
            * particles.concentration.numpy()
            * volume[:, None]
        ).sum(),
        (charge * concentration * volume[:, None]).sum(),
        rtol=RTOL,
        atol=INVENTORY_ATOL,
    )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "order",
    [np.array([0, 1], dtype=np.int32), np.array([1, 0], dtype=np.int32)],
    ids=["registered", "permuted"],
)
def test_resident_one_dimensional_padded_map_matches_particle_oracle(
    order: np.ndarray,
) -> None:
    """Keep sparse slots and an isolated box independent through residency."""
    wp = pytest.importorskip("warp")
    map_source = np.array([0, 1], dtype=np.int32)[order]
    map_destination = np.array([1, 2], dtype=np.int32)[order]
    map_rates = np.array([0.25, 0.5], dtype=np.float64)[order]
    enabled = np.array([1, 0], dtype=np.int32)[order]
    request = _request(
        CommunicationTransportMode.PARTICLES,
        map_source,
        map_destination,
        map_rates,
        None,
        form=CommunicationMapForm.ONE_DIMENSIONAL,
        enabled=enabled,
    )
    particles = cast(Any, request.session.particles)
    initial = (
        particles.masses.numpy().copy(),
        particles.concentration.numpy().copy(),
        particles.charge.numpy().copy(),
        particles.volume.numpy().copy(),
    )
    selected = enabled == 1
    expected = _particle_oracle(
        *initial[:4],
        map_source[selected],
        map_destination[selected],
        map_rates[selected],
    )

    ResidentCommunicationExecutor(request).execute_communication()
    wp.synchronize()

    npt.assert_allclose(
        particles.masses.numpy(), expected[0], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.concentration.numpy(), expected[1], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.charge.numpy(), expected[2], rtol=RTOL, atol=0.0
    )
    npt.assert_allclose(
        particles.concentration.numpy()[2], initial[1][2], rtol=0.0, atol=0.0
    )
