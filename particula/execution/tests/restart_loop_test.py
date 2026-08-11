"""Regression checks for resident transport restart continuation."""

from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import Backend, Device
from particula.execution.checkpoint import restart_resident_session
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


def _communication_request(session: Any, registry: Any, resources: Any) -> Any:
    """Rebuild barrier graph objects around one exact resident binding."""
    from particula.execution import CapabilityRequirements, process_graph

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


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_restart_restores_closed_transport_and_published_stream_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint recreates fresh transport resources and exact RNG words."""
    wp = pytest.importorskip("warp")
    import particula.execution.gpu_resources as gpu_resources
    from particula.execution.communication import (
        CommunicationConfiguration,
        CommunicationMap,
        CommunicationMapForm,
        CommunicationTransportMode,
        PrescribedVolumeUpdate,
    )
    from particula.execution.tests.checkpoint_test import _resident_binding

    session, registry, guard = _resident_binding(n_boxes=2)
    configuration = CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ONE_DIMENSIONAL,
            CommunicationTransportMode.GAS,
            1,
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([0.25], dtype=wp.float64, device="cpu"),
        ),
        PrescribedVolumeUpdate(
            wp.array([2.0, 4.0], dtype=wp.float64, device="cpu")
        ),
        (),
    )
    resources = registry.acquire_communication(configuration)
    source_executor = ResidentCommunicationExecutor(
        _communication_request(session, registry, resources)
    )
    source_executor.execute_communication()
    source_executor.execute_volume_evolution()
    wp.synchronize()
    coagulation = registry.acquire_coagulation(1)
    wall_loss = registry.acquire_wall_loss()
    coagulation.rng_states.assign(np.array([17, 29], dtype=np.uint32))
    wall_loss.rng_states.assign(np.array([31, 43], dtype=np.uint32))
    checkpoint = session.checkpoint(registry, guard)
    source_gas = cast(Any, session.gas).concentration.numpy().copy()
    source_particle_concentration = (
        cast(Any, session.particles).concentration.numpy().copy()
    )
    source_final_volumes = resources.final_volumes.numpy().copy()

    monkeypatch.setattr(
        gpu_resources.StreamRegistry,
        "initialize",
        lambda _self: pytest.fail("restart must retain checkpoint words"),
    )
    monkeypatch.setattr(
        gpu_resources.StreamRegistry,
        "initialize_process",
        lambda _self, _process: pytest.fail("restart must not reseed a stream"),
    )
    restored, restored_registry, restored_guard = restart_resident_session(
        checkpoint, Device(Backend.WARP, "cpu")
    )
    restored_resources = restored_registry._views["communication_gas"]
    restored_coagulation = restored_registry.acquire_coagulation(1)
    restored_wall_loss = restored_registry.acquire_wall_loss()

    assert restored is not session
    assert restored_registry is not registry
    assert restored_guard is not guard
    assert restored_resources is not resources
    assert restored_resources.configuration is not resources.configuration
    assert (
        restored_resources.configuration.communication_map
        is not resources.configuration.communication_map
    )
    for name in ("source_boxes", "destination_boxes", "enabled", "rates"):
        assert getattr(
            restored_resources.configuration.communication_map, name
        ) is not getattr(resources.configuration.communication_map, name)
    assert restored_resources.buffers is not resources.buffers
    for name in ("amounts", "amount_deltas", "outbound_amounts"):
        assert getattr(restored_resources.buffers, name) is not getattr(
            resources.buffers, name
        )
    assert restored_resources.final_volumes is not resources.final_volumes
    assert restored_coagulation.rng_states is not coagulation.rng_states
    assert restored_wall_loss.rng_states is not wall_loss.rng_states
    npt.assert_allclose(
        cast(Any, restored.gas).concentration.numpy(),
        source_gas,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        cast(Any, restored.particles).concentration.numpy(),
        source_particle_concentration,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        restored_resources.final_volumes.numpy(),
        source_final_volumes,
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_array_equal(restored_coagulation.rng_states.numpy(), [17, 29])
    npt.assert_array_equal(restored_wall_loss.rng_states.numpy(), [31, 43])

    source_executor.execute_communication()
    source_executor.execute_volume_evolution()
    restored_executor = ResidentCommunicationExecutor(
        _communication_request(restored, restored_registry, restored_resources)
    )
    restored_executor.execute_communication()
    restored_executor.execute_volume_evolution()
    wp.synchronize()
    npt.assert_allclose(
        cast(Any, restored.gas).concentration.numpy(),
        cast(Any, session.gas).concentration.numpy(),
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        cast(Any, restored.particles).concentration.numpy(),
        cast(Any, session.particles).concentration.numpy(),
        rtol=1e-12,
        atol=1e-30,
    )
    npt.assert_allclose(
        cast(Any, restored.particles).volume.numpy(),
        cast(Any, session.particles).volume.numpy(),
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.warp
def test_restart_rejects_nonexact_device_without_mutating_source() -> None:
    """A mismatched target device is rejected before source state changes."""
    from particula.execution.tests.checkpoint_test import _resident_binding

    session, registry, guard = _resident_binding()
    source = cast(Any, session.gas).concentration.numpy().copy()
    checkpoint = session.checkpoint(registry, guard)

    with pytest.raises(ValueError, match="exactly match"):
        restart_resident_session(checkpoint, Device(Backend.CPU, "cpu"))

    npt.assert_array_equal(cast(Any, session.gas).concentration.numpy(), source)
    assert session.lifecycle.value == "active"
    assert registry._views == {}
