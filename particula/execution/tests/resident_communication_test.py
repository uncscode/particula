"""Tests for concrete resident communication barrier dispatch."""

from typing import Any, cast

import numpy as np
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


def _configuration(
    mode: CommunicationTransportMode,
    *,
    final_volumes: Any | None = None,
) -> CommunicationConfiguration:
    """Build a valid closed two-edge, three-box communication configuration."""
    wp = pytest.importorskip("warp")
    map_data = CommunicationMap(
        CommunicationMapForm.ONE_DIMENSIONAL,
        mode,
        2,
        wp.array([0, 1], dtype=wp.int32, device="cpu"),
        wp.array([1, 2], dtype=wp.int32, device="cpu"),
        wp.array([1, 1], dtype=wp.int32, device="cpu"),
        wp.array([0.1, 0.2], dtype=wp.float64, device="cpu"),
    )
    return CommunicationConfiguration(
        map_data,
        PrescribedVolumeUpdate(final_volumes),
        (
            CommunicationResourceShape(
                "edge_rates", wp.float64, CommunicationShapeKind.E
            ),
        ),
    )


def _request(
    mode: CommunicationTransportMode,
    *,
    final_volumes: Any | None = None,
    duration: float = 0.5,
) -> ResidentCommunicationRequest:
    """Build an exact resource, graph, and request binding for one barrier."""
    from particula.execution import process_graph
    from particula.execution.tests.gpu_resources_test import _session

    session = _session(boxes=3, particle_count=2, species=1)
    registry = GPUResourceRegistry(session)
    resources = registry.acquire_communication(
        _configuration(mode, final_volumes=final_volumes)
    )
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
        duration,
    )


@pytest.mark.warp
@pytest.mark.parametrize(
    "mode",
    [CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES],
)
def test_executor_dispatches_only_the_selected_native_primitive(
    monkeypatch: pytest.MonkeyPatch, mode: CommunicationTransportMode
) -> None:
    """Selected mode dispatches once with retained resident object identities."""
    request = _request(mode)
    import particula.gpu.kernels.communication as native

    calls: list[tuple[object, ...]] = []

    def gas(*args: object) -> str:
        """Record the gas primitive's exact arguments."""
        calls.append(args)
        return "gas"

    def particles(*args: object) -> str:
        """Record the particle primitive's exact arguments."""
        calls.append(args)
        return "particles"

    monkeypatch.setattr(native, "gas_communication_step_gpu", gas)
    monkeypatch.setattr(native, "particle_communication_step_gpu", particles)

    result = ResidentCommunicationExecutor(request).execute_communication()

    assert result == mode.value
    assert len(calls) == 1
    if mode is CommunicationTransportMode.GAS:
        assert calls[0] == (
            request.session.particles,
            request.session.gas,
            request.resources.configuration,
            request.duration,
            request.resources.buffers,
        )
    else:
        assert calls[0] == (
            request.session.particles,
            request.resources.configuration,
            request.duration,
            request.resources.buffers,
        )


@pytest.mark.warp
def test_executor_rejects_invalid_duration_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative duration is preflight-only and leaves resident arrays unchanged."""
    request = _request(CommunicationTransportMode.GAS, duration=-1.0)
    gas_before = cast(Any, request.session.gas).concentration.numpy().copy()
    import particula.gpu.kernels.communication as native

    monkeypatch.setattr(
        native,
        "gas_communication_step_gpu",
        lambda *_args: pytest.fail("native primitive must not be called"),
    )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        ResidentCommunicationExecutor(request).execute_communication()

    np.testing.assert_array_equal(
        cast(Any, request.session.gas).concentration.numpy(), gas_before
    )


@pytest.mark.warp
def test_volume_dispatch_is_skipped_without_prescribed_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent prescribed volumes make the volume barrier an exact no-op."""
    request = _request(CommunicationTransportMode.GAS)
    import particula.gpu.kernels.communication as native

    monkeypatch.setattr(
        native,
        "volume_evolution_step_gpu",
        lambda *_args: pytest.fail("volume primitive must not be called"),
    )

    assert (
        ResidentCommunicationExecutor(request).execute_volume_evolution()
        is None
    )


@pytest.mark.warp
def test_volume_dispatch_retains_final_volume_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present final volumes dispatch exactly once with resident identities."""
    wp = pytest.importorskip("warp")
    final_volumes = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device="cpu")
    request = _request(
        CommunicationTransportMode.GAS, final_volumes=final_volumes
    )
    import particula.gpu.kernels.communication as native

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        native,
        "volume_evolution_step_gpu",
        lambda *args: calls.append(args) or "volume",
    )

    assert (
        ResidentCommunicationExecutor(request).execute_volume_evolution()
        == "volume"
    )
    assert calls == [
        (request.session.particles, request.session.gas, final_volumes)
    ]
