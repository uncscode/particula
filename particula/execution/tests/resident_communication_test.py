"""Tests for concrete resident communication barrier dispatch."""

from dataclasses import replace
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
    _enqueue_prepared_resident_communication,
    setup_prepared_resident_communication,
    validate_resident_communication_request,
)


def _prepared_request(
    monkeypatch: pytest.MonkeyPatch,
    mode: CommunicationTransportMode,
) -> tuple[Any, Any]:
    """Build a production READY request and its P1 carrier for one mode."""
    from particula.execution.graph_capture import (
        GraphCaptureAvailability,
        GraphCaptureCapability,
        ResidentGraphCaptureBinding,
        _attach_resident_graph_capture_binding,
        create_graph_capture_lifecycle,
        create_resident_graph_capture_signature,
    )
    from particula.execution.resident_enqueue import prepare_resident_timestep
    from particula.execution.tests.full_loop_test import _build_loop_fixture

    fixture = _build_loop_fixture(monkeypatch, mode)
    request = fixture.request
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            request.session.metadata.device, GraphCaptureAvailability.AVAILABLE
        ),
        create_resident_graph_capture_signature(request),
    )
    _attach_resident_graph_capture_binding(
        request,
        ResidentGraphCaptureBinding(
            request, request.session, request.registry, request.guard, lifecycle
        ),
    )
    return request, prepare_resident_timestep(request, 0.0)


@pytest.mark.warp
@pytest.mark.parametrize(
    "mode",
    [CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES],
)
def test_prepared_binding_retains_p1_resources_and_dispatches_in_order(
    monkeypatch: pytest.MonkeyPatch, mode: CommunicationTransportMode
) -> None:
    """Prepared enqueue uses only frozen mode inputs and then final volumes."""
    import particula.execution.resident_communication as resident

    calls: list[str] = []
    monkeypatch.setattr(
        resident,
        "_enqueue_prepared_resident_gas_communication",
        lambda *_args: calls.append("gas"),
    )
    monkeypatch.setattr(
        resident,
        "_enqueue_prepared_resident_particle_communication",
        lambda *_args: calls.append("particles"),
    )
    monkeypatch.setattr(
        resident,
        "_enqueue_prepared_resident_volume_evolution",
        lambda *_args: calls.append("volume"),
    )
    request, prepared_timestep = _prepared_request(monkeypatch, mode)
    binding = setup_prepared_resident_communication(
        prepared_timestep, request.communication
    )
    assert binding.prepared_timestep is prepared_timestep
    assert binding.request is request.communication
    assert (
        binding.final_volumes is request.communication.resources.final_volumes
    )

    _enqueue_prepared_resident_communication(binding)

    expected = "gas" if mode is CommunicationTransportMode.GAS else "particles"
    assert calls == [expected] + (
        ["volume"] if binding.final_volumes is not None else []
    )


@pytest.mark.warp
@pytest.mark.parametrize(
    "mode",
    [CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES],
)
def test_prepared_enqueue_uses_frozen_arrays_without_runtime_dispatch(
    monkeypatch: pytest.MonkeyPatch, mode: CommunicationTransportMode
) -> None:
    """Replacing container fields after setup cannot affect bound enqueue."""
    import particula.execution.resident_communication as resident

    calls: list[tuple[object, ...]] = []

    def record(*args: object) -> None:
        """Record only setup-bound communication arguments."""
        calls.append(args)

    monkeypatch.setattr(
        resident, "_enqueue_prepared_resident_gas_communication", record
    )
    monkeypatch.setattr(
        resident, "_enqueue_prepared_resident_particle_communication", record
    )
    request, prepared_timestep = _prepared_request(monkeypatch, mode)
    binding = setup_prepared_resident_communication(
        prepared_timestep, request.communication
    )
    monkeypatch.setattr(
        resident,
        "_enqueue_prepared_resident_gas_communication",
        lambda *_args: pytest.fail("prepared enqueue must not resolve GAS"),
    )
    monkeypatch.setattr(
        resident,
        "_enqueue_prepared_resident_particle_communication",
        lambda *_args: pytest.fail(
            "prepared enqueue must not resolve PARTICLES"
        ),
    )
    wp = pytest.importorskip("warp")
    session_particles = cast(Any, request.session.particles)
    session_gas = cast(Any, request.session.gas)
    session_particles.masses = wp.zeros(
        cast(Any, binding.masses).shape,
        dtype=wp.float64,
        device=binding.device,
    )
    session_gas.concentration = wp.zeros(
        cast(Any, binding.gas_concentration).shape,
        dtype=wp.float64,
        device=binding.device,
    )

    _enqueue_prepared_resident_communication(binding)

    assert calls == [binding.communication_arguments]
    assert binding.communication_enqueue is record


@pytest.mark.warp
def test_prepared_setup_rejects_request_not_retained_by_p1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup rejects a detached request before it can enter prepared enqueue."""
    request, prepared_timestep = _prepared_request(
        monkeypatch, CommunicationTransportMode.GAS
    )
    detached = replace(request.communication, duration=0.0)

    with pytest.raises(ValueError, match="does not retain"):
        setup_prepared_resident_communication(prepared_timestep, detached)


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
def test_capture_registration_retains_one_closed_communication_view(
    mode: CommunicationTransportMode,
) -> None:
    """Test capture registration retains the selected closed family by identity."""
    from particula.execution.tests.gpu_resources_test import _diagnostics_plan

    request = _request(mode)
    registry = cast(GPUResourceRegistry, request.registry)
    wp = pytest.importorskip("warp")
    outputs = tuple(
        wp.zeros((3, 1), dtype=wp.float64, device="cpu") for _ in range(2)
    )
    plan = _diagnostics_plan(request.session, registry, outputs)

    assert registry._capture_inventory is None
    inventory = registry.register_capture_resources(
        request.session, request.resources, plan.registrations
    )

    assert inventory.communication_resources is request.resources
    assert inventory.registrations is plan.registrations
    family = f"communication_{mode.value}"
    assert inventory.families[0].family == family
    assert [role.canonical_name for role in inventory.families[0].roles][
        :4
    ] == [
        f"{family}:source_boxes",
        f"{family}:destination_boxes",
        f"{family}:enabled",
        f"{family}:rates",
    ]
    assert registry.selected_resource_report() is inventory


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

    monkeypatch.setattr(native, "resident_gas_communication_step_gpu", gas)
    monkeypatch.setattr(
        native, "resident_particle_communication_step_gpu", particles
    )

    result = ResidentCommunicationExecutor(request).execute_communication()

    assert result == mode.value
    assert len(calls) == 1
    if mode is CommunicationTransportMode.GAS:
        assert calls[0][:5] == (
            request.session.particles,
            request.session.gas,
            request.resources.configuration,
            request.duration,
            request.resources.buffers,
        )
        assert calls[0][5:] == (
            request.resources.execution_state.invalid,
            request.resources.execution_state.active_or_demand,
        )
    else:
        assert calls[0][:4] == (
            request.session.particles,
            request.resources.configuration,
            request.duration,
            request.resources.buffers,
        )
        assert calls[0][4:] == (
            request.resources.execution_state.invalid,
            request.resources.execution_state.active_or_demand,
            request.resources.execution_state.initial_masses,
            request.resources.execution_state.initial_concentration,
            request.resources.execution_state.initial_charge,
        )


@pytest.mark.warp
def test_particle_dispatch_uses_only_registry_pinned_status_and_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resident particle dispatch does not allocate public-step status buffers."""
    request = _request(CommunicationTransportMode.PARTICLES)
    import particula.gpu.kernels.communication as native

    monkeypatch.setattr(
        native.wp,
        "zeros",
        lambda *_args, **_kwargs: pytest.fail(
            "resident dispatch must not allocate status storage"
        ),
    )

    result = ResidentCommunicationExecutor(request).execute_communication()

    assert result is request.session.particles


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
        "resident_gas_communication_step_gpu",
        lambda *_args: pytest.fail("native primitive must not be called"),
    )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        ResidentCommunicationExecutor(request).execute_communication()

    np.testing.assert_array_equal(
        cast(Any, request.session.gas).concentration.numpy(), gas_before
    )


@pytest.mark.warp
def test_functional_validator_matches_executor_validation() -> None:
    """The no-construction validator retains the accepted request by identity."""
    request = _request(CommunicationTransportMode.GAS)

    assert validate_resident_communication_request(request) is request
    ResidentCommunicationExecutor(request).validate()


@pytest.mark.warp
def test_executor_rejects_mismatched_barrier_node_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Barrier-node identity drift rejects before a native writer is called."""
    request = _request(CommunicationTransportMode.GAS)
    invalid_request = replace(
        request, volume_evolution_node=request.communication_node
    )
    gas_before = cast(Any, request.session.gas).concentration.numpy().copy()
    import particula.gpu.kernels.communication as native

    monkeypatch.setattr(
        native,
        "resident_gas_communication_step_gpu",
        lambda *_args: pytest.fail("native primitive must not be called"),
    )

    with pytest.raises(ValueError, match="barrier nodes do not match graph"):
        ResidentCommunicationExecutor(invalid_request).execute_communication()

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
        "resident_volume_evolution_step_gpu",
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

    def volume(*args: object) -> str:
        """Record the volume primitive's exact arguments."""
        calls.append(args)
        return "volume"

    monkeypatch.setattr(
        native,
        "resident_volume_evolution_step_gpu",
        volume,
    )

    assert (
        ResidentCommunicationExecutor(request).execute_volume_evolution()
        == "volume"
    )
    assert calls == [
        (
            request.session.particles,
            request.session.gas,
            final_volumes,
            request.resources.execution_state.volume_invalid,
            request.resources.execution_state.volume_changed,
        )
    ]
