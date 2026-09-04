"""Test declaration-only graph-capture capability metadata."""

from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

import particula
import particula.execution as execution
import particula.execution.graph_capture as graph_capture
from particula.execution import Backend, Device
from particula.execution.graph_capture import (
    GraphCaptureAvailability,
    GraphCaptureCapability,
    GraphCaptureCompatibility,
    GraphCaptureDriftReason,
    GraphCaptureFailureClassification,
    GraphCaptureLifecycle,
    GraphCaptureLifecycleState,
    GraphCaptureNativeCallables,
    GraphCaptureRuntimeAdapter,
    PreparedGraphCaptureQualification,
    ResidentGraphCaptureBinding,
    ResidentGraphCaptureSignature,
    _attach_resident_graph_capture_binding,
    capture_prepared_resident_graph,
    classify_graph_capture_failure,
    classify_resident_graph_capture_writer_failure,
    close_graph_capture,
    close_resident_graph_capture,
    compare_resident_graph_capture_signature,
    complete_graph_capture,
    complete_resident_graph_capture,
    create_graph_capture_lifecycle,
    create_resident_graph_capture_signature,
    gate_resident_graph_capture,
    invalidate_graph_capture,
    qualify_prepared_resident_graph_capture,
    renew_resident_graph_capture,
    renew_retired_graph_capture,
    resolve_graph_capture_capability,
    retire_graph_capture,
    retire_resident_graph_capture,
)

if TYPE_CHECKING:
    from particula.execution.gpu_resources import (
        CaptureResourceRequirements,
        GPUResourceRegistry,
    )
    from particula.execution.graph_capture import GraphCaptureRuntimeProbe
    from particula.execution.resident_communication import (
        ResidentCommunicationRequest,
    )
    from particula.execution.resident_scheduler import ResidentSimulationRequest


class RecordingProbe:
    """Record lazy graph-capture probe calls and configured outcomes."""

    def __init__(self, runtime: object, device: object, api: object) -> None:
        """Store outcome values without coercing them to booleans."""
        self.runtime = runtime
        self.device = device
        self.api = api
        self.calls: list[str] = []

    def runtime_available(self) -> bool:
        """Record and return the configured runtime outcome."""
        self.calls.append("runtime")
        if isinstance(self.runtime, Exception):
            raise self.runtime
        return self.runtime  # type: ignore[return-value]

    def device_available(self, device: Device) -> bool:
        """Record and return the configured device outcome."""
        del device
        self.calls.append("device")
        if isinstance(self.device, Exception):
            raise self.device
        return self.device  # type: ignore[return-value]

    def capture_api_available(self, device: Device) -> bool:
        """Record and return the configured API outcome."""
        del device
        self.calls.append("api")
        if isinstance(self.api, Exception):
            raise self.api
        return self.api  # type: ignore[return-value]


class AttributeTrapProbe:
    """Fail if capability resolution reads a probe member."""

    def __getattribute__(self, name: str) -> object:
        """Reject every attempted probe attribute access."""
        raise AssertionError(f"unsupported resolution read probe.{name}")


@pytest.mark.parametrize(
    "field",
    (
        "capture_begin",
        "capture_end",
        "capture_instantiate",
        "capture_launch",
        "capture_release",
    ),
)
def test_native_callables_reject_noncallable_fields(field: str) -> None:
    """Native vocabulary validates each callable independently."""

    def callback() -> None:
        """Provide a valid callback."""

    values: dict[str, object] = {
        "capture_begin": callback,
        "capture_end": callback,
        "capture_instantiate": callback,
        "capture_launch": callback,
        "capture_release": callback,
    }
    values[field] = object()

    with pytest.raises(TypeError, match=f"{field} must be callable"):
        GraphCaptureNativeCallables(**values)  # type: ignore[arg-type]


def test_native_callables_retain_distinct_callable_identities() -> None:
    """P1 vocabulary retains callables without a native handle or cleanup."""

    def begin() -> None:
        """Provide one capture-begin callable."""

    def end() -> None:
        """Provide one capture-end callable."""

    def instantiate() -> None:
        """Provide one capture-instantiate callable."""

    def launch() -> None:
        """Provide one capture-launch callable."""

    def release() -> None:
        """Provide one capture-release callable."""

    callables = GraphCaptureNativeCallables(
        begin, end, instantiate, launch, release
    )

    assert callables.capture_begin is begin
    assert callables.capture_end is end
    assert callables.capture_instantiate is instantiate
    assert callables.capture_launch is launch
    assert callables.capture_release is release
    assert not hasattr(callables, "native_handle")
    assert not hasattr(callables, "cleanup")
    assert callables != GraphCaptureNativeCallables(
        begin, end, instantiate, launch, release
    )
    with pytest.raises(FrozenInstanceError):
        callables.capture_begin = end  # type: ignore[misc]


def test_runtime_adapter_is_direct_import_only_protocol() -> None:
    """The concrete adapter protocol remains directly importable."""
    assert GraphCaptureRuntimeAdapter.__module__ == graph_capture.__name__


@pytest.fixture
def resident_request(monkeypatch: pytest.MonkeyPatch) -> object:
    """Build one real, fully published resident request on Warp CPU."""
    pytest.importorskip("warp")
    from particula.execution.communication import CommunicationTransportMode
    from particula.execution.tests.full_loop_test import _build_loop_fixture

    return _build_loop_fixture(
        monkeypatch, CommunicationTransportMode.GAS
    ).request


@pytest.mark.warp
def test_resident_request_requires_exact_capture_requirements_without_lookup(
    resident_request: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request construction validates only the exact requirements carrier."""
    request = cast("ResidentSimulationRequest", resident_request)
    registry = cast("GPUResourceRegistry", request.registry)
    monkeypatch.setattr(
        registry,
        "validate_capture_resource_set",
        lambda *_args: pytest.fail("request construction must not look up"),
    )

    class InexactRequirements(  # type: ignore[misc]
        type(request.capture_resource_requirements)  # type: ignore[misc]
    ):
        """Provide an inexact carrier without running inherited setup."""

    invalid_values = (
        None,
        object(),
        object.__new__(InexactRequirements),
    )
    for value in invalid_values:
        with pytest.raises(
            TypeError,
            match="capture_resource_requirements must be an exact",
        ):
            replace(request, capture_resource_requirements=value)


@pytest.mark.warp
def test_resident_signature_retains_capture_publication_triple(
    resident_request: object,
) -> None:
    """The configurations group ends with requirements, set, and report."""
    request = cast("ResidentSimulationRequest", resident_request)
    registry = cast("GPUResourceRegistry", request.registry)
    capture_set = registry.validate_capture_resource_set(
        cast(Any, request.capture_resource_requirements)
    )

    signature = create_resident_graph_capture_signature(request)

    assert signature.configurations[-3] is request.capture_resource_requirements
    assert signature.configurations[-2] is capture_set
    assert signature.configurations[-1] is capture_set.report


@pytest.mark.parametrize(
    ("device", "outcomes", "availability", "calls"),
    [
        (
            Device(Backend.CPU, "cpu"),
            (True, True, True),
            GraphCaptureAvailability.UNSUPPORTED_CPU,
            [],
        ),
        (
            Device(Backend.WARP, "cpu"),
            (True, True, True),
            GraphCaptureAvailability.UNSUPPORTED_WARP_CPU,
            [],
        ),
        (
            Device(Backend.WARP, "cuda:0"),
            (False, True, True),
            GraphCaptureAvailability.UNAVAILABLE_RUNTIME,
            ["runtime"],
        ),
        (
            Device(Backend.WARP, "opaque-device"),
            (True, False, True),
            GraphCaptureAvailability.UNAVAILABLE_DEVICE,
            ["runtime", "device"],
        ),
        (
            Device(Backend.WARP, "cuda:0"),
            (True, True, False),
            GraphCaptureAvailability.UNSUPPORTED_API,
            ["runtime", "device", "api"],
        ),
        (
            Device(Backend.WARP, "cuda:0"),
            (True, True, True),
            GraphCaptureAvailability.AVAILABLE,
            ["runtime", "device", "api"],
        ),
    ],
)
def test_resolve_capability_uses_ordered_lazy_probes(
    device: Device,
    outcomes: tuple[object, object, object],
    availability: GraphCaptureAvailability,
    calls: list[str],
) -> None:
    """Capability resolution stops at its first unavailable prerequisite."""
    probe = RecordingProbe(*outcomes)

    result = resolve_graph_capture_capability(device, probe)

    assert result.device is device
    assert result.availability is availability
    assert probe.calls == calls


@pytest.mark.parametrize("outcome", [1, None, "true"])
def test_resolve_capability_rejects_non_bool_probe_results(
    outcome: object,
) -> None:
    """Probe results must be literal booleans rather than truthy values."""
    probe = RecordingProbe(outcome, True, True)

    with pytest.raises(
        TypeError, match=r"runtime_available\(\) must return bool"
    ):
        resolve_graph_capture_capability(Device(Backend.WARP, "cuda:0"), probe)

    assert probe.calls == ["runtime"]


@pytest.mark.parametrize(
    ("outcomes", "error", "calls"),
    [
        (
            (True, 1, True),
            r"device_available\(\) must return bool",
            ["runtime", "device"],
        ),
        (
            (True, True, "true"),
            r"capture_api_available\(\) must return bool",
            ["runtime", "device", "api"],
        ),
    ],
)
def test_resolve_capability_rejects_non_bool_later_probe_results(
    outcomes: tuple[object, object, object], error: str, calls: list[str]
) -> None:
    """Every invoked lazy probe must return a literal boolean."""
    probe = RecordingProbe(*outcomes)

    with pytest.raises(TypeError, match=error):
        resolve_graph_capture_capability(Device(Backend.WARP, "cuda:0"), probe)

    assert probe.calls == calls


@pytest.mark.parametrize(
    ("outcomes", "error", "calls"),
    [
        ((np.bool_(True), True, True), "runtime_available", ["runtime"]),
        ((True, None, True), "device_available", ["runtime", "device"]),
        (
            (True, np.bool_(True), True),
            "device_available",
            ["runtime", "device"],
        ),
        (
            (True, True, 1),
            "capture_api_available",
            ["runtime", "device", "api"],
        ),
        (
            (True, True, None),
            "capture_api_available",
            ["runtime", "device", "api"],
        ),
        (
            (True, True, np.bool_(True)),
            "capture_api_available",
            ["runtime", "device", "api"],
        ),
    ],
)
def test_resolve_capability_rejects_every_nonliteral_bool_position(
    outcomes: tuple[object, object, object], error: str, calls: list[str]
) -> None:
    """Every probe position rejects truthy and falsey non-bool objects."""
    probe = RecordingProbe(*outcomes)

    with pytest.raises(TypeError, match=error):
        resolve_graph_capture_capability(Device(Backend.WARP, "cuda:0"), probe)

    assert probe.calls == calls


@pytest.mark.parametrize(
    ("outcomes", "calls"),
    [
        ((RuntimeError("runtime"), True, True), ["runtime"]),
        ((True, RuntimeError("device"), True), ["runtime", "device"]),
        ((True, True, RuntimeError("api")), ["runtime", "device", "api"]),
    ],
)
def test_resolve_capability_propagates_probe_exception_unchanged(
    outcomes: tuple[object, object, object], calls: list[str]
) -> None:
    """A probe exception escapes by identity without invoking later checks."""
    probe = RecordingProbe(*outcomes)
    expected = next(value for value in outcomes if isinstance(value, Exception))

    with pytest.raises(RuntimeError) as raised:
        resolve_graph_capture_capability(Device(Backend.WARP, "cuda:0"), probe)

    assert raised.value is expected
    assert probe.calls == calls


@pytest.mark.parametrize(
    ("method_name", "calls"),
    [
        ("runtime_available", []),
        ("device_available", ["runtime"]),
        ("capture_api_available", ["runtime", "device"]),
    ],
)
def test_resolve_capability_resolves_probe_methods_when_reached(
    method_name: str, calls: list[str]
) -> None:
    """Probe descriptors are resolved only after earlier checks succeed."""
    probe = RecordingProbe(True, True, True)
    setattr(probe, method_name, None)

    with pytest.raises(
        TypeError, match=rf"probe\.{method_name} must be callable"
    ):
        resolve_graph_capture_capability(Device(Backend.WARP, "cuda:0"), probe)

    assert probe.calls == calls


def test_unavailable_runtime_does_not_resolve_later_descriptors() -> None:
    """Runtime rejection returns without touching later probe descriptors."""

    class RuntimeOnlyProbe:
        """Expose only the prerequisite needed for runtime rejection."""

        def runtime_available(self) -> bool:
            return False

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"later descriptor was resolved: {name}")

    result = resolve_graph_capture_capability(
        Device(Backend.WARP, "cuda:0"),
        cast("GraphCaptureRuntimeProbe", RuntimeOnlyProbe()),
    )

    assert result.availability is GraphCaptureAvailability.UNAVAILABLE_RUNTIME


def test_resolve_capability_rejects_inexact_device() -> None:
    """The resolver rejects non-Device values before accessing the probe."""
    with pytest.raises(TypeError, match="exact Device"):
        resolve_graph_capture_capability(
            cast(Device, object()),
            cast("GraphCaptureRuntimeProbe", AttributeTrapProbe()),
        )


@pytest.mark.parametrize(
    ("device", "availability"),
    [
        (Device(Backend.CPU, "cpu"), GraphCaptureAvailability.UNSUPPORTED_CPU),
        (
            Device(Backend.WARP, "cpu"),
            GraphCaptureAvailability.UNSUPPORTED_WARP_CPU,
        ),
    ],
)
def test_unsupported_capability_paths_do_not_access_probe_members(
    device: Device, availability: GraphCaptureAvailability
) -> None:
    """CPU outcomes return before accessing caller-owned probe attributes."""
    result = resolve_graph_capture_capability(
        device, cast("GraphCaptureRuntimeProbe", AttributeTrapProbe())
    )

    assert result.availability is availability


@pytest.mark.warp
def test_resident_signature_accepts_real_request_by_identity(
    resident_request: object,
) -> None:
    """A real, unchanged resident request remains signature-compatible."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)

    compatible = compare_resident_graph_capture_signature(signature, request)

    assert compatible.compatible is True
    assert compatible.reason is None


@pytest.mark.warp
def test_admission_token_reuses_signature_without_reconstruction(
    resident_request: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated compatible admission uses the exact frozen signature token."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)

    def reject_reconstruction(*_args: object, **_kwargs: object) -> object:
        """Fail if the compatible admission path rebuilds a signature."""
        raise AssertionError("admission must reuse the retained signature")

    monkeypatch.setattr(
        graph_capture,
        "create_resident_graph_capture_signature",
        reject_reconstruction,
    )

    for _ in range(2):
        result = compare_resident_graph_capture_signature(
            signature,
            request,
            admission_token=signature,
        )
        assert result.compatible is True
        assert result.reason is None

    with pytest.raises(ValueError, match="retained signature"):
        compare_resident_graph_capture_signature(
            signature,
            request,
            admission_token=copy.copy(signature),
        )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("path", "reason"),
    (
        ("session.particles.masses", GraphCaptureDriftReason.PRIMARY_ARRAYS),
        ("diagnostics.node", GraphCaptureDriftReason.DIAGNOSTICS),
        (
            "communication.resources.configuration.communication_map",
            GraphCaptureDriftReason.COMMUNICATION,
        ),
        (
            "condensation.state.config",
            GraphCaptureDriftReason.CONFIGURATIONS,
        ),
        (
            "coagulation.resources.rng_states",
            GraphCaptureDriftReason.RNG_RESOURCES,
        ),
    ),
)
def test_cached_admission_checks_every_late_structural_group(
    resident_request: object,
    path: str,
    reason: GraphCaptureDriftReason,
) -> None:
    """Cached admission rejects representative nested drift in exact order."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    parent, attribute = _path_parent(request, path)
    original = getattr(parent, attribute)
    object.__setattr__(parent, attribute, object())
    try:
        result = compare_resident_graph_capture_signature(
            signature, request, admission_token=signature
        )
    finally:
        object.__setattr__(parent, attribute, original)

    assert result.reason is reason


@pytest.mark.warp
@pytest.mark.parametrize(
    "family",
    (
        "condensation",
        "coagulation",
        "wall_loss",
        "nucleation",
        "dilution",
        "communication",
        "diagnostics",
    ),
)
def test_capture_publication_rejects_substituted_dispatch_resources(
    resident_request: object,
    family: str,
) -> None:
    """Final dispatch cannot substitute resources outside the capture set."""
    request = cast("ResidentSimulationRequest", resident_request)
    owner: Any
    attribute: str
    replacement: Any
    if family == "condensation":
        owner = request.condensation.state
        attribute = "scratch_buffers"
        replacement = copy.copy(owner.scratch_buffers)
    elif family in {"coagulation", "wall_loss", "nucleation"}:
        owner = getattr(request, family)
        attribute = "resources"
        replacement = copy.copy(owner.resources)
    elif family == "dilution":
        owner = request
        attribute = "dilution"
        replacement = replace(request.dilution, resources=None)
    elif family == "communication":
        owner = request.communication
        attribute = "resources"
        replacement = copy.copy(owner.resources)
    else:
        owner = request.diagnostics
        attribute = "registrations"
        replacement = tuple([*owner.registrations])
    original = getattr(owner, attribute)
    object.__setattr__(owner, attribute, replacement)
    try:
        with pytest.raises(ValueError, match="Capture resource set identities"):
            create_resident_graph_capture_signature(request)
    finally:
        object.__setattr__(owner, attribute, original)


@pytest.mark.warp
def test_signature_reports_schedule_order_after_real_request(
    resident_request: object,
) -> None:
    """Schedule-order replacement is reported after all earlier groups match."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    object.__setattr__(request.schedule, "ordered_node_ids", object())

    result = compare_resident_graph_capture_signature(signature, request)

    assert result.compatible is False
    assert result.reason is GraphCaptureDriftReason.SCHEDULE_ORDER


@pytest.mark.warp
@pytest.mark.parametrize(
    ("target", "attribute", "reason"),
    [
        ("particles", "masses", GraphCaptureDriftReason.PRIMARY_ARRAYS),
        (
            "condensation_state",
            "scratch_buffers",
            GraphCaptureDriftReason.RESOURCE_VIEWS,
        ),
        ("graph", "nodes", GraphCaptureDriftReason.GRAPH),
        ("schedule", "nodes", GraphCaptureDriftReason.SCHEDULE),
        ("diagnostics", "node", GraphCaptureDriftReason.DIAGNOSTICS),
        (
            "communication",
            "communication_node",
            GraphCaptureDriftReason.COMMUNICATION,
        ),
        ("request", "thermodynamics", GraphCaptureDriftReason.CONFIGURATIONS),
    ],
)
def test_signature_reports_representative_real_request_drift(
    resident_request: object,
    target: str,
    attribute: str,
    reason: GraphCaptureDriftReason,
) -> None:
    """Each structural group reports its documented representative drift."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    targets = {
        "particles": request.session.particles,
        "condensation_state": request.condensation.state,
        "graph": request.graph,
        "schedule": request.schedule,
        "diagnostics": request.diagnostics,
        "communication": request.communication,
        "request": request,
    }

    object.__setattr__(targets[target], attribute, object())

    result = compare_resident_graph_capture_signature(signature, request)

    assert result.compatible is False
    assert result.reason is reason


def _path_parent(root: object, path: str) -> tuple[object, str]:
    """Resolve the parent and final attribute for a dotted fixture path."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@pytest.mark.warp
def test_signature_tracks_every_primary_array_identity(
    resident_request: object,
) -> None:
    """Every named primary array contributes to primary-array drift."""
    request = cast("ResidentSimulationRequest", resident_request)
    paths = (
        "session.particles.masses",
        "session.particles.concentration",
        "session.particles.density",
        "session.particles.volume",
        "session.particles.charge",
        "session.gas.molar_mass",
        "session.gas.concentration",
        "session.gas.partitioning",
        "session.gas.vapor_pressure",
        "session.environment.temperature",
        "session.environment.pressure",
        "session.environment.saturation_ratio",
    )

    for path in paths:
        signature = create_resident_graph_capture_signature(request)
        parent, attribute = _path_parent(request, path)
        original = getattr(parent, attribute)
        object.__setattr__(parent, attribute, object())
        try:
            result = compare_resident_graph_capture_signature(
                signature, request
            )
        finally:
            object.__setattr__(parent, attribute, original)
        assert result.reason is GraphCaptureDriftReason.PRIMARY_ARRAYS, path


@pytest.mark.warp
def test_signature_tracks_every_published_resource_view_identity(
    resident_request: object,
) -> None:
    """Every required process and communication view contributes to drift."""
    request = cast("ResidentSimulationRequest", resident_request)
    leaf_paths = (
        "condensation.state.scratch_buffers",
        "coagulation.resources.collision_pairs",
        "coagulation.resources.n_collisions",
        "nucleation.resources.scratch",
        "nucleation.resources.finalized_demand",
        "nucleation.resources.diagnostics",
        "nucleation.resources.exhaustion",
        "communication.resources.buffers",
        "communication.resources.execution_state",
        "communication.resources.final_volumes",
    )
    carrier_paths = (
        "coagulation.resources",
        "wall_loss.resources",
        "nucleation.resources",
        "communication.resources",
    )

    for path in leaf_paths + carrier_paths:
        signature = create_resident_graph_capture_signature(request)
        parent, attribute = _path_parent(request, path)
        original = getattr(parent, attribute)
        replacement = copy.copy(original) if path in carrier_paths else object()
        object.__setattr__(parent, attribute, replacement)
        try:
            result = compare_resident_graph_capture_signature(
                signature, request
            )
        finally:
            object.__setattr__(parent, attribute, original)
        assert result.reason is GraphCaptureDriftReason.RESOURCE_VIEWS, path


def _sidecar_leaves(value: object) -> list[tuple[object, str]]:
    """Return mutable owner/field pairs for every nested dataclass leaf."""
    if not is_dataclass(value):
        return []
    leaves: list[tuple[object, str]] = []
    for field in fields(value):
        child = getattr(value, field.name)
        if is_dataclass(child):
            leaves.extend(_sidecar_leaves(child))
        else:
            leaves.append((value, field.name))
    return leaves


@pytest.mark.warp
def test_signature_tracks_nested_dispatch_sidecar_array_identities(
    resident_request: object,
) -> None:
    """Replacing any nested sidecar under a retained wrapper causes drift."""
    request = cast("ResidentSimulationRequest", resident_request)
    communication_request = cast(
        "ResidentCommunicationRequest", request.communication
    )
    communication = communication_request.resources
    assert communication is not None
    roots = (
        request.condensation.state.scratch_buffers,
        request.nucleation.resources.scratch,
        request.nucleation.resources.finalized_demand,
        request.nucleation.resources.diagnostics,
        request.nucleation.resources.exhaustion,
        communication.buffers,
        communication.execution_state,
    )
    control = create_resident_graph_capture_signature(request)
    assert compare_resident_graph_capture_signature(control, request).compatible

    for root in roots:
        for owner, attribute in _sidecar_leaves(root):
            signature = create_resident_graph_capture_signature(request)
            original = getattr(owner, attribute)
            object.__setattr__(owner, attribute, object())
            try:
                result = compare_resident_graph_capture_signature(
                    signature, request
                )
            finally:
                object.__setattr__(owner, attribute, original)
            assert result.reason is GraphCaptureDriftReason.RESOURCE_VIEWS, (
                type(root).__name__,
                type(owner).__name__,
                attribute,
            )


@pytest.mark.warp
@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("request", GraphCaptureDriftReason.REQUEST),
        ("session", GraphCaptureDriftReason.SESSION),
        ("device", GraphCaptureDriftReason.DEVICE),
        ("dimensions", GraphCaptureDriftReason.DIMENSIONS),
        (
            "primary_containers",
            GraphCaptureDriftReason.PRIMARY_CONTAINERS,
        ),
    ],
)
def test_signature_reports_isolated_identity_group_drift(
    resident_request: object,
    field: str,
    reason: GraphCaptureDriftReason,
) -> None:
    """An isolated replacement reports each early signature group in order."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    replacement: object = object()
    if field == "primary_containers":
        replacement = (replacement,)
    object.__setattr__(signature, field, replacement)

    result = compare_resident_graph_capture_signature(signature, request)

    assert result.compatible is False
    assert result.reason is reason


@pytest.mark.warp
def test_signature_reports_first_reason_when_multiple_groups_drift(
    resident_request: object,
) -> None:
    """The earliest documented drift group wins when several groups change."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    object.__setattr__(signature, "request", object())
    object.__setattr__(signature, "device", object())

    result = compare_resident_graph_capture_signature(signature, request)

    assert result.compatible is False
    assert result.reason is GraphCaptureDriftReason.REQUEST


@pytest.mark.warp
def test_signature_ignores_payload_and_rng_words_when_identities_are_stable(
    resident_request: object,
) -> None:
    """Changing published payload values does not affect identity metadata."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    rng_states = request.coagulation.resources.rng_states
    rng_states.assign(np.full(rng_states.shape, 13, dtype=np.uint32))
    request.coagulation.resources.n_collisions.assign(
        np.full(
            request.coagulation.resources.n_collisions.shape,
            2,
            dtype=np.int32,
        )
    )

    result = compare_resident_graph_capture_signature(signature, request)

    assert result.compatible is True
    assert result.reason is None


@pytest.mark.warp
@pytest.mark.parametrize("owner", ["coagulation", "wall_loss"])
def test_signature_reports_rng_resource_after_resource_views(
    resident_request: object,
    owner: str,
) -> None:
    """Replacing either RNG array reports its dedicated final drift group."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    resources = getattr(request, owner).resources
    object.__setattr__(resources, "rng_states", object())

    result = compare_resident_graph_capture_signature(signature, request)

    assert result.compatible is False
    assert result.reason is GraphCaptureDriftReason.RNG_RESOURCES


def test_signature_rejects_inexact_request_without_attribute_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inexact requests reject before lazy imports or arbitrary attributes."""
    calls: list[str] = []

    class AttributeTrap:
        """Fail if request validation accesses arbitrary metadata."""

        def __getattribute__(self, name: str) -> object:
            calls.append(name)
            raise AssertionError("inexact request attributes must not be read")

    monkeypatch.setattr(
        graph_capture,
        "_resident_request_type",
        lambda: (_ for _ in ()).throw(AssertionError("lazy import occurred")),
    )

    with pytest.raises(TypeError, match="exact ResidentSimulationRequest"):
        create_resident_graph_capture_signature(AttributeTrap())

    assert calls == []


def test_compare_rejects_inexact_signature_before_request_access() -> None:
    """Invalid signatures reject before reading or acting on request payloads."""
    calls: list[str] = []

    class RequestActionTrap:
        """Fail if comparison touches a rejected request or its operations."""

        def __getattribute__(self, name: str) -> object:
            calls.append(name)
            raise AssertionError("rejected request must remain untouched")

    with pytest.raises(TypeError, match="exact ResidentGraphCaptureSignature"):
        compare_resident_graph_capture_signature(
            cast("ResidentGraphCaptureSignature", object()), RequestActionTrap()
        )

    assert calls == []


def test_signature_carrier_requires_exact_identity_metadata() -> None:
    """Signature carriers reject malformed field containers at construction."""
    values: tuple[object, ...] = (object(),) * 14
    with pytest.raises(TypeError):
        ResidentGraphCaptureSignature(*values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "primary_containers",
        "primary_arrays",
        "resource_views",
        "graph",
        "schedule",
        "schedule_order",
        "diagnostics",
        "communication",
        "configurations",
        "rng_resources",
    ],
)
def test_signature_carrier_requires_exact_tuple_for_each_group(
    field: str,
) -> None:
    """Every grouped signature field independently enforces tuple metadata."""
    values = {
        "request": object(),
        "session": object(),
        "device": object(),
        "dimensions": object(),
        "primary_containers": (),
        "primary_arrays": (),
        "resource_views": (),
        "graph": (),
        "schedule": (),
        "schedule_order": (),
        "diagnostics": (),
        "communication": (),
        "configurations": (),
        "rng_resources": (),
    }
    values[field] = []

    with pytest.raises(TypeError, match=rf"{field} must be an exact tuple"):
        ResidentGraphCaptureSignature(**values)  # type: ignore[arg-type]


def test_capability_and_compatibility_are_identity_carriers() -> None:
    """Equal-content result carriers intentionally preserve identity semantics."""
    device = Device(Backend.CPU, "cpu")
    first = GraphCaptureCapability(
        device, GraphCaptureAvailability.UNSUPPORTED_CPU
    )
    second = GraphCaptureCapability(
        device, GraphCaptureAvailability.UNSUPPORTED_CPU
    )

    assert first is not second
    assert first != second
    assert GraphCaptureCompatibility(True, None).reason is None
    assert (
        GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.REQUEST
        ).compatible
        is False
    )
    with pytest.raises(ValueError, match="if and only if"):
        GraphCaptureCompatibility(True, GraphCaptureDriftReason.REQUEST)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((object(), GraphCaptureAvailability.AVAILABLE), "exact Device"),
        (
            (Device(Backend.CPU, "cpu"), "available"),
            "exact GraphCaptureAvailability",
        ),
    ],
)
def test_capability_carrier_rejects_inexact_fields(
    args: tuple[object, object], error: str
) -> None:
    """Capability fields require their exact declaration types."""
    with pytest.raises(TypeError, match=error):
        GraphCaptureCapability(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("compatible", "reason", "exception"),
    [
        (1, None, TypeError),
        (True, "request", TypeError),
        (False, None, ValueError),
        (True, GraphCaptureDriftReason.REQUEST, ValueError),
    ],
)
def test_compatibility_carrier_rejects_invalid_invariants(
    compatible: object, reason: object, exception: type[Exception]
) -> None:
    """Compatibility requires exact types and a consistent drift reason."""
    with pytest.raises(exception):
        GraphCaptureCompatibility(compatible, reason)  # type: ignore[arg-type]


def test_graph_capture_names_remain_direct_import_only() -> None:
    """Concrete graph-capture declarations do not alter frozen exports."""
    names = (
        "GraphCaptureAvailability",
        "GraphCaptureCapability",
        "GraphCaptureRuntimeProbe",
        "GraphCaptureDriftReason",
        "GraphCaptureCompatibility",
        "ResidentGraphCaptureSignature",
        "GraphCaptureLifecycleState",
        "GraphCaptureFailureClassification",
        "GraphCaptureLifecycle",
        "resolve_graph_capture_capability",
        "create_resident_graph_capture_signature",
        "compare_resident_graph_capture_signature",
        "create_graph_capture_lifecycle",
        "complete_graph_capture",
        "invalidate_graph_capture",
        "classify_graph_capture_failure",
        "retire_graph_capture",
        "renew_retired_graph_capture",
        "close_graph_capture",
        "CapturedResidentGraph",
        "capture_prepared_resident_graph",
    )
    for name in names:
        assert name not in execution.__all__
        assert not hasattr(execution, name)
        assert not hasattr(particula, name)


def _available_lifecycle() -> GraphCaptureLifecycle:
    """Create minimal host-only lifecycle metadata with an available capability."""
    signature = ResidentGraphCaptureSignature(
        object(),
        object(),
        object(),
        object(),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
    )
    return create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"), GraphCaptureAvailability.AVAILABLE
        ),
        signature,
    )


def test_lifecycle_successful_transitions_retain_p1_carrier_identities() -> (
    None
):
    """Successful lifecycle successors retain immutable P1 metadata identities."""
    ready = _available_lifecycle()
    captured = complete_graph_capture(ready)
    invalidated = invalidate_graph_capture(
        captured,
        GraphCaptureCompatibility(False, GraphCaptureDriftReason.REQUEST),
    )
    retired = retire_graph_capture(invalidated)
    renewed = renew_retired_graph_capture(
        retired, _available_lifecycle().signature
    )
    faulted = classify_graph_capture_failure(
        captured, GraphCaptureFailureClassification.WRITER_MAY_HAVE_LAUNCHED
    )

    assert ready.state is GraphCaptureLifecycleState.READY
    assert captured.state is GraphCaptureLifecycleState.CAPTURED
    assert invalidated.state is GraphCaptureLifecycleState.INVALIDATED
    assert retired.state is GraphCaptureLifecycleState.RETIRED
    assert renewed.state is GraphCaptureLifecycleState.READY
    assert faulted.state is GraphCaptureLifecycleState.FAULTED
    for successor in (captured, invalidated, retired, faulted):
        assert successor.capability is ready.capability
        assert successor.signature is ready.signature
    assert renewed.capability is retired.capability
    assert renewed.signature is not retired.signature
    assert renewed.first_invalidation_reason is None
    assert retired.first_invalidation_reason is GraphCaptureDriftReason.REQUEST


@pytest.mark.parametrize(
    "availability",
    [
        GraphCaptureAvailability.UNSUPPORTED_CPU,
        GraphCaptureAvailability.UNSUPPORTED_WARP_CPU,
        GraphCaptureAvailability.UNAVAILABLE_RUNTIME,
        GraphCaptureAvailability.UNAVAILABLE_DEVICE,
        GraphCaptureAvailability.UNSUPPORTED_API,
    ],
)
def test_create_lifecycle_rejects_nonavailable_capability(
    availability: GraphCaptureAvailability,
) -> None:
    """Only an exact available capability can start a lifecycle."""
    lifecycle = _available_lifecycle()
    capability = GraphCaptureCapability(
        Device(Backend.WARP, "cuda:0"), availability
    )

    with pytest.raises(ValueError, match="available"):
        create_graph_capture_lifecycle(capability, lifecycle.signature)


@pytest.mark.parametrize(
    ("state", "reason", "exception"),
    [
        (
            GraphCaptureLifecycleState.READY,
            GraphCaptureDriftReason.REQUEST,
            ValueError,
        ),
        (
            GraphCaptureLifecycleState.CAPTURED,
            GraphCaptureDriftReason.REQUEST,
            ValueError,
        ),
        (GraphCaptureLifecycleState.INVALIDATED, None, ValueError),
        (GraphCaptureLifecycleState.RETIRED, None, ValueError),
        (GraphCaptureLifecycleState.FAULTED, None, None),
        (
            GraphCaptureLifecycleState.RETIRED,
            GraphCaptureDriftReason.REQUEST,
            None,
        ),
        (GraphCaptureLifecycleState.CLOSED, None, None),
    ],
)
def test_lifecycle_reason_state_invariants(
    state: GraphCaptureLifecycleState,
    reason: GraphCaptureDriftReason | None,
    exception: type[Exception] | None,
) -> None:
    """Lifecycle construction accepts only documented reason-state combinations."""
    lifecycle = _available_lifecycle()
    if exception is None:
        result = GraphCaptureLifecycle(
            lifecycle.capability, lifecycle.signature, state, reason
        )
        assert result.state is state
    else:
        with pytest.raises(exception):
            GraphCaptureLifecycle(
                lifecycle.capability, lifecycle.signature, state, reason
            )


def test_direct_lifecycle_construction_requires_available_capability() -> None:
    """Every direct lifecycle state rejects unavailable capability metadata."""
    lifecycle = _available_lifecycle()
    capability = GraphCaptureCapability(
        Device(Backend.WARP, "cuda:0"),
        GraphCaptureAvailability.UNAVAILABLE_DEVICE,
    )

    with pytest.raises(ValueError, match="capability must be available"):
        GraphCaptureLifecycle(
            capability,
            lifecycle.signature,
            GraphCaptureLifecycleState.READY,
            None,
        )


def test_invalidation_and_failure_paths_preserve_first_reason_and_identity() -> (
    None
):
    """Compatible and repeated paths preserve captured metadata by identity."""
    captured = complete_graph_capture(_available_lifecycle())
    assert (
        invalidate_graph_capture(
            captured, GraphCaptureCompatibility(True, None)
        )
        is captured
    )
    invalidated = invalidate_graph_capture(
        captured,
        GraphCaptureCompatibility(False, GraphCaptureDriftReason.REQUEST),
    )
    repeated = invalidate_graph_capture(
        invalidated,
        GraphCaptureCompatibility(False, GraphCaptureDriftReason.SESSION),
    )
    assert repeated is invalidated
    assert repeated.first_invalidation_reason is GraphCaptureDriftReason.REQUEST
    assert (
        invalidate_graph_capture(
            invalidated, GraphCaptureCompatibility(True, None)
        )
        is invalidated
    )
    assert (
        classify_graph_capture_failure(
            invalidated, GraphCaptureFailureClassification.READ_ONLY
        )
        is invalidated
    )
    faulted = classify_graph_capture_failure(
        invalidated, GraphCaptureFailureClassification.WRITER_MAY_HAVE_LAUNCHED
    )
    assert faulted.first_invalidation_reason is GraphCaptureDriftReason.REQUEST
    for classification in GraphCaptureFailureClassification:
        assert (
            classify_graph_capture_failure(faulted, classification) is faulted
        )


def test_retirement_is_idempotent_and_retains_first_invalidation_reason() -> (
    None
):
    """Repeated retirement preserves the original retired lifecycle record."""
    captured = complete_graph_capture(_available_lifecycle())
    invalidated = invalidate_graph_capture(
        captured,
        GraphCaptureCompatibility(False, GraphCaptureDriftReason.REQUEST),
    )
    retired = retire_graph_capture(invalidated)

    repeated = retire_graph_capture(retired)

    assert repeated is retired
    assert (
        repeated.first_invalidation_reason
        is invalidated.first_invalidation_reason
    )


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (GraphCaptureLifecycleState.READY, None),
        (GraphCaptureLifecycleState.CAPTURED, None),
        (
            GraphCaptureLifecycleState.INVALIDATED,
            GraphCaptureDriftReason.REQUEST,
        ),
    ],
)
def test_read_only_failure_is_an_identity_no_op_from_every_open_state(
    state: GraphCaptureLifecycleState,
    reason: GraphCaptureDriftReason | None,
) -> None:
    """Read-only failures preserve every accepted pre-failure lifecycle."""
    base = _available_lifecycle()
    lifecycle = GraphCaptureLifecycle(
        base.capability,
        base.signature,
        state,
        reason,
    )

    result = classify_graph_capture_failure(
        lifecycle,
        GraphCaptureFailureClassification.READ_ONLY,
    )

    assert result is lifecycle


def test_writer_failure_faults_ready_lifecycle_without_a_reason() -> None:
    """Writer-capable failure faults ready metadata without inventing drift."""
    ready = _available_lifecycle()

    faulted = classify_graph_capture_failure(
        ready,
        GraphCaptureFailureClassification.WRITER_MAY_HAVE_LAUNCHED,
    )

    assert faulted is not ready
    assert faulted.state is GraphCaptureLifecycleState.FAULTED
    assert faulted.first_invalidation_reason is None
    assert faulted.capability is ready.capability
    assert faulted.signature is ready.signature


@pytest.mark.parametrize(
    "state",
    [
        GraphCaptureLifecycleState.READY,
        GraphCaptureLifecycleState.CAPTURED,
        GraphCaptureLifecycleState.INVALIDATED,
        GraphCaptureLifecycleState.FAULTED,
        GraphCaptureLifecycleState.RETIRED,
        GraphCaptureLifecycleState.CLOSED,
    ],
)
def test_close_transitions_every_open_state_and_is_idempotent(
    state: GraphCaptureLifecycleState,
) -> None:
    """Close accepts every lifecycle state and preserves retained reason metadata."""
    base = _available_lifecycle()
    reason = (
        GraphCaptureDriftReason.REQUEST
        if state
        in (
            GraphCaptureLifecycleState.INVALIDATED,
            GraphCaptureLifecycleState.FAULTED,
            GraphCaptureLifecycleState.RETIRED,
        )
        else None
    )
    lifecycle = GraphCaptureLifecycle(
        base.capability, base.signature, state, reason
    )
    closed = close_graph_capture(lifecycle)

    assert closed.state is GraphCaptureLifecycleState.CLOSED
    assert closed.first_invalidation_reason is reason
    assert close_graph_capture(closed) is closed


@pytest.mark.parametrize(
    ("operation", "states"),
    [
        (
            "complete",
            ("captured", "invalidated", "faulted", "retired", "closed"),
        ),
        ("invalidate", ("ready", "faulted", "retired", "closed")),
        ("classify", ("retired", "closed")),
        ("retire", ("ready", "captured", "faulted", "closed")),
        ("renew", ("ready", "captured", "invalidated", "faulted", "closed")),
    ],
)
def test_lifecycle_operations_reject_illegal_source_states(
    operation: str, states: tuple[str, ...]
) -> None:
    """Each transition operation rejects every source state outside its table."""
    base = _available_lifecycle()
    for state_name in states:
        state = GraphCaptureLifecycleState(state_name)
        reason = (
            GraphCaptureDriftReason.REQUEST
            if state
            in (
                GraphCaptureLifecycleState.INVALIDATED,
                GraphCaptureLifecycleState.FAULTED,
                GraphCaptureLifecycleState.RETIRED,
            )
            else None
        )
        lifecycle = GraphCaptureLifecycle(
            base.capability, base.signature, state, reason
        )
        with pytest.raises(ValueError):
            if operation == "complete":
                complete_graph_capture(lifecycle)
            elif operation == "invalidate":
                invalidate_graph_capture(
                    lifecycle, GraphCaptureCompatibility(True, None)
                )
            elif operation == "classify":
                classify_graph_capture_failure(
                    lifecycle, GraphCaptureFailureClassification.READ_ONLY
                )
            elif operation == "retire":
                retire_graph_capture(lifecycle)
            else:
                renew_retired_graph_capture(lifecycle, base.signature)


def test_lifecycle_operations_reject_inexact_carriers_before_attribute_access() -> (
    None
):
    """Every lifecycle operation rejects inexact inputs without attribute reads."""

    class AttributeTrap:
        """Fail if a rejected carrier attribute is read."""

        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"unexpected attribute access: {name}")

    lifecycle = _available_lifecycle()
    with pytest.raises(TypeError):
        create_graph_capture_lifecycle(
            cast(GraphCaptureCapability, AttributeTrap()), lifecycle.signature
        )
    with pytest.raises(TypeError):
        create_graph_capture_lifecycle(
            lifecycle.capability,
            cast(ResidentGraphCaptureSignature, AttributeTrap()),
        )
    with pytest.raises(TypeError):
        complete_graph_capture(cast(GraphCaptureLifecycle, AttributeTrap()))
    with pytest.raises(TypeError):
        invalidate_graph_capture(
            cast(GraphCaptureLifecycle, AttributeTrap()),
            GraphCaptureCompatibility(True, None),
        )
    with pytest.raises(TypeError):
        invalidate_graph_capture(
            lifecycle, cast(GraphCaptureCompatibility, AttributeTrap())
        )
    with pytest.raises(TypeError):
        classify_graph_capture_failure(
            cast(GraphCaptureLifecycle, AttributeTrap()),
            GraphCaptureFailureClassification.READ_ONLY,
        )
    with pytest.raises(TypeError):
        classify_graph_capture_failure(
            lifecycle, cast(GraphCaptureFailureClassification, AttributeTrap())
        )
    with pytest.raises(TypeError):
        renew_retired_graph_capture(
            cast(GraphCaptureLifecycle, AttributeTrap()), lifecycle.signature
        )
    with pytest.raises(TypeError):
        renew_retired_graph_capture(
            lifecycle, cast(ResidentGraphCaptureSignature, AttributeTrap())
        )
    with pytest.raises(TypeError):
        close_graph_capture(cast(GraphCaptureLifecycle, AttributeTrap()))


def test_binary_lifecycle_apis_reject_left_argument_before_second_access() -> (
    None
):
    """Binary lifecycle APIs reject their first invalid carrier left-to-right."""
    accesses: list[str] = []

    class AttributeTrap:
        """Record and reject any attribute access to an invalid carrier."""

        def __init__(self, label: str) -> None:
            """Store the label used to identify this invalid carrier."""
            self.label = label

        def __getattribute__(self, name: str) -> object:
            if name == "label":
                return object.__getattribute__(self, name)
            accesses.append(object.__getattribute__(self, "label"))
            raise AssertionError(f"unexpected attribute access: {name}")

    with pytest.raises(TypeError):
        create_graph_capture_lifecycle(
            cast(GraphCaptureCapability, AttributeTrap("first")),
            cast(ResidentGraphCaptureSignature, AttributeTrap("second")),
        )
    with pytest.raises(TypeError):
        invalidate_graph_capture(
            cast(GraphCaptureLifecycle, AttributeTrap("first")),
            cast(GraphCaptureCompatibility, AttributeTrap("second")),
        )
    with pytest.raises(TypeError):
        classify_graph_capture_failure(
            cast(GraphCaptureLifecycle, AttributeTrap("first")),
            cast(GraphCaptureFailureClassification, AttributeTrap("second")),
        )
    with pytest.raises(TypeError):
        renew_retired_graph_capture(
            cast(GraphCaptureLifecycle, AttributeTrap("first")),
            cast(ResidentGraphCaptureSignature, AttributeTrap("second")),
        )

    assert accesses == []


@pytest.mark.parametrize(
    "field_values",
    [
        (
            object(),
            _available_lifecycle().signature,
            GraphCaptureLifecycleState.READY,
            None,
        ),
        (
            _available_lifecycle().capability,
            object(),
            GraphCaptureLifecycleState.READY,
            None,
        ),
        (
            _available_lifecycle().capability,
            _available_lifecycle().signature,
            "ready",
            None,
        ),
        (
            _available_lifecycle().capability,
            _available_lifecycle().signature,
            GraphCaptureLifecycleState.READY,
            "request",
        ),
    ],
)
def test_lifecycle_carrier_rejects_inexact_fields(
    field_values: tuple[object, object, object, object],
) -> None:
    """Lifecycle construction requires exact carriers and enumerations."""
    with pytest.raises(TypeError):
        GraphCaptureLifecycle(*field_values)  # type: ignore[arg-type]


def test_lifecycle_sequence_does_not_import_warp_or_resident_boundaries() -> (
    None
):
    """Legal host metadata transitions retain the declaration-only import boundary."""
    script = textwrap.dedent(
        """
        import builtins
        import importlib
        import sys

        blocked = ("warp", "particula.gpu", "particula.execution.gpu_session",
                   "particula.execution.gpu_resources",
                   "particula.execution.resident_scheduler")
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if any(name == item or name.startswith(item + ".") for item in blocked):
                raise AssertionError(f"forbidden import: {name}")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import
        module = importlib.import_module("particula.execution.graph_capture")
        from particula.execution import Backend, Device
        signature = module.ResidentGraphCaptureSignature(*((object(),) * 4), *(((),) * 10))
        capability = module.GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"), module.GraphCaptureAvailability.AVAILABLE)
        ready = module.create_graph_capture_lifecycle(capability, signature)
        captured = module.complete_graph_capture(ready)
        invalidated = module.invalidate_graph_capture(
            captured, module.GraphCaptureCompatibility(False, module.GraphCaptureDriftReason.REQUEST))
        assert module.retire_graph_capture(invalidated).state is module.GraphCaptureLifecycleState.RETIRED
        assert all(name not in sys.modules for name in blocked)
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_graph_capture_import_does_not_import_warp() -> None:
    """The declaration module itself has no optional Warp import dependency."""
    script = textwrap.dedent(
        """
        import builtins
        import importlib
        import sys

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "warp" or name.startswith("warp."):
                raise AssertionError(f"forbidden import: {name}")
            if name == "particula.gpu" or name.startswith("particula.gpu."):
                raise AssertionError(f"forbidden import: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        importlib.import_module("particula.execution.graph_capture")
        assert "particula.gpu" not in sys.modules
        assert "warp" not in sys.modules
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.warp
def test_binding_attaches_once_to_the_final_request(
    resident_request: object,
) -> None:
    """A binding retains and attaches exactly one final request identity."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )

    _attach_resident_graph_capture_binding(request, binding)

    assert request.graph_capture_binding is binding
    assert binding.lifecycle is lifecycle
    with pytest.raises(ValueError, match="already has"):
        _attach_resident_graph_capture_binding(request, binding)


@pytest.mark.warp
def test_binding_rejects_a_swapped_equivalent_guard(
    resident_request: object,
) -> None:
    """A binding rejects a distinct guard even when it has the same owners."""
    from particula.execution.gpu_session import ResidentStepGuard

    request = cast("ResidentSimulationRequest", resident_request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        create_resident_graph_capture_signature(request),
    )

    with pytest.raises(ValueError, match="identities do not match"):
        ResidentGraphCaptureBinding(
            request,
            request.session,
            cast("GPUResourceRegistry", request.registry),
            ResidentStepGuard(
                request.session, cast("GPUResourceRegistry", request.registry)
            ),
            lifecycle,
        )

    assert request.graph_capture_binding is None


@pytest.mark.warp
def test_binding_renewal_rejects_a_signature_for_another_request(
    resident_request: object,
) -> None:
    """Renewal preserves the binding's final request and session identities."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    complete_resident_graph_capture(binding)
    binding._lifecycle = invalidate_graph_capture(
        binding.lifecycle,
        GraphCaptureCompatibility(False, GraphCaptureDriftReason.REQUEST),
    )
    retire_resident_graph_capture(binding)
    foreign_signature = copy.copy(signature)
    object.__setattr__(foreign_signature, "request", object())

    with pytest.raises(ValueError, match="exact binding"):
        renew_resident_graph_capture(binding, foreign_signature)

    assert binding.lifecycle.state is GraphCaptureLifecycleState.RETIRED


@pytest.mark.warp
def test_binding_completion_updates_only_its_lifecycle(
    resident_request: object,
) -> None:
    """Binding completion retains its resident carriers and captures metadata."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)

    captured = complete_resident_graph_capture(binding)

    assert captured is binding.lifecycle
    assert captured.state is GraphCaptureLifecycleState.CAPTURED
    assert captured.capability is lifecycle.capability
    assert captured.signature is lifecycle.signature


@pytest.mark.warp
def test_binding_completion_rejects_detached_binding_without_mutation(
    resident_request: object,
) -> None:
    """Completion requires the binding's prior exact request attachment."""
    request = cast("ResidentSimulationRequest", resident_request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        create_resident_graph_capture_signature(request),
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )

    with pytest.raises(ValueError, match="attachment does not match"):
        complete_resident_graph_capture(binding)

    assert binding.lifecycle is lifecycle
    _attach_resident_graph_capture_binding(request, binding)
    assert complete_resident_graph_capture(binding).state is (
        GraphCaptureLifecycleState.CAPTURED
    )


@pytest.mark.warp
def test_binding_close_is_owned_validated_and_idempotent(
    resident_request: object,
) -> None:
    """Attached binding closure updates only its lifecycle and repeats by identity."""
    request = cast("ResidentSimulationRequest", resident_request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        create_resident_graph_capture_signature(request),
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    complete_resident_graph_capture(binding)

    closed = close_resident_graph_capture(binding)

    assert closed.state is GraphCaptureLifecycleState.CLOSED
    assert close_resident_graph_capture(binding) is closed
    with pytest.raises(ValueError, match="terminal state"):
        classify_resident_graph_capture_writer_failure(binding)


@pytest.mark.warp
def test_binding_close_rejects_detached_or_stale_ownership(
    resident_request: object,
) -> None:
    """Binding-owned closure rejects missing attachment and stale ownership."""
    request = cast("ResidentSimulationRequest", resident_request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        create_resident_graph_capture_signature(request),
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    with pytest.raises(ValueError, match="attachment does not match"):
        close_resident_graph_capture(binding)

    _attach_resident_graph_capture_binding(request, binding)
    object.__setattr__(request, "graph_capture_binding", None)
    with pytest.raises(ValueError, match="attachment does not match"):
        close_resident_graph_capture(binding)
    assert binding.lifecycle is lifecycle


@pytest.mark.warp
def test_binding_gate_rejects_warp_cpu_before_dispatch(
    resident_request: object,
) -> None:
    """A captured binding still rejects the unsupported Warp CPU resident."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cpu"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    complete_resident_graph_capture(binding)

    with pytest.raises(ValueError, match="CUDA"):
        gate_resident_graph_capture(binding)

    assert binding.lifecycle.state is GraphCaptureLifecycleState.CAPTURED


@pytest.mark.warp
def test_binding_writer_classification_faults_captured_lifecycle(
    resident_request: object,
) -> None:
    """A scheduler-confirmed writer outcome faults only binding metadata."""
    request = cast("ResidentSimulationRequest", resident_request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        create_resident_graph_capture_signature(request),
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)

    classify_resident_graph_capture_writer_failure(binding)

    assert binding.lifecycle.state is GraphCaptureLifecycleState.FAULTED
    assert binding.lifecycle.signature is lifecycle.signature


@pytest.mark.warp
def test_binding_renews_only_after_explicit_retirement(
    resident_request: object,
) -> None:
    """A binding renews retired metadata and still requires completion."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    complete_resident_graph_capture(binding)
    binding._lifecycle = invalidate_graph_capture(
        binding.lifecycle,
        GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.CONFIGURATIONS
        ),
    )

    retired = retire_resident_graph_capture(binding)
    renewed = renew_resident_graph_capture(binding, signature)

    assert retired.state is GraphCaptureLifecycleState.RETIRED
    assert renewed.state is GraphCaptureLifecycleState.READY
    assert complete_resident_graph_capture(binding).state is (
        GraphCaptureLifecycleState.CAPTURED
    )


@pytest.mark.warp
def test_qualification_rejects_warp_cpu_without_native_activity(
    resident_request: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualification fails closed before probing or acting on Warp CPU."""
    from particula.execution.resident_communication import (
        PreparedResidentCommunicationBinding,
    )
    from particula.execution.resident_enqueue import prepare_resident_timestep
    from particula.execution.resident_scheduler import (
        PreparedResidentSimulation,
    )
    from particula.execution.thermodynamic_updates import (
        PreparedResidentThermodynamicSequence,
    )

    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    capability = GraphCaptureCapability(
        request.session.metadata.device,
        GraphCaptureAvailability.AVAILABLE,
    )
    lifecycle = create_graph_capture_lifecycle(capability, signature)
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    registry = cast("GPUResourceRegistry", request.registry)
    capture_set = registry.validate_capture_resource_set(
        cast(
            "CaptureResourceRequirements",
            request.capture_resource_requirements,
        )
    )
    timestep = prepare_resident_timestep(request, 0.0)
    prepared = PreparedResidentSimulation(
        timestep=timestep,
        request=request,
        session=request.session,
        registry=request.registry,
        guard=request.guard,
        lifecycle=lifecycle,
        signature=signature,
        graph=request.graph,
        schedule=request.schedule,
        ordered_node_ids=request.schedule.ordered_node_ids,
        primary_arrays=signature.primary_arrays,
        resource_views=signature.resource_views,
        capture_requirements=request.capture_resource_requirements,
        capture_set=capture_set,
        capture_report=capture_set.report,
        nodes=(),
        thermal=cast("PreparedResidentThermodynamicSequence", object()),
        communication=cast("PreparedResidentCommunicationBinding", object()),
        environment=object(),
        gas=object(),
        condensation=object(),
        coagulation=object(),
        dilution=object(),
        wall_loss=object(),
        nucleation=object(),
        diagnostics=object(),
        operations=(),
        duration=timestep.duration,
    )

    monkeypatch.setattr(
        request.guard,
        "begin_step",
        lambda: pytest.fail("qualification must not open a resident step"),
    )

    with pytest.raises(
        ValueError, match="prepared resident simulation identities"
    ):
        qualify_prepared_resident_graph_capture(
            binding,
            prepared,
            capture_set,
            cast("GraphCaptureRuntimeAdapter", AttributeTrapProbe()),
        )

    assert request.guard.completed_steps == 0
    assert binding.lifecycle is lifecycle
    assert binding.lifecycle.state is GraphCaptureLifecycleState.READY


@pytest.mark.warp
def test_qualification_retains_native_vocabulary_without_invoking_it(
    resident_request: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """READY qualification retains one adapter vocabulary by identity only."""
    from particula.execution.resident_scheduler import (
        prepare_resident_simulation,
    )

    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    capability = GraphCaptureCapability(
        request.session.metadata.device,
        GraphCaptureAvailability.AVAILABLE,
    )
    lifecycle = create_graph_capture_lifecycle(capability, signature)
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)

    class PreparedCall:
        """Provide a retained write-free operation for preparation."""

        def execute(self) -> None:
            return None

    class PreparedAdapter:
        """Replace fixture adapters with the scheduler preparation protocol."""

        def prepare(self, _request: object) -> PreparedCall:
            return PreparedCall()

    from particula.execution import resident_scheduler

    for name in (
        "WarpCondensationExecutionAdapter",
        "ResidentBrownianCoagulationExecutionAdapter",
        "ResidentDilutionAdapter",
        "ResidentWallLossAdapter",
        "ResidentNucleationAdapter",
    ):
        monkeypatch.setattr(resident_scheduler, name, PreparedAdapter)

    prepared = prepare_resident_simulation(request, 0.0)
    capture_set = cast(
        "GPUResourceRegistry", request.registry
    ).validate_capture_resource_set(
        cast(
            "CaptureResourceRequirements", request.capture_resource_requirements
        )
    )

    cuda_device = Device(Backend.WARP, "cuda:0")
    object.__setattr__(request.session.metadata, "device", cuda_device)
    object.__setattr__(signature, "device", cuda_device)
    object.__setattr__(capability, "device", cuda_device)
    monkeypatch.setattr(
        request.registry, "validate_pinned_session", lambda _: None
    )

    calls: list[str] = []

    def native_callable() -> None:
        calls.append("native")

    vocabulary = GraphCaptureNativeCallables(
        native_callable,
        native_callable,
        native_callable,
        native_callable,
        native_callable,
    )

    class RecordingAdapter:
        """Provide ordered availability probes and a native vocabulary."""

        def runtime_available(self) -> bool:
            calls.append("runtime")
            return True

        def device_available(self, device: Device) -> bool:
            assert device is cuda_device
            calls.append("device")
            return True

        def capture_api_available(self, device: Device) -> bool:
            assert device is cuda_device
            calls.append("api")
            return True

        def capture_callables(
            self, device: Device
        ) -> GraphCaptureNativeCallables:
            assert device is cuda_device
            calls.append("callables")
            return vocabulary

    result = qualify_prepared_resident_graph_capture(
        binding,
        prepared,
        capture_set,
        RecordingAdapter(),
    )

    assert result.binding is binding
    assert result.lifecycle is lifecycle
    assert result.signature is signature
    assert result.prepared is prepared
    assert result.capture_set is capture_set
    assert result.native_callables is vocabulary
    assert binding.lifecycle is lifecycle
    assert binding.lifecycle.state is GraphCaptureLifecycleState.READY
    assert calls == ["runtime", "device", "api", "callables"]


@pytest.mark.warp
def test_qualification_rejects_adapter_lifecycle_reentrancy(
    resident_request: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualification rejects an adapter that changes its retained READY state."""
    from particula.execution.resident_scheduler import (
        prepare_resident_simulation,
    )

    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    capability = GraphCaptureCapability(
        request.session.metadata.device,
        GraphCaptureAvailability.AVAILABLE,
    )
    lifecycle = create_graph_capture_lifecycle(capability, signature)
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)

    class PreparedCall:
        """Provide a retained write-free operation for preparation."""

        def execute(self) -> None:
            return None

    class PreparedAdapter:
        """Replace fixture adapters with the scheduler preparation protocol."""

        def prepare(self, _request: object) -> PreparedCall:
            return PreparedCall()

    from particula.execution import resident_scheduler

    for name in (
        "WarpCondensationExecutionAdapter",
        "ResidentBrownianCoagulationExecutionAdapter",
        "ResidentDilutionAdapter",
        "ResidentWallLossAdapter",
        "ResidentNucleationAdapter",
    ):
        monkeypatch.setattr(resident_scheduler, name, PreparedAdapter)

    prepared = prepare_resident_simulation(request, 0.0)
    capture_set = cast(
        "GPUResourceRegistry", request.registry
    ).validate_capture_resource_set(
        cast(
            "CaptureResourceRequirements", request.capture_resource_requirements
        )
    )
    cuda_device = Device(Backend.WARP, "cuda:0")
    object.__setattr__(request.session.metadata, "device", cuda_device)
    object.__setattr__(signature, "device", cuda_device)
    object.__setattr__(capability, "device", cuda_device)
    monkeypatch.setattr(
        request.registry, "validate_pinned_session", lambda _: None
    )

    calls: list[str] = []

    def native_callable() -> None:
        calls.append("native")

    vocabulary = GraphCaptureNativeCallables(
        native_callable,
        native_callable,
        native_callable,
        native_callable,
        native_callable,
    )

    class ReentrantAdapter:
        """Close the attached binding while its callable vocabulary is resolved."""

        def runtime_available(self) -> bool:
            calls.append("runtime")
            return True

        def device_available(self, device: Device) -> bool:
            assert device is cuda_device
            calls.append("device")
            return True

        def capture_api_available(self, device: Device) -> bool:
            assert device is cuda_device
            calls.append("api")
            return True

        def capture_callables(
            self, device: Device
        ) -> GraphCaptureNativeCallables:
            assert device is cuda_device
            calls.append("callables")
            close_resident_graph_capture(binding)
            return vocabulary

    with pytest.raises(ValueError):
        qualify_prepared_resident_graph_capture(
            binding,
            prepared,
            capture_set,
            ReentrantAdapter(),
        )

    assert binding.lifecycle.state is GraphCaptureLifecycleState.CLOSED
    assert calls == ["runtime", "device", "api", "callables"]


@pytest.mark.warp
def test_capture_prepared_graph_dispatches_between_native_capture_boundaries(
    resident_request: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native capture retains its end handle after frozen dispatch succeeds."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    capture_set = cast(
        "GPUResourceRegistry", request.registry
    ).validate_capture_resource_set(
        cast(
            "CaptureResourceRequirements", request.capture_resource_requirements
        )
    )
    trace: list[str] = []
    handle = object()

    def capture_begin() -> None:
        trace.append("begin")

    def capture_end() -> object:
        trace.append("end")
        return handle

    def unexpected_callable() -> None:
        pytest.fail("only capture begin and end belong to this phase")

    def capture_release(_handle: object) -> None:
        trace.append("release")

    qualification = PreparedGraphCaptureQualification(
        binding,
        lifecycle,
        signature,
        request,
        request.session,
        request.registry,
        request.guard,
        object(),
        object(),
        request.capture_resource_requirements,
        capture_set,
        capture_set.report,
        Device(Backend.WARP, "cuda:0"),
        request.session.dimensions,
        request.graph,
        request.schedule,
        request.schedule.ordered_node_ids,
        0.0,
        True,
        signature.primary_arrays,
        signature.resource_views,
        GraphCaptureNativeCallables(
            capture_begin,
            capture_end,
            unexpected_callable,
            unexpected_callable,
            unexpected_callable,
        ),
    )
    monkeypatch.setattr(
        graph_capture,
        "_validate_prepared_graph_capture_qualification",
        lambda value: value,
    )
    import particula.execution.resident_scheduler as resident_scheduler

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        lambda _prepared: trace.append("dispatch"),
    )

    captured = capture_prepared_resident_graph(qualification)

    assert trace == ["begin", "dispatch", "end"]
    assert captured.handle is handle
    assert captured.qualification is qualification
    assert captured.lifecycle is binding.lifecycle
    assert binding.lifecycle.state is GraphCaptureLifecycleState.CAPTURED
    for field in (
        "binding",
        "lifecycle",
        "signature",
        "request",
        "session",
        "registry",
        "guard",
        "prepared",
        "timestep",
        "capture_requirements",
        "capture_set",
        "capture_report",
        "device",
    ):
        with pytest.raises(ValueError, match="identities do not match"):
            replace(captured, **cast(Any, {field: object()}))


@pytest.mark.warp
def test_capture_prepared_graph_ends_and_faults_after_dispatch_failure(
    resident_request: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch failure ends capture and faults its attached lifecycle."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            Device(Backend.WARP, "cuda:0"),
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    capture_set = cast(
        "GPUResourceRegistry", request.registry
    ).validate_capture_resource_set(
        cast(
            "CaptureResourceRequirements", request.capture_resource_requirements
        )
    )
    trace: list[str] = []

    def capture_begin() -> None:
        trace.append("begin")

    def capture_end() -> object:
        trace.append("end")
        return object()

    def unexpected_callable() -> None:
        pytest.fail("only capture begin and end belong to this phase")

    def capture_release(_handle: object) -> None:
        trace.append("release")

    qualification = PreparedGraphCaptureQualification(
        binding,
        lifecycle,
        signature,
        request,
        request.session,
        request.registry,
        request.guard,
        object(),
        object(),
        request.capture_resource_requirements,
        capture_set,
        capture_set.report,
        Device(Backend.WARP, "cuda:0"),
        request.session.dimensions,
        request.graph,
        request.schedule,
        request.schedule.ordered_node_ids,
        0.0,
        True,
        signature.primary_arrays,
        signature.resource_views,
        GraphCaptureNativeCallables(
            capture_begin,
            capture_end,
            unexpected_callable,
            unexpected_callable,
            capture_release,
        ),
    )
    monkeypatch.setattr(
        graph_capture,
        "_validate_prepared_graph_capture_qualification",
        lambda value: value,
    )
    import particula.execution.resident_scheduler as resident_scheduler

    def fail_dispatch(_prepared: object) -> None:
        trace.append("dispatch")
        raise RuntimeError("frozen dispatch failed")

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        fail_dispatch,
    )

    with pytest.raises(RuntimeError, match="frozen dispatch failed"):
        capture_prepared_resident_graph(qualification)

    assert trace == ["begin", "dispatch", "end", "release"]
    assert binding.lifecycle.state is GraphCaptureLifecycleState.FAULTED
