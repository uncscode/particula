"""Test declaration-only graph-capture capability metadata."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, cast

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
    ResidentGraphCaptureSignature,
    compare_resident_graph_capture_signature,
    create_resident_graph_capture_signature,
    resolve_graph_capture_capability,
)

if TYPE_CHECKING:
    from particula.execution.graph_capture import GraphCaptureRuntimeProbe
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
        return self.runtime  # type: ignore[return-value]

    def device_available(self, device: Device) -> bool:
        """Record and return the configured device outcome."""
        del device
        self.calls.append("device")
        return self.device  # type: ignore[return-value]

    def capture_api_available(self, device: Device) -> bool:
        """Record and return the configured API outcome."""
        del device
        self.calls.append("api")
        return self.api  # type: ignore[return-value]


@pytest.fixture
def resident_request(monkeypatch: pytest.MonkeyPatch) -> object:
    """Build one real, fully published resident request on Warp CPU."""
    pytest.importorskip("warp")
    from particula.execution.communication import CommunicationTransportMode
    from particula.execution.tests.full_loop_test import _build_loop_fixture

    return _build_loop_fixture(
        monkeypatch, CommunicationTransportMode.GAS
    ).request


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


def test_resolve_capability_rejects_missing_probe_method_before_resolution() -> (
    None
):
    """The complete lazy-probe protocol is required before device branching."""
    probe = object()

    with pytest.raises(
        TypeError, match="probe.runtime_available must be callable"
    ):
        resolve_graph_capture_capability(
            Device(Backend.CPU, "cpu"), cast("GraphCaptureRuntimeProbe", probe)
        )


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


@pytest.mark.warp
def test_signature_ignores_rng_words_when_rng_array_identity_is_stable(
    resident_request: object,
) -> None:
    """Changing words in a published RNG sidecar does not inspect payloads."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    rng_states = request.coagulation.resources.rng_states
    rng_states.assign(np.full(rng_states.shape, 13, dtype=np.uint32))

    result = compare_resident_graph_capture_signature(signature, request)

    assert result.compatible is True
    assert result.reason is None


@pytest.mark.warp
def test_signature_reports_rng_resource_after_resource_views(
    resident_request: object,
) -> None:
    """Replacing an RNG array reports its dedicated final drift group."""
    request = cast("ResidentSimulationRequest", resident_request)
    signature = create_resident_graph_capture_signature(request)
    object.__setattr__(request.coagulation.resources, "rng_states", object())

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


def test_signature_carrier_requires_exact_identity_metadata() -> None:
    """Signature carriers reject malformed field containers at construction."""
    values: tuple[object, ...] = (object(),) * 14
    with pytest.raises(TypeError):
        ResidentGraphCaptureSignature(*values)  # type: ignore[arg-type]


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


def test_graph_capture_names_remain_direct_import_only() -> None:
    """Concrete graph-capture declarations do not alter frozen exports."""
    assert "GraphCaptureCapability" not in execution.__all__
    assert not hasattr(execution, "GraphCaptureCapability")
    assert not hasattr(particula, "GraphCaptureCapability")


def test_graph_capture_import_does_not_import_warp() -> None:
    """The declaration module itself has no optional Warp import dependency."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            (
                "import sys; import particula.execution.graph_capture; "
                "assert 'particula.gpu' not in sys.modules; "
                "assert 'warp' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
