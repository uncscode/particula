"""Integration regressions for explicit CPU fallback boundaries."""

from dataclasses import dataclass
from typing import cast

import pytest

from particula.execution import (
    Backend,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    CPUExecutionState,
    Device,
    ExecutionAdapter,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    MutationDeclaration,
    MutationScope,
    Process,
)
from particula.execution.availability import (
    AvailabilityProvider,
    resolve_availability,
)
from particula.execution.errors import UnavailableDeviceError
from particula.execution.fallback import (
    CPUStateAuthority,
    FallbackBoundary,
    FallbackPolicy,
    FallbackRequest,
    dispatch_cpu_fallback,
)


class RecordingProvider:
    """Record the availability phases invoked by the resolver."""

    def __init__(
        self,
        log: list[str],
        *,
        recognizes_result: bool,
        runtime_result: bool,
        device_result: bool,
    ) -> None:
        """Store phase results and the shared call log."""
        self.log = log
        self.recognizes_result = recognizes_result
        self.runtime_result = runtime_result
        self.device_result = device_result

    def recognizes(self, _: Device) -> bool:
        """Record and return the configured recognition result."""
        self.log.append("recognition")
        return self.recognizes_result

    def runtime_available(self) -> bool:
        """Record and return the configured runtime result."""
        self.log.append("runtime")
        return self.runtime_result

    def device_available(self, _: Device) -> bool:
        """Record and return the configured device result."""
        self.log.append("device")
        return self.device_result


class RecordingAdapter:
    """Return a native result while recording one CPU adapter call."""

    def __init__(self, seams: "ForbiddenSeams") -> None:
        """Initialize the call counter and empty native result."""
        self.calls = 0
        self.result: ExecutionResult | None = None
        self.seams = seams

    def execute(self, state: CPUExecutionState) -> ExecutionResult:
        """Return a non-mutating result retaining the exact state."""
        self.calls += 1
        self.result = ExecutionResult(
            state,
            (("native", "metadata"),),
            MutationDeclaration(frozenset({MutationScope.NONE})),
        )
        return self.result


class RecordingGPUAdapter:
    """Fail if explicit fallback dispatch selects the GPU adapter."""

    def __init__(self, seams: "ForbiddenSeams") -> None:
        """Initialize the GPU adapter call counter."""
        self.calls = 0
        self.seams = seams

    def execute(self, _: CPUExecutionState) -> ExecutionResult:
        """Record accidental entry and fail immediately."""
        self.calls += 1
        self.seams.conversion()
        self.seams.transfer()
        self.seams.synchronization()
        self.seams.kernel()
        self.seams.mutation()
        self.seams.checkpoint()
        self.seams.restore()
        raise AssertionError("The GPU adapter must not be selected.")


class RaisingAdapter:
    """Raise one exact adapter error after recording the sole invocation."""

    def __init__(self, error: RuntimeError, seams: "ForbiddenSeams") -> None:
        """Store the exact error and initialize the call counter."""
        self.calls = 0
        self.error = error
        self.seams = seams

    def execute(self, _: CPUExecutionState) -> ExecutionResult:
        """Raise the configured error unchanged."""
        self.calls += 1
        raise self.error


class CountingContext(ExecutionContext):
    """Count adapter lookups while retaining normal context selection."""

    def __init__(self, matrix: CapabilityMatrix) -> None:
        """Initialize a context with no lookups."""
        super().__init__(matrix)
        self.lookups = 0

    def resolve(self, request: ExecutionRequest) -> ExecutionAdapter:
        """Count and delegate one adapter lookup."""
        self.lookups += 1
        return super().resolve(request)


class ForbiddenSeams:
    """Count forbidden execution seams that must remain unused."""

    def __init__(self) -> None:
        """Initialize zero counts for every forbidden seam."""
        self.conversion_calls = 0
        self.transfer_calls = 0
        self.synchronization_calls = 0
        self.kernel_calls = 0
        self.mutation_calls = 0
        self.checkpoint_calls = 0
        self.restore_calls = 0

    def conversion(self) -> None:
        """Record a forbidden conversion seam."""
        self.conversion_calls += 1

    def transfer(self) -> None:
        """Record a forbidden transfer seam."""
        self.transfer_calls += 1

    def synchronization(self) -> None:
        """Record a forbidden synchronization seam."""
        self.synchronization_calls += 1

    def kernel(self) -> None:
        """Record a forbidden kernel seam."""
        self.kernel_calls += 1

    def mutation(self) -> None:
        """Record a forbidden mutation seam."""
        self.mutation_calls += 1

    def checkpoint(self) -> None:
        """Record a forbidden checkpoint seam."""
        self.checkpoint_calls += 1

    def restore(self) -> None:
        """Record a forbidden restore seam."""
        self.restore_calls += 1


@dataclass(frozen=True)
class RejectedAvailability:
    """Retain fresh collaborators from one rejected availability request."""

    request: ExecutionRequest
    matrix: CapabilityMatrix
    error: UnavailableDeviceError
    provider_log: list[str]
    state_log: list[str]
    request_snapshot: tuple[
        ExecutionRequest,
        Device,
        Process,
        CapabilityRequirements,
        frozenset[CapabilityDeclaration],
    ]


def _request() -> ExecutionRequest:
    """Build one valid Warp request with nonempty requirements."""
    return ExecutionRequest(
        Backend.WARP,
        Device(Backend.WARP, "opaque:0"),
        Process("fallback_integration"),
        CapabilityRequirements(
            frozenset({Capability("fallback_integration_capability")})
        ),
    )


def _matrix(request: ExecutionRequest) -> CapabilityMatrix:
    """Build exact Warp and canonical CPU declarations for a request."""
    return CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(
                    request.device,
                    request.process,
                    request.requirements,
                ),
                CapabilityDeclaration(
                    Device(Backend.CPU, "cpu"),
                    request.process,
                    request.requirements,
                ),
            }
        )
    )


def _providers(log: list[str]) -> dict[Backend, AvailabilityProvider]:
    """Build a complete registry whose selected Warp device is unavailable."""
    return cast(
        dict[Backend, AvailabilityProvider],
        {
            Backend.CPU: RecordingProvider(
                log,
                recognizes_result=True,
                runtime_result=True,
                device_result=True,
            ),
            Backend.WARP: RecordingProvider(
                log,
                recognizes_result=True,
                runtime_result=True,
                device_result=False,
            ),
        },
    )


def _ledger_counts(seams: ForbiddenSeams) -> tuple[int, ...]:
    """Return every forbidden seam counter for compact assertions."""
    return (
        seams.conversion_calls,
        seams.transfer_calls,
        seams.synchronization_calls,
        seams.kernel_calls,
        seams.mutation_calls,
        seams.checkpoint_calls,
        seams.restore_calls,
    )


def _reject_availability() -> RejectedAvailability:
    """Return fresh P1 data and the exact P2 unavailable-device error."""
    request = _request()
    matrix = _matrix(request)
    request_snapshot = (
        request,
        request.device,
        request.process,
        request.requirements,
        matrix.declarations,
    )
    provider_log: list[str] = []
    state_log: list[str] = []

    def validate_state(_: ExecutionRequest) -> bool:
        """Record state validation if device availability permits it."""
        state_log.append("state")
        return True

    with pytest.raises(UnavailableDeviceError) as raised:
        resolve_availability(
            request,
            matrix,
            providers=_providers(provider_log),
            state_validator=validate_state,
        )
    return RejectedAvailability(
        request,
        matrix,
        raised.value,
        provider_log,
        state_log,
        request_snapshot,
    )


def _context(
    matrix: CapabilityMatrix,
    request: ExecutionRequest,
    cpu_adapter: ExecutionAdapter,
    gpu_adapter: ExecutionAdapter,
) -> CountingContext:
    """Register fresh CPU and GPU adapters in one counting context."""
    context = CountingContext(matrix)
    context.register_adapter(request.process, Backend.CPU, cpu_adapter)
    context.register_adapter(request.process, Backend.WARP, gpu_adapter)
    return context


def _state() -> tuple[CPUExecutionState, list[float], tuple[int, list[float]]]:
    """Build a CPU state with identity and value snapshots of its payload."""
    payload = [1.0, 2.0]
    return (
        CPUExecutionState(payload, 1.0, 1),
        payload,
        (id(payload), payload.copy()),
    )


def test_unavailable_device_short_circuits_before_dispatch() -> None:
    """P2 device rejection stops before state validation or adapter lookup."""
    rejected = _reject_availability()
    seams = ForbiddenSeams()
    cpu_adapter = RecordingAdapter(seams)
    gpu_adapter = RecordingGPUAdapter(seams)
    context = _context(
        rejected.matrix,
        rejected.request,
        cpu_adapter,
        gpu_adapter,
    )

    assert rejected.error.backend == "warp"
    assert rejected.error.device == "opaque:0"
    assert rejected.provider_log == ["recognition", "runtime", "device"]
    assert rejected.state_log == []
    assert rejected.request is rejected.request_snapshot[0]
    assert rejected.request.device is rejected.request_snapshot[1]
    assert rejected.request.process is rejected.request_snapshot[2]
    assert rejected.request.requirements is rejected.request_snapshot[3]
    assert rejected.matrix.declarations == rejected.request_snapshot[4]
    assert context.lookups == 0
    assert cpu_adapter.calls == 0
    assert gpu_adapter.calls == 0
    assert _ledger_counts(seams) == (0, 0, 0, 0, 0, 0, 0)

    fallback = FallbackRequest(
        rejected.request,
        rejected.error,
        context,
        None,
    )

    assert fallback.original_error is rejected.error
    assert context.lookups == 0


def test_explicit_cpu_fallback_dispatches_once_with_provenance() -> None:
    """Caller-authored P3 fallback retains CPU state and exact provenance."""
    rejected = _reject_availability()
    seams = ForbiddenSeams()
    cpu_adapter = RecordingAdapter(seams)
    gpu_adapter = RecordingGPUAdapter(seams)
    context = _context(
        rejected.matrix,
        rejected.request,
        cpu_adapter,
        gpu_adapter,
    )
    cpu_state, payload, payload_snapshot = _state()
    fallback = FallbackRequest(
        rejected.request,
        rejected.error,
        context,
        cpu_state,
        FallbackPolicy.CPU,
        FallbackBoundary.PRE_UPLOAD,
        CPUStateAuthority.CPU_AUTHORITATIVE,
    )

    dispatch = dispatch_cpu_fallback(fallback)

    assert context.lookups == 1
    assert cpu_adapter.calls == 1
    assert gpu_adapter.calls == 0
    assert id(payload) == payload_snapshot[0]
    assert payload == payload_snapshot[1]
    assert dispatch.resolution.original_request is rejected.request
    assert dispatch.resolution.original_error is rejected.error
    assert dispatch.resolution.cpu_state is cpu_state
    assert dispatch.result is cpu_adapter.result
    assert dispatch.result.state is cpu_state
    assert dispatch.result.metadata == (("native", "metadata"),)
    assert _ledger_counts(seams) == (0, 0, 0, 0, 0, 0, 0)
    assert dispatch.metadata == (
        ("requested_backend", "warp"),
        ("selected_backend", "cpu"),
        ("capability_reason", "device_unavailable"),
    )


def test_cpu_adapter_failure_propagates_without_retry_or_reselection() -> None:
    """The selected CPU adapter error escapes unchanged after one call."""
    rejected = _reject_availability()
    sentinel = RuntimeError("sentinel adapter failure")
    seams = ForbiddenSeams()
    cpu_adapter = RaisingAdapter(sentinel, seams)
    gpu_adapter = RecordingGPUAdapter(seams)
    context = _context(
        rejected.matrix,
        rejected.request,
        cpu_adapter,
        gpu_adapter,
    )
    cpu_state, payload, payload_snapshot = _state()
    fallback = FallbackRequest(
        rejected.request,
        rejected.error,
        context,
        cpu_state,
        FallbackPolicy.CPU,
        FallbackBoundary.PRE_UPLOAD,
        CPUStateAuthority.CPU_AUTHORITATIVE,
    )

    with pytest.raises(RuntimeError) as raised:
        dispatch_cpu_fallback(fallback)

    assert raised.value is sentinel
    assert context.lookups == 1
    assert cpu_adapter.calls == 1
    assert gpu_adapter.calls == 0
    assert id(payload) == payload_snapshot[0]
    assert payload == payload_snapshot[1]
    assert _ledger_counts(seams) == (0, 0, 0, 0, 0, 0, 0)
