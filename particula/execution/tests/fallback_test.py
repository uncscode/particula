"""Tests for the explicit, direct-only CPU fallback boundary."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
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
from particula.execution.errors import (
    ExecutionCapabilityError,
    ExecutionCapabilityReason,
    FallbackDisallowedError,
)
from particula.execution.fallback import (
    CPUStateAuthority,
    FallbackBoundary,
    FallbackDispatchResult,
    FallbackPolicy,
    FallbackRequest,
    FallbackResolution,
    dispatch_cpu_fallback,
    resolve_cpu_fallback,
)


class CountingContext(ExecutionContext):
    """Count selection calls while retaining the normal context behavior."""

    def __init__(self, matrix: CapabilityMatrix) -> None:
        """Initialize context with no selection calls."""
        super().__init__(matrix)
        self.calls = 0

    def resolve(self, request: ExecutionRequest) -> ExecutionAdapter:
        """Count and delegate one selection."""
        self.calls += 1
        return super().resolve(request)


class Adapter:
    """Return one configured native result and count execution."""

    def __init__(self) -> None:
        """Initialize the adapter counter."""
        self.calls = 0
        self.result: ExecutionResult | None = None

    def execute(self, state: CPUExecutionState) -> ExecutionResult:
        """Return a native result retaining the supplied state."""
        self.calls += 1
        self.result = ExecutionResult(
            state,
            (("native", "metadata"),),
            MutationDeclaration(frozenset({MutationScope.STATE})),
        )
        return self.result


class RaisingAdapter:
    """Raise one supplied exception after recording execution."""

    def __init__(self, error: Exception) -> None:
        """Store the exact exception to propagate."""
        self.calls = 0
        self.error = error

    def execute(self, _: CPUExecutionState) -> ExecutionResult:
        """Raise the configured exception without recovery."""
        self.calls += 1
        raise self.error


class MalformedAdapter:
    """Return an invalid result carrier for boundary validation."""

    def __init__(self) -> None:
        """Initialize the execution counter."""
        self.calls = 0

    def execute(self, _: CPUExecutionState) -> object:
        """Return an invalid native result."""
        self.calls += 1
        return object()


class WrongStateAdapter:
    """Return a valid result carrier associated with a different CPU state."""

    def __init__(self, state: CPUExecutionState) -> None:
        """Store the different state and initialize the execution counter."""
        self.calls = 0
        self.state = state

    def execute(self, _: CPUExecutionState) -> ExecutionResult:
        """Return a syntactically valid result with the wrong state identity."""
        self.calls += 1
        return ExecutionResult(
            self.state,
            (),
            MutationDeclaration(frozenset({MutationScope.STATE})),
        )


def _request() -> ExecutionRequest:
    """Build a valid non-CPU request."""
    return ExecutionRequest(
        Backend.WARP,
        Device(Backend.WARP, "opaque:0"),
        Process("fallback_test"),
        CapabilityRequirements(frozenset({Capability("fallback_capability")})),
    )


def _context(request: ExecutionRequest) -> tuple[CountingContext, Adapter]:
    """Build a context with only the canonical CPU declaration and adapter."""
    cpu_device = Device(Backend.CPU, "cpu")
    matrix = CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(
                    cpu_device,
                    request.process,
                    request.requirements,
                )
            }
        )
    )
    context = CountingContext(matrix)
    adapter = Adapter()
    context.register_adapter(request.process, Backend.CPU, adapter)
    return context, adapter


def _fallback(
    *,
    reason: ExecutionCapabilityReason = ExecutionCapabilityReason.UNKNOWN_DEVICE,
    policy: FallbackPolicy = FallbackPolicy.CPU,
    authority: CPUStateAuthority = CPUStateAuthority.CPU_AUTHORITATIVE,
    boundary: FallbackBoundary = FallbackBoundary.PRE_UPLOAD,
) -> tuple[FallbackRequest, CountingContext, Adapter]:
    """Build one complete valid fallback request and its local collaborators."""
    request = _request()
    context, adapter = _context(request)
    error = ExecutionCapabilityError(
        reason,
        backend=request.backend.value,
        device=request.device.native,
        process=request.process.name,
        capability=repr(request.requirements),
    )
    fallback = FallbackRequest(
        request,
        error,
        context,
        CPUExecutionState(object(), 1.0, 1),
        policy,
        boundary,
        authority,
    )
    return fallback, context, adapter


@pytest.mark.parametrize(
    "reason",
    [
        ExecutionCapabilityReason.UNKNOWN_DEVICE,
        ExecutionCapabilityReason.RUNTIME_UNAVAILABLE,
        ExecutionCapabilityReason.DEVICE_UNAVAILABLE,
        ExecutionCapabilityReason.PROCESS_UNSUPPORTED,
        ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED,
    ],
)
def test_default_policy_reraises_exact_eligible_error_without_lookup(
    reason: ExecutionCapabilityReason,
) -> None:
    """Default-deny policy re-raises without selecting or reading CPU payload."""
    fallback, context, adapter = _fallback(
        reason=reason, policy=FallbackPolicy.RAISE
    )

    with pytest.raises(ExecutionCapabilityError) as raised:
        resolve_cpu_fallback(fallback)

    assert raised.value is fallback.original_error
    assert context.calls == 0
    assert adapter.calls == 0


def test_resolution_retains_identity_and_uses_one_canonical_cpu_request() -> (
    None
):
    """Explicit CPU fallback selects once and retains caller-owned values."""
    fallback, context, adapter = _fallback()

    resolution = resolve_cpu_fallback(fallback)

    assert context.calls == 1
    assert resolution.original_request is fallback.original_request
    assert resolution.original_error is fallback.original_error
    assert resolution.cpu_state is fallback.cpu_state
    assert resolution.adapter is adapter
    assert resolution.cpu_request.backend is Backend.CPU
    assert resolution.cpu_request.device == Device(Backend.CPU, "cpu")
    assert resolution.cpu_request.process is fallback.original_request.process
    assert (
        resolution.cpu_request.requirements
        is fallback.original_request.requirements
    )
    with pytest.raises(FrozenInstanceError):
        resolution.adapter = adapter  # type: ignore[misc]


def test_dispatch_executes_once_and_keeps_native_metadata_unchanged() -> None:
    """Dispatch invokes the selected adapter once without modifying its result."""
    fallback, context, adapter = _fallback()

    dispatch = dispatch_cpu_fallback(fallback)

    assert context.calls == 1
    assert adapter.calls == 1
    assert dispatch.resolution.adapter is adapter
    assert dispatch.result is adapter.result
    assert dispatch.result.metadata == (("native", "metadata"),)
    assert dispatch.metadata == (
        ("requested_backend", "warp"),
        ("selected_backend", "cpu"),
        ("capability_reason", "unknown_device"),
    )


@pytest.mark.parametrize(
    "boundary",
    [FallbackBoundary.PRE_UPLOAD, FallbackBoundary.RESTORED],
)
def test_both_authoritative_boundaries_dispatch_without_restoration(
    boundary: FallbackBoundary,
) -> None:
    """Both caller-visible CPU-authoritative boundaries permit dispatch."""
    fallback, context, adapter = _fallback(boundary=boundary)

    dispatch_cpu_fallback(fallback)

    assert context.calls == 1
    assert adapter.calls == 1


@pytest.mark.parametrize(
    "authority",
    [
        CPUStateAuthority.RESIDENT,
        CPUStateAuthority.UPLOADED,
        CPUStateAuthority.MUTATED,
    ],
)
def test_non_authoritative_claims_fail_closed_before_lookup(
    authority: CPUStateAuthority,
) -> None:
    """Resident, uploaded, and mutated claims cannot select CPU fallback."""
    fallback, context, adapter = _fallback(authority=authority)

    with pytest.raises(FallbackDisallowedError) as raised:
        resolve_cpu_fallback(fallback)

    assert raised.value.__cause__ is fallback.original_error
    assert context.calls == 0
    assert adapter.calls == 0


@pytest.mark.parametrize(
    "reason",
    [
        ExecutionCapabilityReason.UNKNOWN_BACKEND,
        ExecutionCapabilityReason.INVALID_STATE,
        ExecutionCapabilityReason.FALLBACK_DISALLOWED,
    ],
)
def test_ineligible_reason_precedes_default_reraise(
    reason: ExecutionCapabilityReason,
) -> None:
    """Ineligible reasons fail closed even under the default policy."""
    fallback, context, adapter = _fallback(
        reason=reason, policy=FallbackPolicy.RAISE
    )

    with pytest.raises(FallbackDisallowedError) as raised:
        resolve_cpu_fallback(fallback)

    assert raised.value.__cause__ is fallback.original_error
    assert context.calls == 0
    assert adapter.calls == 0


def test_missing_cpu_state_and_error_context_mismatch_fail_before_lookup() -> (
    None
):
    """State and original-error context are both preflight-only validations."""
    fallback, context, adapter = _fallback()

    with pytest.raises(FallbackDisallowedError) as missing:
        resolve_cpu_fallback(replace(fallback, cpu_state=None))
    assert missing.value.__cause__ is fallback.original_error
    assert context.calls == 0

    mismatch = ExecutionCapabilityError(
        fallback.original_error.reason,
        backend="not_warp",
    )
    invalid = replace(fallback, original_error=mismatch)
    with pytest.raises(FallbackDisallowedError) as raised:
        resolve_cpu_fallback(invalid)
    assert raised.value.__cause__ is mismatch
    assert context.calls == 0
    assert adapter.calls == 0


def test_cpu_origin_request_fails_closed_before_lookup() -> None:
    """A CPU request cannot recursively use the explicit CPU fallback seam."""
    fallback, context, adapter = _fallback()
    cpu_request = ExecutionRequest(
        Backend.CPU,
        Device(Backend.CPU, "cpu"),
        fallback.original_request.process,
        fallback.original_request.requirements,
    )
    cpu_error = ExecutionCapabilityError(
        ExecutionCapabilityReason.UNKNOWN_DEVICE
    )

    with pytest.raises(FallbackDisallowedError) as raised:
        resolve_cpu_fallback(
            replace(
                fallback, original_request=cpu_request, original_error=cpu_error
            )
        )

    assert raised.value.__cause__ is cpu_error
    assert context.calls == 0
    assert adapter.calls == 0


def test_context_support_and_registration_errors_propagate_after_one_lookup() -> (
    None
):
    """CPU support and registration remain owned by the unchanged context call."""
    fallback, _, _ = _fallback()
    empty_context = CountingContext(CapabilityMatrix(frozenset()))

    with pytest.raises(ValueError):
        resolve_cpu_fallback(replace(fallback, context=empty_context))
    assert empty_context.calls == 1

    matrix_context, _ = _context(fallback.original_request)
    missing_context = CountingContext(matrix_context._matrix)  # noqa: SLF001
    with pytest.raises(LookupError):
        resolve_cpu_fallback(replace(fallback, context=missing_context))
    assert missing_context.calls == 1


def test_adapter_exceptions_and_malformed_results_are_not_retried() -> None:
    """Post-selection adapter failures propagate with one selection and call."""
    fallback, context, _ = _fallback()
    sentinel = RuntimeError("adapter failure")
    raising_adapter = RaisingAdapter(sentinel)
    context = CountingContext(context._matrix)  # noqa: SLF001
    context.register_adapter(
        fallback.original_request.process,
        Backend.CPU,
        raising_adapter,
    )

    with pytest.raises(RuntimeError) as raised:
        dispatch_cpu_fallback(replace(fallback, context=context))
    assert raised.value is sentinel
    assert context.calls == 1
    assert raising_adapter.calls == 1

    fallback, context, _ = _fallback()
    malformed_adapter = MalformedAdapter()
    context = CountingContext(context._matrix)  # noqa: SLF001
    context.register_adapter(
        fallback.original_request.process,
        Backend.CPU,
        malformed_adapter,
    )
    with pytest.raises(TypeError, match="ExecutionResult"):
        dispatch_cpu_fallback(replace(fallback, context=context))
    assert context.calls == 1
    assert malformed_adapter.calls == 1


def test_wrong_state_result_is_not_retried_or_reselected() -> None:
    """A valid result for another CPU state fails after one dispatch."""
    fallback, context, _ = _fallback()
    wrong_state_adapter = WrongStateAdapter(CPUExecutionState(object(), 1.0, 1))
    context = CountingContext(context._matrix)  # noqa: SLF001
    context.register_adapter(
        fallback.original_request.process,
        Backend.CPU,
        wrong_state_adapter,
    )

    with pytest.raises(ValueError, match="ExecutionResult.state"):
        dispatch_cpu_fallback(replace(fallback, context=context))

    assert context.calls == 1
    assert wrong_state_adapter.calls == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("original_request", object(), "original_request"),
        ("original_error", object(), "original_error"),
        ("context", object(), "context"),
        ("cpu_state", object(), "cpu_state"),
        ("policy", "cpu", "policy"),
        ("boundary", "pre_upload", "boundary"),
        ("state_authority", "cpu_authoritative", "state_authority"),
    ],
)
def test_fallback_request_rejects_invalid_carrier_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    """Fallback request construction rejects invalid closed-carrier fields."""
    fallback, _, _ = _fallback()

    with pytest.raises(TypeError, match=message):
        replace(fallback, **{field: value})  # type: ignore[arg-type]


def test_forged_enum_and_request_metadata_fail_before_context_lookup() -> None:
    """Post-construction carrier tampering cannot reach CPU selection."""
    fallback, context, adapter = _fallback()
    object.__setattr__(fallback, "policy", "cpu")

    with pytest.raises(FallbackDisallowedError) as raised:
        resolve_cpu_fallback(fallback)

    assert raised.value.__cause__ is fallback.original_error
    assert context.calls == 0
    assert adapter.calls == 0

    fallback, context, adapter = _fallback()
    object.__setattr__(fallback.original_request, "backend", "warp")

    with pytest.raises(TypeError, match="original_request.backend"):
        resolve_cpu_fallback(fallback)

    assert context.calls == 0
    assert adapter.calls == 0


def test_forged_nested_carriers_fail_before_context_lookup() -> None:
    """Nested fallback request and context matrix values are fail-closed."""
    fallback, context, adapter = _fallback()
    object.__setattr__(fallback.original_request.device, "native", 1)

    with pytest.raises(TypeError, match="original_request.device.native"):
        resolve_cpu_fallback(fallback)

    assert context.calls == 0
    assert adapter.calls == 0

    fallback, context, adapter = _fallback()
    object.__setattr__(context._matrix, "declarations", ())  # noqa: SLF001

    with pytest.raises(TypeError, match="context.matrix.declarations"):
        resolve_cpu_fallback(fallback)

    assert context.calls == 0
    assert adapter.calls == 0


def test_omitted_state_authority_is_rejected() -> None:
    """Fallback callers must explicitly attest CPU state authority."""
    fallback, _, _ = _fallback()

    with pytest.raises(TypeError, match="state_authority"):
        FallbackRequest(
            fallback.original_request,
            fallback.original_error,
            fallback.context,
            fallback.cpu_state,
        )


@pytest.mark.parametrize("field", ["state", "fallback_boundary"])
def test_error_state_context_disallows_fallback_before_lookup(
    field: str,
) -> None:
    """Error state and boundary context cannot be supplied by a fallback caller."""
    fallback, context, adapter = _fallback()
    error = ExecutionCapabilityError(
        fallback.original_error.reason,
        **{field: "forged"},
    )

    with pytest.raises(FallbackDisallowedError) as raised:
        resolve_cpu_fallback(replace(fallback, original_error=error))

    assert raised.value.__cause__ is error
    assert context.calls == 0
    assert adapter.calls == 0


def test_resolution_and_dispatch_result_validate_exact_provenance() -> None:
    """Resolution and dispatch records reject invalid closed metadata."""
    fallback, _, _ = _fallback()
    resolution = resolve_cpu_fallback(fallback)
    dispatch = dispatch_cpu_fallback(fallback)

    with pytest.raises(ValueError, match="selected_backend"):
        replace(resolution, selected_backend=Backend.WARP)
    with pytest.raises(ValueError, match="cpu_request.backend"):
        replace(resolution, cpu_request=fallback.original_request)
    with pytest.raises(ValueError, match="requested_backend"):
        replace(resolution, requested_backend=Backend.CPU)
    with pytest.raises(TypeError, match="requested_backend"):
        replace(resolution, requested_backend="warp")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="capability_reason"):
        replace(
            resolution,
            capability_reason=cast(ExecutionCapabilityReason, "unknown_device"),
        )
    with pytest.raises(ValueError, match="exact fallback provenance"):
        replace(dispatch, metadata=())
    with pytest.raises(TypeError, match="resolution"):
        FallbackDispatchResult(
            cast(FallbackResolution, object()),
            dispatch.result,
            dispatch.metadata,
        )

    assert isinstance(resolution, FallbackResolution)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("policy", "policy"),
        ("original_request", "original_request.backend"),
        ("original_error", "original_error.reason"),
    ],
)
def test_resolution_rejects_unauthorized_fallback_provenance(
    field: str,
    message: str,
) -> None:
    """Direct construction cannot claim fallback without resolver admission."""
    fallback, context, adapter = _fallback()
    cpu_request = ExecutionRequest(
        Backend.CPU,
        Device(Backend.CPU, "cpu"),
        fallback.original_request.process,
        fallback.original_request.requirements,
    )
    original_request = (
        cpu_request
        if field == "original_request"
        else fallback.original_request
    )
    original_error = (
        ExecutionCapabilityError(ExecutionCapabilityReason.UNKNOWN_BACKEND)
        if field == "original_error"
        else fallback.original_error
    )
    policy = FallbackPolicy.RAISE if field == "policy" else fallback.policy

    with pytest.raises(ValueError, match=message):
        FallbackResolution(
            original_request,
            cpu_request,
            original_error,
            cast(CPUExecutionState, fallback.cpu_state),
            adapter,
            policy,
            fallback.boundary,
            original_request.backend,
            Backend.CPU,
            original_error.reason,
        )

    assert context.calls == 0
    assert adapter.calls == 0


def test_fallback_import_is_direct_only_and_unexported() -> None:
    """Fallback import avoids optional and concrete execution dependencies."""
    script = """
import sys
import particula
import particula.execution as execution
import particula.execution.fallback

forbidden = (
    'warp', 'particula.gpu', 'particula.execution.availability',
    'particula.execution.adapters', 'particula.execution.gpu_session',
    'particula.execution.checkpoint', 'particula.execution.gpu_resources',
    'particula.execution.conversion',
)
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for prefix in forbidden for name in sys.modules
)
assert 'fallback' not in execution.__all__
assert not hasattr(particula, 'fallback')
assert execution.FallbackPolicy is particula.FallbackPolicy
assert execution.FallbackBoundary is particula.FallbackBoundary
assert execution.CPUStateAuthority is particula.CPUStateAuthority
"""

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
