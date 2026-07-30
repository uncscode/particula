---
title: Backend Selection and Explicit CPU Fallback
---

# Backend Selection and Explicit CPU Fallback

`particula.execution` provides a frozen, dependency-neutral value API for
declaring and selecting a registered execution adapter. It is separate from
the supported experimental `particula.gpu` containers, explicit-transfer
helpers, and direct kernels. See [Data Containers and GPU
Foundations](data-containers-and-gpu-foundations.md), the [Epic G
roadmap](Roadmap/data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation),
and [GPU resident checkpoints](gpu_resident_checkpoints.md).

## Stable public values

The ordered public surface is exactly:

1. `Backend`, `Device`, `Process`, `Capability`, `CapabilityRequirements`,
   `CapabilityDeclaration`, `CapabilityMatrix`, `ExecutionRequest`,
   `ExecutionAdapter`, `ExecutionContext`
2. `ExecutionCapabilityReason`, `ExecutionCapabilityError`,
   `UnknownExecutionTargetError`, `UnavailableExecutionTargetError`,
   `UnsupportedExecutionRequestError`, `UnknownBackendError`,
   `UnknownDeviceError`, `UnavailableRuntimeError`, `UnavailableDeviceError`,
   `UnsupportedProcessError`, `UnsupportedCapabilityError`,
   `InvalidExecutionStateError`, `FallbackDisallowedError`
3. `FallbackPolicy`, `FallbackBoundary`, `CPUStateAuthority`

Availability and fallback mechanics, adapters, resident session and checkpoint
seams, registries, and GPU sidecars are concrete-module-only. They are not a
general high-level execution API and must not be imported through the stable
value surface.

### CPU-only selection example

This executable example only registers and selects a local adapter. It neither
invokes the adapter nor resolves availability or fallback.

```python
from particula.execution import (
    Backend,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionContext,
    ExecutionRequest,
    Process,
)


class LocalAdapter:
    calls = 0

    def execute(self) -> None:
        self.calls += 1
        raise RuntimeError("Selection must not invoke this adapter.")


device = Device(Backend.CPU, "cpu")
process = Process("condensation")
requirements = CapabilityRequirements(frozenset())
matrix = CapabilityMatrix(
    frozenset({CapabilityDeclaration(device, process, requirements)})
)
context = ExecutionContext(matrix)
adapter = LocalAdapter()
context.register_adapter(process, Backend.CPU, adapter)
request = ExecutionRequest(Backend.CPU, device, process, requirements)

assert context.resolve(request) is adapter
```

## Availability and fallback boundaries

`particula.execution.availability.resolve_availability()` is concrete-only. Its
immutable decision resolves in this exact order: complete provider registry,
target recognition, process declaration, capability declaration, lazy runtime
availability, device availability, then request-associated state. It neither
selects an adapter nor executes, transfers, synchronizes, allocates execution
resources, or mutates state.

| Reason | Caller action |
| --- | --- |
| `UNKNOWN_BACKEND` | Reject; correct the declared backend. |
| `UNKNOWN_DEVICE` | Reject by default; explicit CPU policy is eligible. |
| `RUNTIME_UNAVAILABLE` | Reject by default; explicit CPU policy is eligible. |
| `DEVICE_UNAVAILABLE` | Reject by default; explicit CPU policy is eligible. |
| `PROCESS_UNSUPPORTED` | Reject by default; explicit CPU policy is eligible. |
| `CAPABILITY_UNSUPPORTED` | Reject by default; explicit CPU policy is eligible. |
| `INVALID_STATE` | Reject; establish a valid caller-owned boundary. |
| `FALLBACK_DISALLOWED` | Reject; correct policy, authority, or boundary. |

`FallbackPolicy.RAISE` is default-deny. `FallbackPolicy.CPU` permits one
canonical CPU dispatch only for an eligible reason, exact
`CPUStateAuthority.CPU_AUTHORITATIVE` state, and a visible
`FallbackBoundary.PRE_UPLOAD` or caller-asserted `FallbackBoundary.RESTORED`
boundary. Fallback provenance is separate from and does not change native
result metadata. Adapter, kernel, and runtime errors after invocation propagate
without CPU retry.

### Resident boundary illustration (non-executed pseudocode)

The following is a boundary illustration, not an availability API or automatic
fallback path. Active resident state requires a caller-owned checkpoint/finalize
snapshot and a separate explicit restoration to CPU-authoritative state plus a
CPU-authority declaration before a distinct CPU request. Resident restart is
an exact-device GPU lifecycle operation, not CPU restoration or fallback. No
component restores, transfers, synchronizes, migrates, or retries silently.

```python
WARP_AVAILABLE = False

if WARP_AVAILABLE:
    from particula.execution.checkpoint import ResidentCheckpointController
    from particula.execution.gpu_session import ResidentSession

    controller = ResidentCheckpointController()
    checkpoint = resident_session.checkpoint(registry, guard)
    finalized = resident_session.finalize(registry, guard)
    cpu_state = caller_restore_to_cpu_authoritative_state(checkpoint)
    cpu_authority = "caller declares CPU_AUTHORITATIVE after explicit restore"
    cpu_request = "a distinct canonical CPU request"
```
