# ADR-009: Resident Process Delegation Adapters

**Status:** Accepted
**Date:** 2026-07-29
**Decision Makers:** ADW Development Team
**Technical Story:** [#1494](https://github.com/Gorkowski/particula/issues/1494)

## Context

The resident-session lifecycle provides an exact active session and
identity-pinned `GPUResourceRegistry`. Dilution needs that binding to delegate
on resident containers, while wall loss and nucleation also need their exact
already-published E7-F4 resource views. The direct GPU kernels already own
physical validation, mutation, and post-launch behavior.

### Problem Statement

Provide a bounded resident delegation seam for dilution, wall loss, and
nucleation without creating a public resident-process API, a scheduler, resource
acquisition policy, or an alternate physics implementation.

### Forces

**Driving Forces:**
- Delegation must preserve resident container, sidecar, output, and persistent
  RNG identities.
- Wall-loss and nucleation calls must use only exact registry-published views.
- Direct kernels must remain the single authority for numerical validation and
  writer semantics.

**Restraining Forces:**
- Implicit transfer, synchronization, acquisition, replacement, retry, or
  fallback would violate resident ownership and direct-kernel contracts.
- A package export would create an unsupported public orchestration commitment.
- Rollback cannot be promised after an asynchronous direct writer launches.

## Decision

Add the concrete-only direct-import module
`particula.execution.process_adapters`. It defines frozen, identity-oriented
request carriers and separate resident dilution, wall-loss, and nucleation
adapters. The module is deliberately not exported through
`particula.execution`, its adapters package, or top-level `particula`.

Each adapter validates the exact pinned active `ResidentSession` and
`GPUResourceRegistry` binding before lazily resolving and calling exactly one
supported direct GPU kernel. Wall-loss and nucleation adapters additionally use
new metadata-only registry seams to require the exact already-published resource
view and its pinned sidecar bindings. The adapter forwards all native inputs by
identity and returns the direct kernel's native result unchanged.

### Chosen Option

**Option 2: Concrete identity-pinned delegation adapters**

Keep resident delegation in a narrow execution-layer module that validates
ownership only and delegates one-to-one to the supported direct kernel.

## Alternatives Considered

### Option 1: Let adapters acquire or reconstruct resource views

**Pros:** Callers would supply fewer established objects.

**Cons:** Changes registry state, obscures resource ownership, and risks
replacing persistent sidecars such as RNG state.

**Reason for Rejection:** Delegation must consume exact published views, not
become an acquisition or allocation boundary.

---

### Option 2: Concrete identity-pinned delegation adapters (chosen)

**Pros:** Preserves exact resident ownership while making a single supported
direct-kernel call explicit.

**Cons:** Callers must establish and retain required resource views first.

**Reason for Selection:** It adds only the needed delegation seam and preserves
the existing kernel and registry ownership boundaries.

---

### Option 3: Add a public runnable, scheduler, or fallback layer

**Pros:** Could offer a broader high-level process API.

**Cons:** Implies unresolved scheduling, recovery, backend-selection, transfer,
and public compatibility policies.

**Reason for Rejection:** Those policies are deferred and outside this bounded
resident delegation scope.

## Rationale

Exact session and published-view identity validation prevents a manually
constructed, stale, unacquired, or session-drifted sidecar record from entering
metadata-only preflight. One-to-one delegation leaves direct kernels responsible
for all physical checks, mutation, and any limits after launch.

### Trade-offs Accepted

1. **Concrete-only imports:** Users must not treat adapters as supported public
   APIs.
2. **Established views required:** Wall loss and nucleation cannot self-acquire
   sidecars during dispatch.
3. **No recovery:** Direct-kernel exceptions, including post-launch failures,
   propagate without adapter rollback or retry.

## Consequences

### Positive

- Resident process calls retain containers, sidecars, outputs, and RNG state by
  identity.
- Resource ownership is checked before a kernel is imported or invoked.
- Direct GPU APIs remain the only process-physics and mutation boundaries.

### Negative

- The caller must explicitly create and retain the correct registry views.
- The adapters intentionally provide no convenience transfer, synchronization,
  fallback, or recovery behavior.

### Neutral

- No package export, resident scheduler, resource schema, or direct-kernel
  signature changes.

## Implementation

### Required Changes

1. **Registry established-view validation**
   (`particula/execution/gpu_resources.py`)
   - Add metadata-only wall-loss and nucleation validation seams that require
     exact active session, publication, and pinned bindings.
2. **Resident delegation** (`particula/execution/process_adapters.py`)
   - Add exact-type request carriers and one-call direct-kernel adapters.
3. **Regression coverage** (`particula/execution/tests/process_adapters_test.py`)
   - Verify identity forwarding, exact preflight, one-call dispatch, lazy
     imports, and direct exception propagation.

### Testing Strategy

Use focused Warp-optional tests for established-view rejection, no mutation on
preflight failure, identity-preserving argument forwarding, persistent RNG
forwarding, import isolation, and no retry or recovery after a direct exception.

### Rollback Plan

Remove the concrete module and registry validation seams. No package export,
persistent format, or public API migration is required.

## Validation

### Success Criteria

- [x] Each valid adapter call delegates exactly once to its supported direct
  kernel and returns the native result unchanged.
- [x] Wall-loss and nucleation require exact established published resource
  views; rejected preflight does not acquire or mutate resources.
- [x] The boundary remains concrete-only and performs no transfer,
  synchronization, fallback, retry, rollback, or physics.

## References

- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [ADR-006: Resident GPU Step Lifecycle Guard](ADR-006-resident-gpu-step-lifecycle-guard.md)
- [ADR-008: Resident Session Failure and Terminal Close Semantics](ADR-008-resident-session-failure-close-semantics.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1494](https://github.com/Gorkowski/particula/issues/1494)

## Notes

This ADR extends ADRs 004 and 006, complements ADR-008, and supersedes none.
