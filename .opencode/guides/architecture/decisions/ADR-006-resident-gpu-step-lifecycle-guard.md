# ADR-006: Resident GPU Step Lifecycle Guard

**Status:** Accepted
**Date:** 2026-07-28
**Decision Makers:** ADW Development Team
**Technical Story:** [#1487](https://github.com/Gorkowski/particula/issues/1487)

## Context

ADR-004 established an immutable, concrete-only resident-session carrier, and
the P3 registry pins reusable sidecars to one active session. Scheduler-facing
resident execution also needs a minimal way to prevent overlapping timesteps
and to record only completed timestep metadata, without prematurely defining a
scheduler, adapter dispatch, transfer lifecycle, or fault policy.

### Problem Statement

Provide one exact-session, exact-registry timestep bookkeeping boundary that
prevents a second guarded timestep from opening and keeps lifecycle operations
out of an unfinished timestep window.

### Forces

**Driving Forces:**
- Resident adapters need a stable, identity-bound open/complete timestep seam.
- Completion count and simulated time must not advance for failed or unfinished
  adapter work.
- Future checkpoint, restore, close, and fault boundaries need an explicit
  closed-step precondition.

**Restraining Forces:**
- Dispatch, synchronization, conversion, allocation, and recovery have separate
  ownership and must not become hidden guard behavior.
- Session and registry invariants must be checked without acquiring sidecars.
- The ten-name `particula.execution` export surface must remain unchanged.

## Decision

Add direct-import-only P4 `ResidentStepGuard` and identity-only,
`@dataclass(frozen=True, eq=False)` `ResidentStepToken` to
`particula.execution.gpu_session`. The guard retains one exact
`ResidentSession` and `GPUResourceRegistry` binding, permits at most one open
token, and advances completed-step count and simulated time only after
`complete_step()` receives that exact outstanding token.

`GPUResourceRegistry.validate_pinned_session(session)` is the narrow
metadata-only integration seam. It first requires exact retained-session
identity, then reuses its existing active lifecycle, signature, and schema
validation; it neither acquires nor allocates resources.

Future checkpoint, restore, finalize, close, fault, conversion, resize, and
rebind boundaries must call `assert_step_closed()` before doing their own work.
P5/P6 retain implementation and policy for those operations.

### Chosen Option

**Option 2: Identity-bound bookkeeping guard with an explicit closed-step gate**

The guard validates the pinned active binding before opening or completing a
step. It does not invoke adapters: a scheduler completes the token only after
its ordered work succeeds. Raw low-level helpers remain outside the guard's
global interception.

## Alternatives Considered

### Option 1: Put mutable timestep state on `ResidentSession`

**Pros:** Keeps session-related state on one object.

**Cons:** Violates the immutable P1 carrier boundary and mixes operational state
with schema metadata.

**Reason for Rejection:** P4 bookkeeping must remain separate from the frozen
resident-session carrier.

---

### Option 2: Identity-bound bookkeeping guard with an explicit closed-step gate

**Pros:** Preserves immutable session state, makes concurrency boundaries
explicit, and has no hidden runtime work.

**Cons:** Future lifecycle boundaries must explicitly invoke the gate.

**Reason for Selection:** It supplies the required lifecycle invariant without
claiming scheduler or lifecycle-operation ownership.

---

### Option 3: Implement a scheduler and lifecycle manager now

**Pros:** Could centralize adapter ordering, recovery, and transitions.

**Cons:** Prematurely commits dispatch, transfer, synchronization, checkpoint,
fault, close, and fallback semantics.

**Reason for Rejection:** Those concerns belong to later P5/P6 decisions.

## Rationale

An opaque identity token makes completion unforgeable through value equality,
while registry validation preserves the existing pinned-session invariant.
Separating successful completion from adapter invocation leaves native execution
errors and recovery policy with their established owners.

### Trade-offs Accepted

1. **Explicit gate adoption**: Future lifecycle boundaries must call
   `assert_step_closed()`; raw helpers are not intercepted.
2. **No rollback policy**: A failed adapter leaves the token open and metadata
   unchanged until later policy handles it.
3. **Concrete-only API**: The guard and token are not package exports.

## Consequences

### Positive

- One active binding cannot have overlapping guarded timesteps.
- Completed metadata accurately represents matching completed work only.
- Lifecycle boundaries have a small, testable precondition hook.

### Negative

- A scheduler must retain and complete the returned token explicitly.
- P4 alone cannot prevent callers from bypassing future boundaries with raw
  low-level helpers.

### Neutral

- P1 carrier identity, P3 sidecar acquisition, direct GPU kernels, and the
  package export contract remain unchanged.

## Implementation

### Required Changes

1. **P4 guard and token** (`particula/execution/gpu_session.py`)
   - Maintain the sole open token and post-completion bookkeeping.
   - Provide the side-effect-free `assert_step_closed()` operation gate.
2. **Pinned-binding validation** (`particula/execution/gpu_resources.py`)
   - Add no-acquisition exact-session validation for guard transitions.
3. **Focused regression coverage and architecture documentation**
   - Verify identity, failure preservation, no-allocation behavior, export
     isolation, and future-boundary gate use.

### Testing Strategy

Use focused Warp-optional session/registry tests to verify valid and invalid
transitions, registry drift rejection, identity-only tokens, non-allocation, and
the absence of conversion or synchronization. Run the direct GPU
process-sequence regression separately.

### Rollback Plan

Remove the concrete-only guard and its tests. No public API, resident-state
schema, sidecar manifest, transfer path, or lifecycle transition must be
migrated.

## Validation

### Success Criteria

- [x] One exact active session/registry binding permits only one open token.
- [x] Count and simulated time advance only after matching completion.
- [x] Guard transitions and binding validation neither execute, transfer,
  synchronize, acquire, nor allocate resources.
- [x] The guard and token remain absent from package-level export surfaces.

## References

- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [ADR-005: One-Time GPU Resident-Session Setup](ADR-005-one-time-gpu-resident-session-setup.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1487](https://github.com/Gorkowski/particula/issues/1487)

## Notes

This ADR extends ADR-004 and ADR-005 and supersedes none. It defines P4
bookkeeping only; P5/P6 retain scheduler execution and operational lifecycle
policy.
