# ADR-008: Resident Session Failure and Terminal Close Semantics

**Status:** Accepted
**Date:** 2026-07-29
**Decision Makers:** ADW Development Team
**Technical Story:** [#1489](https://github.com/Gorkowski/particula/issues/1489)

## Context

ADRs 004--007 establish a concrete resident-session carrier, identity-pinned
resource registry, step guard, and explicit P5 checkpoint/finalize/restart
boundary. Direct resident operation owners need a bounded way to distinguish a
read-only failure from a failure after a device writer may have launched, and a
way to terminally dispose of a binding without treating close as runtime work or
recovery.

### Problem Statement

Define explicit failure outcomes and close/discard transitions without adding a
public API, changing P5 checkpoint behavior, or implying rollback, hidden
runtime work, or automatic recovery.

### Forces

**Driving Forces:**
- A read-only failed operation should not consume a guard step or make a valid
  resident binding unusable.
- A possible post-launch mutation cannot safely be retried or represented as a
  reusable active session.
- Terminal disposal must retain exact identity ownership and be idempotent.

**Restraining Forces:**
- Raw GPU helpers cannot be globally intercepted or reliably classified from an
  exception type.
- No rollback, synchronization, transfer, checkpoint, restore, restart, or
  fallback may be implicit.
- P5 finalization and cached-checkpoint identity must remain unchanged.

## Decision

Add P6 only to concrete `particula.execution.gpu_session`. A private outcome
classification supplied by the direct operation owner distinguishes `READ_ONLY`
from `WRITER_MAY_HAVE_LAUNCHED`; it is never inferred. With an exact active
session/registry/guard/token binding, failure handling aborts and releases the
open token without advancing guard counters or time. Read-only outcomes leave
the session `ACTIVE`; possible writer outcomes transition that exact session to
`FAULTED` and preserve the original operational exception. No rollback is
attempted.

Add concrete-only `close()` and `discard()` terminal lifecycle operations.
`ACTIVE -> CLOSED` validates the pinned registry binding and closed guard;
`FAULTED -> CLOSED` uses an identity-only closed-guard check. `CLOSED` and
`FINALIZED` repeated close paths are write-free no-ops. Close/discard do not
perform implicit runtime work, and P5 checkpoint/finalize/restart semantics are
unchanged.

### Chosen Option

**Option 2: Explicit owner-classified fault seam with identity-bound close**

Keep failure knowledge at the direct owner and keep lifecycle changes local to
the existing concrete session/guard/registry binding.

## Alternatives Considered

### Option 1: Infer failure outcome from exception type

**Pros:** Requires less explicit owner code.

**Cons:** Exception classes do not reliably indicate whether a device writer
launched, and raw helper calls remain outside this boundary.

**Reason for Rejection:** It could incorrectly mark a mutated session reusable.

---

### Option 2: Explicit owner-classified fault seam with identity-bound close

**Pros:** Preserves exact ownership, read-only reuse, and conservative
post-launch faulting without changing direct-kernel contracts.

**Cons:** Direct operation owners must classify their own known launch state.

**Reason for Selection:** Only the owner can make the bounded classification
without globally intercepting runtime work.

---

### Option 3: Close by checkpointing, restoring, or automatically retrying

**Pros:** Could appear to offer recovery.

**Cons:** Adds hidden synchronization, transfer, allocation, recovery policy,
and potentially unsafe assumptions about post-launch mutation.

**Reason for Rejection:** P6 is disposal/fault semantics, not recovery.

## Consequences

### Positive

- Read-only operation failures can leave an exact resident binding reusable.
- Possible writer failures conservatively prevent further resident operations.
- Close/discard has deterministic, identity-bound terminal behavior.

### Negative

- A faulted session has no rollback, retry, or automatic recovery route.
- Owners must explicitly classify failures when they know writer-launch state.

### Neutral

- Names remain concrete-only; package exports and P5 checkpoint/finalize/restart
  behavior do not change.

## Implementation

### Required Changes

1. **Failure seam** (`particula/execution/gpu_session.py`)
   - Add private outcome classification, exact-token abort, and original-error
     preservation for direct operation owners.
2. **Terminal lifecycle** (`particula/execution/gpu_session.py`)
   - Add identity-bound close/discard semantics with no implicit runtime work.
3. **Regression coverage** (`particula/execution/tests/gpu_session_test.py`)
   - Cover reuse, faulting, original errors, identity validation, terminal
     no-ops, and retained P5 finalization behavior.

### Testing Strategy

Use focused Warp-CPU lifecycle tests to verify token/counter preservation,
observable post-launch mutation, exact original-exception propagation, closed
binding requirements, and zero implicit close/discard runtime work. CUDA rows
remain optional and skip cleanly when unavailable.

### Rollback Plan

Remove the concrete P6 helpers and lifecycle methods. No public export, schema,
or persisted checkpoint format needs migration.

## Validation

### Success Criteria

- [x] Read-only failures release the exact token and preserve `ACTIVE` reuse.
- [x] Possible writer failures fault the exact session without rollback and
  preserve the operational exception.
- [x] Close/discard are identity-bound terminal operations with write-free
  `CLOSED`/`FINALIZED` no-ops.
- [x] P5 checkpoint/finalize/restart behavior and concrete-only exports remain
  unchanged.

## References

- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [ADR-006: Resident GPU Step Lifecycle Guard](ADR-006-resident-gpu-step-lifecycle-guard.md)
- [ADR-007: Resident Session Checkpoint, Finalize, and Restart Boundary](ADR-007-resident-session-checkpoint-finalize-restart.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1489](https://github.com/Gorkowski/particula/issues/1489)

## Notes

This ADR extends ADRs 004, 006, and 007 and supersedes none.
