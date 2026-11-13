# ADR-014: Opt-In CPU Fallback Boundary

**Status:** Accepted
**Date:** 2026-07-30
**Decision Makers:** ADW Development Team
**Technical Story:** [#1502](https://github.com/Gorkowski/particula/issues/1502)

## Context

The E7-F6 availability resolver reports typed failures without selecting an
adapter or moving data. Some callers need an explicitly bounded way to run an
already CPU-authoritative request when a non-CPU request cannot be made
available or supported. Existing selection, resident, and direct-GPU boundaries
must retain their no-implicit-fallback behavior.

### Problem Statement

Allow an explicit CPU adapter selection only when it cannot conceal ownership,
transfer, lifecycle, or recovery behavior.

### Forces

**Driving Forces:**
- Availability and support failures have typed, inspectable reasons.
- CPU-authoritative callers can safely retain CPU state without data recovery.
- Adapter selection must remain context-local and identity-preserving.

**Restraining Forces:**
- Resident, uploaded, and mutated state cannot be safely treated as CPU state.
- Fallback must not broaden package exports or alter native adapter results.
- Automatic retry, restoration, synchronization, and rollback have separate
  ownership and failure semantics.

## Decision

Add `particula.execution.fallback` as a concrete, direct-import-only opt-in CPU
fallback policy boundary.

1. Its default `FallbackPolicy.RAISE` re-raises the exact eligible original
   typed error.
2. Explicit CPU policy accepts only typed availability/support reasons and
   caller-provided CPU-authoritative `PRE_UPLOAD` or `RESTORED` state.
3. After read-only preflight, it constructs the canonical CPU request, performs
   exactly one context lookup, and invokes the selected adapter once.
4. It records requested backend, selected CPU backend, and the original
   capability reason in its dispatch carrier without altering native
   `ExecutionResult.metadata`.

### Chosen Option

**Option 2: Isolated, default-deny fallback policy boundary**

The boundary consumes the original typed error and caller assertion, rather than
probing a runtime or inferring whether non-CPU state can be recovered.

## Alternatives Considered

### Option 1: Add automatic fallback to availability or selection

**Pros:** A shorter apparent execution path.

**Cons:** Conflates availability, adapter selection, and recovery policy; risks
hidden adapter changes and data movement.

**Reason for Rejection:** Package-level selection remains dependency-neutral and
selection-only.

---

### Option 2: Isolated, default-deny fallback policy boundary (chosen)

**Pros:** Keeps fallback explicit, typed, identity-aware, and limited to
CPU-authoritative state.

**Cons:** Callers must directly import the concrete module and supply explicit
state authority.

**Reason for Selection:** It permits the narrow supported case without changing
resident or direct-GPU ownership boundaries.

---

### Option 3: Recover resident or uploaded state automatically

**Pros:** Could provide a broader recovery experience.

**Cons:** Requires transfer, synchronization, lifecycle, restore, and rollback
semantics that are not owned by this boundary.

**Reason for Rejection:** State recovery is unsafe to infer and remains outside
this policy seam.

## Rationale

Typed capability failures are a stable boundary for policy, while CPU state
authority must be supplied explicitly because this module cannot establish it.
The default-deny policy prevents accidental backend substitution. Restricting
dispatch to one canonical CPU request and one adapter invocation preserves
context registration semantics and makes fallback provenance observable without
mutating adapter-owned result metadata.

### Trade-offs Accepted

1. **Direct import:** This policy is not a package-level convenience API.
2. **Caller assertion:** `RESTORED` is accepted as a caller assertion; the
   boundary neither restores nor verifies restoration.
3. **No recovery:** Errors after adapter invocation propagate unchanged, with no
   retry or rollback guarantee.

## Consequences

### Positive

- CPU fallback is explicit, typed, and fail-closed for non-authoritative state.
- The public selection surface and native result metadata remain unchanged.
- Callers can inspect fallback provenance separately from adapter metadata.

### Negative

- Only CPU-authoritative callers can use the boundary.
- Callers must register a CPU adapter and handle its native execution failures.

### Neutral

- No CPU↔GPU transfer, conversion, synchronization, lifecycle transition,
  checkpoint/finalize, restore, retry, or rollback behavior is introduced.

## Implementation

### Required Changes

1. **Fallback boundary** (`particula/execution/fallback.py`)
   - Define typed policy, boundary, authority, request, resolution, and dispatch
     carriers.
   - Validate typed errors and authoritative CPU state before one CPU lookup and
     one adapter dispatch.
2. **Contract evidence** (`particula/execution/tests/fallback_test.py`)
   - Cover default re-raise, eligible explicit dispatch, provenance, fail-closed
     state claims, and no recovery behavior.
3. **User-facing contract**
   - Document the direct-only boundary in the execution architecture and feature
     documentation.

### Testing Strategy

Use CPU-only fakes to verify one lookup and dispatch, carrier identity,
unmodified native result metadata, rejected state authority, and propagation of
adapter/result-validation failures without retry or recovery.

### Rollback Plan

Remove the isolated concrete module and tests. No package export, resident
session lifecycle, direct-GPU entry point, or native adapter result schema
depends on it.

## Validation

### Success Criteria

- [x] Default policy re-raises the original eligible typed error.
- [x] Explicit CPU policy accepts only CPU-authoritative `PRE_UPLOAD` and
  caller-asserted `RESTORED` state.
- [x] Resolution performs at most one canonical CPU lookup and dispatch invokes
  the retained adapter once.
- [x] Provenance is available without changing native result metadata.
- [x] The boundary provides no implicit movement, lifecycle, restoration,
  retry, or rollback behavior and has no package/top-level export.

## References

- [ADR-003: Dependency-Neutral Execution Capability Vocabulary](ADR-003-dependency-neutral-execution-capabilities.md)
- [ADR-013: Pre-Execution Availability Resolution](ADR-013-pre-execution-availability-resolution.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1502](https://github.com/Gorkowski/particula/issues/1502)

## Notes

This ADR extends E7-F6 fallback policy and supersedes no prior ADR. Existing
prohibitions on implicit fallback in selection, resident, and direct-GPU
boundaries remain in effect.
