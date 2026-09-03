# ADR-019: Prepared Resident Graph-Capture Qualification

**Status:** Accepted
**Date:** 2026-09-03
**Decision Makers:** ADW Development Team
**Technical Story:** [#1567](https://github.com/Gorkowski/particula/issues/1567)

## Context

E8-F1 supplies an attached resident graph-capture lifecycle, E8-F2 supplies a
prepared resident timestep, and E8-F3 publishes its resource set. Issue #1567
adds a host-only E8-F4 P1 qualification seam connecting those exact existing
records to a lazily resolved native callable vocabulary.

### Problem Statement

Qualify native graph-capture capability without widening exports, rebuilding
resident state, or implying native capture, replay, handle ownership, or cleanup
already exists.

### Forces

**Driving Forces:**
- Eligibility must bind exact existing lifecycle, prepared-state, and resource
  identities.
- Native-device qualification must be lazy and retain opaque `Device.native`.

**Restraining Forces:**
- P1 must not enter a guard token, dispatch work, or mutate lifecycle state.
- Native capture/replay, handles, cleanup, transfer, allocation, and
  synchronization are explicitly out of scope.

## Decision

Use `particula.execution.graph_capture` as a concrete-only E8-F4 P1 boundary:

1. Bind exact attached E8-F1 `READY`, E8-F2 prepared, and E8-F3 published
   records by identity.
2. Lazily qualify non-CPU Warp devices through a caller-owned adapter and retain
   validated native callable vocabulary by identity.
3. Return immutable metadata without changing `READY` or owning a native handle
   or cleanup operation.

### Chosen Option

**Option 2: Concrete qualification-only controller (chosen)**

Structural validation precedes adapter access. CPU and Warp-CPU reject before
probing; non-CPU Warp native identifiers pass unchanged to the adapter. The
controller retains callable references but never invokes them.

## Alternatives Considered

### Option 1: Capture during qualification

**Description:** Begin/end capture and publish native handles as part of P1.

**Pros:** Reduces later setup.

**Cons:** Couples probing to lifecycle mutation, handle ownership, and cleanup.

**Reason for Rejection:** It implements deferred P2/P3 work.

---

### Option 2: Qualification-only controller (chosen)

**Description:** Validate exact records and retain callable vocabulary only.

**Pros:** Preserves lifecycle and ownership boundaries; enables later phases.

**Cons:** Capture/replay still needs later implementation.

**Reason for Selection:** It supplies eligibility without unsupported execution.

---

### Option 3: Package-level export

**Description:** Re-export the controller through package APIs.

**Pros:** Shorter import path.

**Cons:** Makes an incomplete internal lifecycle seam public.

**Reason for Rejection:** The resident composition architecture is concrete-only.

## Rationale

Exact binding prevents reuse with different resident primaries, resources,
prepared operations, or lifecycle metadata. Lazy adapter qualification avoids
runtime imports and keeps native identifiers opaque. Separating callable
retention from invocation makes the P2/P3 boundary enforceable.

### Trade-offs Accepted

1. Callers use the concrete module import path.
2. Qualification alone cannot capture or replay a timestep.

## Consequences

### Positive

- Qualification is bounded host-metadata work and preserves `READY`.
- Invalid state fails before native callable invocation.
- Native-handle and cleanup ownership remain unambiguous.

### Negative

- Native capture/replay remains unavailable after qualification.
- Callers provide the lazy runtime/device/API adapter.

### Neutral

- Package and top-level exports remain unchanged.

## Implementation

### Required Changes

1. **Qualification controller** (`particula/execution/graph_capture.py`)
   - Bind E8-F1/E8-F2/E8-F3 records and retain adapter callables without use.
2. **Tests** (`particula/execution/tests/graph_capture_test.py`)
   - Cover identity gates, adapter order, READY preservation, and no handle or
     cleanup ownership.
3. **Export tests** (`particula/execution/tests/exports_test.py`)
   - Deny E8-F4 P1 names from package and top-level exports.

### Testing Strategy

Use host-only recording adapters to verify ordered checks, no adapter use after
structural rejection, and no callable invocation or guard-token entry. Verify
that success and failures preserve the exact `READY` lifecycle.

### Rollback Plan

Remove the qualification records and operation; E8-F1 lifecycle, E8-F2
preparation, and E8-F3 resource publication remain independent.

## Validation

### Success Criteria

- [x] Exact E8-F1/E8-F2/E8-F3 identities are required.
- [x] CPU and Warp-CPU reject before adapter use.
- [x] Qualification preserves `READY` and owns no handle or cleanup owner.
- [x] Capture, replay, token entry, dispatch, transfer, allocation, and
  synchronization are excluded.
- [x] The names remain concrete-module-only.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [ADR-015](ADR-015-execution-public-surface-and-experimental-gpu-policy.md)

## Notes

No prior ADR is superseded. P2/P3 capture/replay and handle-cleanup work is
deferred and is not documented as implemented.
