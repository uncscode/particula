# ADR-021: Graph-Capture Teardown Ownership

**Status:** Accepted
**Date:** 2026-09-03
**Decision Makers:** ADW Development Team
**Technical Story:** [#1570](https://github.com/Gorkowski/particula/issues/1570)

## Context

P2-issued resident native graph handles are authenticated through private
issuance provenance. Structural binding drift and resident terminal/fault paths
can make a captured graph unsafe to replay. Those paths span graph capture,
resident-session cleanup, checkpoint finalization, and resource stream writers.

### Problem Statement

Ensure stale captured records fail provenance before replay/token entry and that
each issued handle is released at most once, without allowing adjacent resident
modules to own opaque handles or captured-lifecycle transitions.

### Forces

**Driving Forces:**
- Native handle release and provenance invalidation must have one authority.
- Writer faults and terminal transitions must invalidate old replay records.
- Notification must retain exact resident session, registry, and closed-guard
  identity.

**Restraining Forces:**
- Concrete resident seams must not introduce public API or checkpoint payloads.
- Adjacent modules must not import graph-capture types eagerly or inspect opaque
  native handles.
- No retry, rollback, fallback, or automatic recapture is available after a
  possible writer launch.

## Decision

`particula.execution.graph_capture` is the sole owner of issued native handles,
provenance removal, native release, and graph-capture lifecycle transitions.

1. It removes an affected record from issuance provenance before exactly one
   release for structural drift, writer fault, finalization, close/discard, and
   explicit retirement.
2. It transitions the graph to a nondispatchable successor before surfacing a
   release failure; later notifications are idempotent and do not retry release.
3. `gpu_session`, `checkpoint`, and `gpu_resources` may only invoke a lazy,
   private exact-context notification seam. They neither inspect a handle nor
   transition graph lifecycle state.
4. Renewal retains metadata-only READY behavior; only a fresh P1/P2 capture can
   issue a new authentic handle.

### Chosen Option

**Option 2: Graph-capture-owned teardown with exact-context notification**

The graph-capture module retains the private session attachment and centralizes
teardown. Other concrete owners report an exact resident event and retain their
own session/checkpoint/resource responsibilities.

## Alternatives Considered

### Option 1: Let each terminal owner release its graph handle

**Description:** Session, checkpoint, and resource modules import graph-capture
records and release handles during their own cleanup.

**Pros:** Local terminal code could release directly.

**Cons:** Duplicates ownership, risks double release, and exposes opaque-handle
and lifecycle mechanics outside graph capture.

**Reason for Rejection:** It breaks the single authority required for native
release and provenance invalidation.

---

### Option 2: Graph-capture-owned teardown with exact-context notification

**Description:** One private graph-capture helper invalidates provenance,
releases the native handle, and transitions lifecycle state after exact-context
notification.

**Pros:** Preserves exact-once release, fail-closed replay rejection, and module
ownership boundaries.

**Cons:** Terminal paths depend on a lazy private notification seam.

**Reason for Selection:** It centralizes native authority without adding a
public coupling or import-time dependency.

---

### Option 3: Leave provenance valid until release succeeds

**Description:** Remove issuance provenance only after a successful native
release callback.

**Pros:** Retains provenance while release errors are reported.

**Cons:** A release failure could leave a stale record eligible for replay.

**Reason for Rejection:** Stale replay must fail before token entry regardless
of release outcome.

## Rationale

Opaque native graph handles and their release protocol are graph-capture
concerns. Central teardown ensures every invalidating event first makes replay
fail closed, then makes one best-effort release. Exact-context notification
preserves resident binding authority while preventing session, checkpoint, and
resource code from acquiring graph lifecycle responsibilities.

### Trade-offs Accepted

1. **Lazy private coupling:** Terminal owners perform a local lazy import only
   when they have an exact session/registry/guard context.
2. **No recovery after release failure:** The record remains stale and release is
   not retried, even when the native callback raises.

## Consequences

### Positive

- Old records are unreplayable before token entry or native launch.
- Each issued native handle has one release authority and at most one release.
- Replay provenance cannot be restored by renewal or a later terminal call.

### Negative

- Release failures can leave native runtime cleanup incomplete while graph state
  remains nondispatchable.
- Exact-context validation can reject incorrectly associated notifications.

### Neutral

- Public exports, checkpoint schemas/payloads, and explicit transfer boundaries
  remain unchanged.

## Implementation

### Required Changes

1. **Graph-capture teardown** (`particula/execution/graph_capture.py`)
   - Attach exact resident bindings privately and centralize provenance removal,
     native release, and lifecycle transition.
2. **Resident notifications** (`particula/execution/gpu_session.py`,
   `checkpoint.py`, and `gpu_resources.py`)
   - Call only the lazy private exact-context notification seam on applicable
     writer-fault and terminal paths.
3. **Lifecycle regressions** (`particula/execution/tests/`)
   - Verify stale provenance, exact-once release, failure ordering, and
     recapture independence.

### Testing Strategy

Use deterministic fake native callbacks to cover structural drift, writer
faults, finalization, close/discard, retirement, raising release callbacks, and
renewal followed by a distinct recapture. Assert old records reject before token
entry and no path releases a handle twice.

### Rollback Plan

Remove the P4 notification and centralized teardown changes. This is not
recommended because distributed release ownership would reintroduce stale replay
and double-release risk.

## Validation

### Success Criteria

- [x] Invalidating events remove old-record provenance before replay admission.
- [x] Graph capture performs at most one release per issued handle.
- [x] Only graph capture owns native handles and capture lifecycle transitions.
- [x] Adjacent resident modules use exact-context lazy notification only.
- [x] No public API, checkpoint schema, retry, rollback, fallback, or automatic
  recapture is introduced.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [ADR-006](ADR-006-resident-gpu-step-lifecycle-guard.md)
- [ADR-007](ADR-007-resident-session-checkpoint-finalize-restart.md)
- [ADR-008](ADR-008-resident-session-failure-close-semantics.md)
- [ADR-019](ADR-019-prepared-resident-graph-capture-qualification.md)
- [ADR-020](ADR-020-authenticated-native-resident-graph-replay.md)

## Notes

This ADR extends ADR-019 and ADR-020. It supersedes no prior ADR.
