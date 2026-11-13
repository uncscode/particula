# ADR-020: Authenticated Native Resident-Graph Replay

**Status:** Accepted
**Date:** 2026-09-03
**Decision Makers:** ADW Development Team
**Technical Story:** [#1569](https://github.com/Gorkowski/particula/issues/1569)

## Context

E8-F4 P2 captures an exact prepared resident binding and returns an opaque
native graph handle. Native replay must use only handles that P2 actually
issued, preserve the exact captured binding, and retain existing resident
guard and writer-failure semantics without widening the public API.

### Problem Statement

Provide one authenticated native replay operation without allowing forged or
tampered captured records, reusing stale bindings, or introducing hidden
runtime work and recovery behavior.

### Forces

**Driving Forces:**
- Native replay must forward the original opaque handle without inspecting it.
- Replay must remain bound to the exact resident session, resources, lifecycle,
  prepared timestep, and qualified device captured by P2.

**Restraining Forces:**
- The concrete execution boundary has no package or top-level public API.
- A native launch or post-launch completion failure may have written device
  state and cannot promise rollback.

## Decision

Use `particula.execution.graph_capture.replay_captured_resident_graph()` as a
concrete-only P3 native replay boundary:

1. Accept only an exact P2-issued `CapturedResidentGraph`; its opaque handle is
   absent from the carrier and retained in one private provenance record.
2. Revalidate provenance and the entire captured binding before token entry,
   including capture lifecycle/publication, qualified device, and duration.
3. On success, open exactly one `ResidentStepGuard` token, call native
   `capture_launch` exactly once with the retained handle, then complete that
   token.
4. Treat native-launch and post-launch completion errors as writer-capable:
   release the token through existing failure handling and fault resident and
   capture metadata without rollback, retry, fallback, recapture, or release.
5. Authenticate and acquire a per-record launch lease while holding the short
   provenance lock, then invoke the native launch outside that lock. A
   RELEASING or RELEASED/tombstoned record cannot acquire another lease.
6. Serialize replay admission with terminal session intent. An admitted replay
   retains its session and record leases through launch; close, discard, and
   finalize reject rather than race or deadlock on callback reentry.

### Chosen Option

**Option 2: Issuance-provenance replay with exact identity validation (chosen)**

P2 owns native begin/end/release and authentic handle publication. P3 owns only
one validated `capture_launch` call and guard-token completion.

## Alternatives Considered

### Option 1: Accept any structurally valid captured record

**Description:** Validate record fields but do not authenticate issuance.

**Pros:** No provenance registry is required.

**Cons:** Constructed or tampered records could forward arbitrary handles.

**Reason for Rejection:** Replay authority must remain with successful P2
publication.

---

### Option 2: Issuance-provenance replay with exact identity validation (chosen)

**Description:** Retain P2 issuance metadata and require exact identities at
replay time.

**Pros:** Rejects forged handles and stale/rebound resident state before launch.

**Cons:** Replay is intentionally coupled to the exact captured binding.

**Reason for Selection:** It provides fail-closed native replay while preserving
opaque-handle and concrete-only boundaries.

---

### Option 3: Re-capture or fall back after replay drift/failure

**Description:** Automatically create a replacement capture or execute a host
path.

**Pros:** Could make callers appear more resilient.

**Cons:** Adds runtime work, changes execution authority, and obscures
writer-capable failure state.

**Reason for Rejection:** Drift and post-launch errors must fail closed.

## Rationale

P2 issuance provenance establishes that a handle resulted from successful native
capture rather than caller construction. Exact binding validation preserves the
fixed resident graph assumptions while allowing ordinary payload and RNG-word
updates that retain array identities. Reusing `ResidentStepGuard` failure
handling makes the single native launch consistent with resident writer-failure
semantics.

### Trade-offs Accepted

1. Captured graphs cannot migrate across sessions, devices, or replaced
   resources.
2. Replay has no automatic recovery after native work may have launched.

## Consequences

### Positive

- Forged, tampered, stale, or incompatible captured records reject before token
  entry and launch.
- A successful replay has one token, one native launch, and one completion.
- No opaque-handle inspection, hidden transfer, synchronization, allocation, or
  fallback is introduced.

### Negative

- Callers must retain the authentic P2 record and its unchanged exact binding.
- Writer-capable failures fault the resident/capture lifecycle and provide no
  rollback.

### Neutral

- Package and top-level execution exports remain unchanged.

## Implementation

### Required Changes

1. **Replay provenance and gate** (`particula/execution/graph_capture.py`)
   - Retain P2-issued handle provenance privately and validate it by identity.
   - Revalidate captured metadata before one guarded native launch.
2. **Replay regression tests** (`particula/execution/tests/graph_capture_test.py`)
   - Cover authentic forwarding, drift rejection, no hidden work, and
     writer-capable failure semantics.
3. **Architecture documentation** (`.opencode/guides/architecture/`)
   - Record the concrete-only replay authority and constraints.

### Testing Strategy

Use controlled native bindings to verify exact-handle forwarding, one guard
token per successful replay, prelaunch rejection for provenance and binding
drift, and fault/no-rollback behavior after launch or completion failures.

### Rollback Plan

Remove the P3 replay operation and issuance registry. P1 qualification and P2
capture remain independent concrete-only boundaries.

## Validation

### Success Criteria

- [x] Only authentic P2-issued records can launch replay.
- [x] Accepted replay performs exactly one token, launch, and completion.
- [x] Identity/lifecycle/device/duration drift rejects before launch.
- [x] Post-launch failures fault without rollback, retry, fallback, or
  recapture.
- [x] Public exports and explicit transfer boundaries remain unchanged.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Architecture Reference](../../architecture_reference.md)
- [ADR-006](ADR-006-resident-gpu-step-lifecycle-guard.md)
- [ADR-019](ADR-019-prepared-resident-graph-capture-qualification.md)

## Notes

This ADR extends the E8-F4 graph-capture architecture; it does not supersede
ADR-019's qualification decision.
