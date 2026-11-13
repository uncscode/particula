# ADR-011: Resident Thermodynamic Freshness Coordinator

**Status:** Accepted
**Date:** 2026-07-29
**Decision Makers:** ADW Development Team
**Technical Story:** [#1496](https://github.com/Gorkowski/particula/issues/1496)

## Context

Resident updates and direct processes can invalidate vapor pressure and
saturation ratio while the process graph and scheduler remain declaration-only
metadata seams. Condensation and diagnostics require those derived fields to be
fresh, but no existing concrete boundary records successful ordinary work and
orders refreshes immediately before those consumers.

### Problem Statement

Provide a narrow resident freshness boundary that uses resolved metadata to
order derived-field writers without turning the scheduler, state-update
executor, or process adapters into a general resident dispatcher.

### Forces

**Driving Forces:**
- Refreshes must bind exact active session, pinned registry, graph, schedule,
  and thermodynamic configuration carriers.
- Vapor pressure and saturation ratio must be refreshed on resident state in
  dependency order immediately before supported consumers.
- Caller-owned consumer execution and resident identities must remain explicit.

**Restraining Forces:**
- The scheduler must remain dependency-neutral declaration metadata.
- The existing vapor-pressure primitive remains the authoritative writer.
- Writer failures after a launch cannot offer a transactional rollback promise.

## Decision

Add `particula.execution.thermodynamic_updates` as a concrete-only,
direct-import Warp-dependent resident freshness coordinator. It is not exported
through `particula.execution` or top-level `particula`.

Callers report only successful ordinary nodes through `record_completed()`. The
coordinator owns a schedule cursor plus vapor-pressure and saturation-ratio
stale markers. `execute_consumer()` consumes immediately preceding virtual
refresh IDs when present, writes only stale fields in vapor-then-saturation
order, and invokes the next canonical condensation or diagnostics callback once.
Stale markers persist across ordinary scheduled nodes: for example,
`condensation -> wall_loss -> diagnostics` implicitly refreshes saturation
immediately before diagnostics even though no virtual refresh ID is adjacent to
diagnostics. It advances the consumer cursor and applies consumer invalidations
only after callback success.

### Chosen Option

**Option 2: Concrete identity-bound freshness coordinator**

Keep freshness ordering in a narrow execution-layer boundary that consumes
already-resolved graph and schedule metadata, rather than making a scheduler or
adapter infer state mutation.

## Alternatives Considered

### Option 1: Put refresh execution in the scheduler

**Pros:** Scheduling metadata and refresh IDs are colocated.

**Cons:** Couples a dependency-neutral metadata resolver to Warp, resident
payloads, and writer failure semantics.

**Reason for Rejection:** The scheduler must not import or execute backend work.

---

### Option 2: Concrete identity-bound freshness coordinator (chosen)

**Pros:** Preserves explicit caller dispatch, validates exact metadata binding,
and localizes stale-marker and partial-writer semantics.

**Cons:** Callers must report ordinary success and bracket supported consumers.

**Reason for Selection:** It supplies ordering without a full timestep loop or
hidden dispatch.

---

### Option 3: Let each update or process adapter refresh derived fields

**Pros:** No additional coordinator object for callers.

**Cons:** Duplicates ordering policy, obscures which calls succeeded, and makes
narrow copy/delegation boundaries responsible for downstream consumers.

**Reason for Rejection:** State updates and process adapters deliberately retain
their focused mutation and single-kernel delegation ownership.

## Rationale

Exact identity and resolver-provenance validation prevents foreign resident
state or fabricated canonical roles from authorizing derived-state writes.
Explicit successful-node reporting makes invalidation source ownership visible.
Virtual refresh nodes retain the schedule's ordering while permitting fresh
fields to elide only their writers, never a scheduled consumer.

### Trade-offs Accepted

1. **Direct imports:** The request and coordinator are not public APIs.
2. **Explicit reporting:** Callers own ordinary execution and must report it
   only after success.
3. **Partial writer state:** If vapor refresh succeeds and saturation refresh
   fails, vapor pressure remains fresh while saturation remains stale.

## Consequences

### Positive

- Condensation and diagnostics see required stale derived fields refreshed in
  one canonical order.
- The authoritative vapor-pressure primitive remains the only vapor writer.
- Saturation calculation stays on resident device arrays and preserves
  container and primary-array identities.

### Negative

- The coordinator is intentionally not a general scheduler or process callback
  registry.
- Partial writer success is observable; no rollback is provided after a writer
  launches.

### Neutral

- No lifecycle transition, resource acquisition, transfer, synchronization,
  conversion, fallback, or public export is added.
- Delegated vapor-pressure configuration validation retains its documented
  fingerprint reads; the coordinator does not read resident payloads.

## Implementation

### Required Changes

1. **Freshness coordinator** (`particula/execution/thermodynamic_updates.py`)
   - Bind exact resident and resolved-metadata carriers.
   - Delegate vapor pressure and launch private on-device saturation refresh.
2. **Focused tests** (`particula/execution/tests/thermodynamic_updates_test.py`)
   - Verify cursor/order, identity, failure markers, and device calculation.
3. **Architecture documentation**
   - Record the concrete-only boundary, exclusions, and ADR index.

### Testing Strategy

Use focused Warp-optional tests for exact binding, canonical role and cursor
ordering, stale-writer elision, saturation physics, callback suppression, and
non-transactional writer failures.

### Rollback Plan

Remove the concrete module and its tests. No public API, resident schema, or
serialized checkpoint format changes are required.

## Validation

### Success Criteria

- [x] Successful ordinary nodes explicitly drive coordinator stale markers.
- [x] Virtual writers run in vapor-then-saturation order before supported
  consumers and preserve consumer execution order.
- [x] The boundary remains concrete-only and does not acquire resources, own
  lifecycle, transfer, synchronize, fall back, or dispatch a full schedule.

## References

- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [ADR-009: Resident Process Delegation Adapters](ADR-009-resident-process-delegation-adapters.md)
- [ADR-010: Resident State Update Boundary](ADR-010-resident-state-update-boundary.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1496](https://github.com/Gorkowski/particula/issues/1496)

## Notes

This ADR extends the concrete resident-session architecture and supersedes none.
