# ADR-010: Resident State Update Boundary

**Status:** Accepted
**Date:** 2026-07-29
**Decision Makers:** ADW Development Team
**Technical Story:** [#1495](https://github.com/Gorkowski/particula/issues/1495)

## Context

Resident-session execution retains active Warp particle, gas, and environment
containers by identity. The resolved process graph declares canonical
`environment_update` and `gas_update` nodes, but scheduling and derived-state
refresh remain separate E7-F5 responsibilities. Callers need a bounded way to
replace prescribed resident environment temperature/pressure or gas
concentration without replacing containers or widening the public API.

### Problem Statement

Provide identity-bound, graph-authorized resident state updates while preserving
fixed resident ownership, direct-import-only constraints, and the separation of
updates from scheduling and derived-state refresh.

### Forces

**Driving Forces:**
- Update requests must be bound to the exact active session, pinned registry,
  resolved graph, and canonical graph node.
- Valid updates must preserve container and primary-array identities and mutate
  only the designated arrays.
- Invalid metadata, ownership, or values must be rejected before a copy writer.

**Restraining Forces:**
- Public exports would create an unsupported resident-process API commitment.
- Scheduling, vapor-pressure/saturation refresh, transport, and lifecycle policy
  are owned by other boundaries.
- Empty canonical schemas must remain valid without unsafe empty-pointer work.

## Decision

Add `particula.execution.state_updates` as a concrete-only, direct-import
Warp-resident boundary. It provides frozen identity-oriented environment and gas
update requests and an executor. No names are exported through
`particula.execution` or top-level `particula`.

The executor validates the exact pinned session/registry binding,
resolver-produced graph provenance, exact graph membership, and canonical
`environment_update` or `gas_update` role, then input schema, nonaliasing, and
scalar payloads before committing. A successful
environment request copies only temperature then pressure; a successful gas
request copies only concentration. The existing resident arrays and containers
remain authoritative. Canonical zero-box and zero-species schemas are successful
write-free no-ops.

### Chosen Option

**Option 2: Concrete identity-pinned state-update boundary**

Keep narrowly prescribed resident copies in an execution-layer module that is
bound to the resolved graph but does not execute graph policy.

## Alternatives Considered

### Option 1: Put copies in the scheduler or process graph

**Pros:** Centralizes graph-related names.

**Cons:** Mixes declaration or ordering policy with Warp payload validation and
mutation, and implies scheduler execution responsibilities.

**Reason for Rejection:** The scheduler and graph remain metadata/policy seams,
not resident mutation boundaries.

---

### Option 2: Concrete identity-pinned state-update boundary (chosen)

**Pros:** Enforces exact resident and canonical-node ownership while limiting
mutation to prescribed arrays.

**Cons:** Callers must retain the resolved graph and exact canonical node.

**Reason for Selection:** It supplies the required mutation seam without adding
scheduling, freshness, or public orchestration policy.

---

### Option 3: Expose a public update API or replace resident containers

**Pros:** Offers a simpler high-level caller surface.

**Cons:** Weakens resident identity guarantees and commits to unresolved public
lifecycle, transfer, fallback, and compatibility policy.

**Reason for Rejection:** Resident boundaries intentionally remain direct-import
only and preserve fixed container identities.

## Rationale

Exact identity binding and resolver-produced graph provenance prevent stale
sessions, foreign registries, manually constructed graph roles, and unrelated
nodes from authorizing writes. Completing read-only validation before the copy
sequence makes preflight rejection
write-free. Retaining updates separately from P5 leaves vapor-pressure and
saturation-ratio invalidation and refresh policy explicit rather than hidden.

### Trade-offs Accepted

1. **Direct imports:** The carriers and executor are unsupported package-level
   APIs.
2. **Limited mutation:** The boundary cannot update particle volume or any
   untargeted resident field.
3. **No post-launch recovery:** Rollback is not promised if an asynchronous copy
   writer fails after launch.

## Consequences

### Positive

- Prescribed updates preserve resident identities and protect untargeted state.
- Canonical graph-node binding makes the allowed mutation role explicit.
- Empty resident schemas have intentional, safe no-op semantics.

### Negative

- Callers must construct and retain exact request dependencies.
- Derived environment and gas state must be refreshed by later explicit work.

### Neutral

- No package export, scheduler, lifecycle transition, resource acquisition,
  transport, host transfer, or fallback behavior is added.

## Implementation

### Required Changes

1. **Resident state updates** (`particula/execution/state_updates.py`)
   - Add frozen exact-identity request carriers and deterministic preflight.
   - Copy only the canonical environment or gas fields in place.
2. **Regression coverage** (`particula/execution/tests/state_updates_test.py`)
   - Verify binding order, identity preservation, no-writer rejection, canonical
     empty-schema no-ops, and import isolation.
3. **Architecture documentation**
   - Document the direct-only boundary and non-goals in the architecture guide
     and outline.

### Testing Strategy

Use focused Warp-optional tests for exact binding and canonical-role rejection,
schema and alias rejection, finite physical payload checks, copy order, target
identity, untouched primary state, empty-schema no-ops, and absence from package
exports.

### Rollback Plan

Remove the concrete module and its tests. No public API, serialized format, or
resident container schema migration is required.

## Validation

### Success Criteria

- [x] Requests require exact active session, registry, graph, and canonical node
  identities.
- [x] Successful calls copy only the prescribed fields in place.
- [x] Rejected preflight performs no writer and canonical empty schemas are
  write-free no-ops.
- [x] The module remains direct-import-only with no scheduling, refresh,
  transport, lifecycle, transfer, or fallback behavior.

## References

- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [ADR-006: Resident GPU Step Lifecycle Guard](ADR-006-resident-gpu-step-lifecycle-guard.md)
- [ADR-009: Resident Process Delegation Adapters](ADR-009-resident-process-delegation-adapters.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1495](https://github.com/Gorkowski/particula/issues/1495)

## Notes

This ADR extends the concrete resident-session architecture and supersedes none.
