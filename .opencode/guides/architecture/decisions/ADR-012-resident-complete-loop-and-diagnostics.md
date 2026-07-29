# ADR-012: Resident Complete Loop and Closed Diagnostics

**Status:** Accepted
**Date:** 2026-07-29
**Decision Makers:** ADW Development Team
**Technical Story:** [#1497](https://github.com/Gorkowski/particula/issues/1497)

## Context

The dependency-neutral scheduler resolves process metadata but intentionally
does not execute backend work. Resident state updates, direct process adapters,
and thermodynamic freshness coordination now need one bounded owner for a
complete timestep. Resident inspection also needs current gas and saturation
snapshots without exposing callbacks or mutable resident internals.

### Problem Statement

Compose one exact ten-node resident timestep and its supported diagnostics
without broadening package exports, adding hidden transport, or weakening the
resident lifecycle and post-launch failure contracts.

### Forces

**Driving Forces:**
- A complete loop must use resolver-produced node identities and order.
- Every loop must retain one exact active session, pinned registry, and closed
  guard binding.
- Diagnostics need separate caller-owned outputs and current derived state.

**Restraining Forces:**
- `particula.execution.scheduler` must remain dependency-neutral metadata.
- Direct GPU kernels and existing adapters retain physical and payload
  validation ownership.
- A writer that may have launched cannot promise rollback.

## Decision

Add two concrete, direct-import-only execution seams:

1. `particula.execution.diagnostics` provides exactly ordered gas-concentration
   and saturation-ratio snapshots into validated caller-owned `(B, S)` outputs.
2. `particula.execution.resident_scheduler` composes only the resolver-produced
   ten-node loop, opens one guard token after preflight, dispatches the resolved
   order, and completes that token only after all work succeeds.

Virtual vapor-pressure and saturation-refresh nodes are consumed only through
the thermodynamic consumer windows for condensation and diagnostics. Neither
module is exported through `particula.execution` or top-level `particula`.

### Chosen Option

**Option 2: Concrete identity-bound composition and closed diagnostics**

Keep runtime composition and diagnostic copying in narrow Warp-dependent
modules that bind exact resident and resolver metadata by identity.

## Alternatives Considered

### Option 1: Execute the loop in `execution.scheduler`

**Pros:** Colocates resolved order and dispatch.

**Cons:** Makes the dependency-neutral metadata seam import Warp and own
resident mutation/lifecycle failure behavior.

**Reason for Rejection:** It violates the declaration-only scheduler boundary.

---

### Option 2: Concrete identity-bound composition and closed diagnostics
(chosen)

**Pros:** Preserves resolver provenance, direct-kernel ownership, and explicit
lifecycle semantics while supporting the complete loop.

**Cons:** Callers use concrete imports and provide all exact request carriers.

**Reason for Selection:** It supplies the required composition without creating
a public resident runtime or general callback API.

---

### Option 3: General diagnostic callback registration

**Pros:** Allows arbitrary inspection extensions.

**Cons:** Exposes resident internals, creates callback lifecycle ambiguity, and
prevents a bounded ownership and aliasing contract.

**Reason for Rejection:** The initial diagnostic contract is deliberately closed.

## Rationale

Resolved graph and schedule provenance prevents a hand-written or foreign order
from authorizing resident writes. A closed two-operation protocol makes output
ownership and no-aliasing validation explicit. One token spans the entire
composition boundary: failures before writer-capable dispatch remain read-only;
after a writer may launch, the token closes and the exact session faults with no
rollback guarantee.

### Trade-offs Accepted

1. **Fixed scope:** Only the canonical ten-node schedule and two diagnostics
   are supported.
2. **Concrete imports:** Runtime carriers and executors remain unexported.
3. **No transactional recovery:** Observable device mutation may remain after a
   post-launch failure.

## Consequences

### Positive

- Complete resident loops preserve one exact lifecycle and resolved ordering.
- Diagnostics copy current resident state without callbacks or host transfers.
- Empty valid output schemas are successful write-free no-ops.

### Negative

- Alternate schedules, arbitrary diagnostics, resource replacement, and
  fallback are outside the boundary.
- Valid composition requires substantial exact identity/provenance preflight.

### Neutral

- Existing state updates, adapters, thermodynamic writes, and direct kernels
  retain their own validation and mutation authority.

## Implementation

### Required Changes

1. **Closed diagnostics** (`particula/execution/diagnostics.py`)
   - Validate plan provenance, ordered operations, and caller-owned outputs.
   - Copy only gas concentration and saturation ratio on the resident device.
2. **Complete-loop composition** (`particula/execution/resident_scheduler.py`)
   - Preflight exact bindings and duration agreement before one token opens.
   - Dispatch resolver order and route thermodynamic consumers through their
     freshness windows.
3. **Architecture documentation**
   - Record direct-import-only scope, lifecycle, exclusions, and ADR index.

### Testing Strategy

Test exact ten-node ordering, one-token success, diagnostic ordering/current
state, output nonaliasing and empty schemas, pre-dispatch active-session
failures, post-launch faulting, identity stability, and no implicit transport.
Use Warp CPU as the baseline and skip CUDA cleanly when unavailable.

### Rollback Plan

Remove the two concrete modules and their tests. No package export, resident
container schema, checkpoint format, or direct-kernel API requires migration.

## Validation

### Success Criteria

- [x] Only the exact resolver-produced ten-node schedule is accepted.
- [x] Diagnostics support only ordered gas and saturation snapshots with
  separately owned validated outputs.
- [x] One successful loop begins and completes one guard token; a possible
  post-launch failure faults the session without rollback.
- [x] Normal composition performs no implicit transfer, synchronization,
  checkpointing, fallback, or resource replacement.

## References

- [ADR-006: Resident GPU Step Lifecycle Guard](ADR-006-resident-gpu-step-lifecycle-guard.md)
- [ADR-009: Resident Process Delegation Adapters](ADR-009-resident-process-delegation-adapters.md)
- [ADR-011: Resident Thermodynamic Freshness Coordinator](ADR-011-resident-thermodynamic-freshness-coordinator.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1497](https://github.com/Gorkowski/particula/issues/1497)

## Notes

This ADR extends the resident-session architecture and supersedes none.
