# ADR-017: Direct GPU Particle-Transport Boundary

**Status:** Accepted
**Date:** 2026-08-08
**Decision Makers:** ADW Development Team
**Technical Story:** [#1510](https://github.com/Gorkowski/particula/issues/1510)

## Context

E7-F7 P1 owns fixed-shape communication-map declaration and read-only
validation, while P2 owns the isolated final-volume writer. Particle transport
needs a mutation boundary without coupling map declaration to execution or
introducing resident orchestration.

### Problem Statement

Provide deterministic, fixed-capacity device-resident particle transport for
prescribed closed `PARTICLES` maps while preserving particle inventories and
existing explicit-ownership boundaries.

### Forces

**Driving Forces:**
- Transport must preserve concentration-weighted particle number, mass, and
  signed charge.
- Destination placement must be deterministic and based on immutable pre-call
  state.
- Invalid plans must reject before primary mutation.

**Restraining Forces:**
- P1 remains read-only; P3 and P5 retain admission and resident binding.
- The direct GPU model forbids hidden transfer, synchronization, and fallback.
- Fixed capacity disallows resizing, compaction, or implicit activation.

## Decision

Add the concrete-only direct-import P4 seam
`particula.gpu.kernels.communication.ParticleCommunicationBuffers` and
`particle_communication_step_gpu`.

### Chosen Option

**Option 2: Immutable-plan, gated-commit direct kernel (chosen)**

The seam accepts only closed, in-domain `PARTICLES` maps. It creates requests
from immutable pre-call state, matches exact destination populations or reserves
ascending pre-step free slots, and executes one gated commit after the plan is
valid.

## Alternatives Considered

### Option 1: Add transport to the P1 declaration boundary

**Pros:** One communication module owns declaration and execution.

**Cons:** Violates P1 read-only ownership and couples map metadata to mutation.

**Reason for Rejection:** P1 must remain a declaration and validation seam.

---

### Option 2: Immutable-plan, gated-commit direct kernel (chosen)

**Pros:** Preserves deterministic fixed-capacity placement and pre-launch
atomicity while retaining explicit device ownership.

**Cons:** Requires concrete-module imports and caller-owned buffers.

**Reason for Selection:** It isolates P4 transport without prematurely adding
resident scheduling or public API scope.

---

### Option 3: Integrate transport with resident scheduling

**Pros:** Could centralize future orchestration.

**Cons:** Couples a direct primitive to deferred lifecycle and scheduling policy.

**Reason for Rejection:** P5 retains resident binding; scheduler integration is
outside P4.

## Rationale

Immutable planning prevents slots freed during a call from being reused and
makes destination selection independent of registration order. A single gated
commit preserves primary state on rejected plans, subject to the standard
no-rollback guarantee after an asynchronous writer launches.

## Consequences

### Positive

- Closed-map transport conserves weighted particle number, mass lanes, and
  signed charge.
- Exact-match and ascending-free-slot rules produce deterministic placement.
- P1, P3, and P5 ownership boundaries remain separate.

### Negative

- Callers allocate buffers, manage device placement, and synchronize before
  inspection.
- Transport remains unavailable through package-level imports or a runnable.

### Neutral

- Gas/volume mutation, RNG, transfers, fallback, resizing, compaction, and
  resident/scheduler integration remain out of scope.

## Implementation

1. Add the carrier and direct function to
   `particula/gpu/kernels/communication.py` without package exports.
2. Validate schemas, aliases, maps, slot state, and capacity before the primary
   commit; allow documented planning buffers to change during device planning.
3. Test a separate NumPy oracle, conservation, no-op, invalid-plan, Warp CPU,
   and optional CUDA behavior.

## Validation

- [x] Only closed, in-domain `PARTICLES` maps are accepted.
- [x] Planning uses immutable pre-call state and a single gated commit.
- [x] Successful calls preserve weighted particle number, mass, and charge.
- [x] No package export, hidden transfer/synchronization/fallback, RNG,
  resize/compaction, or resident/scheduler integration is introduced.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [ADR-016: Direct GPU Volume-Evolution Boundary](ADR-016-direct-gpu-volume-evolution-boundary.md)
- [Issue #1510](https://github.com/Gorkowski/particula/issues/1510)

## Notes

No prior ADR is superseded.
