# ADR-016: Direct GPU Volume-Evolution Boundary

**Status:** Accepted
**Date:** 2026-08-08
**Decision Makers:** ADW Development Team
**Technical Story:** [#1508](https://github.com/Gorkowski/particula/issues/1508)

## Context

E7-F7 P1 already declares and read-only validates fixed-shape communication
maps in `particula.execution.communication`. A prescribed final volume needs a
separate device-resident operation that can preserve extensive particle and gas
inventories without turning the P1 declaration seam into an execution,
transport, resident-session, or scheduling boundary.

### Problem Statement

Provide a safe, concrete direct-Warp writer for prescribed per-box final
volumes while keeping P1 map validation and later communication phases separate.

### Forces

**Driving Forces:**
- Final-volume updates must preserve extensive inventory in fixed-shape resident
  storage.
- Invalid schemas, aliases, domains, factors, and scaled outputs must reject
  before mutation.
- Existing direct GPU contracts require explicit caller ownership and no hidden
  host transfer or fallback.

**Restraining Forces:**
- P1 must remain read-only and must not acquire a primary-state writer.
- P3+ still own transfer admission, transport, and resident integration.
- The change must not establish a package-level API or a `Runnable`.

## Decision

Create `particula.gpu.kernels.communication` as the concrete-only,
direct-import E7-F7 P2 final-volume evolution boundary. It exposes
`volume_evolution_step_gpu` only from its concrete module.

### Chosen Option

**Option 2: Separate direct-Warp final-volume writer (chosen)**

The writer accepts caller-owned active-device contiguous `wp.float64` final
volumes of shape `(B,)` in m³. After complete read-only preflight, it updates
only `particles.volume` and particle/gas concentrations by
`old_volume / final_volume`, returns the exact input containers, and preserves
their identities and extensive inventories.

## Alternatives Considered

### Option 1: Add volume mutation to P1 communication validation

**Description:** Extend `particula.execution.communication` to write primary
resident volumes and concentrations.

**Pros:**
- Keeps communication-related code in one module.

**Cons:**
- Breaks P1's declaration/read-only validation ownership.
- Couples map metadata to Warp primary-state mutation.

**Reason for Rejection:** P1 must remain a separate, non-mutating map boundary.

---

### Option 2: Separate direct-Warp final-volume writer (chosen)

**Description:** Place the isolated P2 mutation in the GPU kernels package.

**Pros:**
- Preserves explicit device-resident ownership and direct-kernel conventions.
- Keeps P1 and future transport phases independently evolvable.

**Cons:**
- Requires a longer concrete-module import.

**Reason for Selection:** It maintains phase and module boundaries without
prematurely introducing resident orchestration.

---

### Option 3: Integrate the writer with scheduler or session machinery

**Description:** Make volume evolution a resident scheduler node.

**Pros:**
- Could later centralize orchestration.

**Cons:**
- Couples an isolated primitive to unshipped transport policy and lifecycle.
- Implies scheduling, binding, and failure responsibilities outside P2 scope.

**Reason for Rejection:** P3+ own those future responsibilities.

## Rationale

The direct kernel boundary owns schema/domain/nonaliasing and scaling-safety
preflight immediately before the mutation it authorizes. This establishes an
atomic pre-launch rejection boundary while retaining the standard limitation
that rollback is not promised after an asynchronous writer launches. The
operation remains narrowly mathematical: scaling concentrations by the inverse
volume change preserves extensive inventories without changing protected state.

### Trade-offs Accepted

1. **Concrete-only import:** The writer is intentionally not re-exported.
2. **No orchestration:** Maps, transport, session binding, and scheduling remain
   outside the direct writer.

## Consequences

### Positive

- P1 retains its read-only communication-map contract.
- Final-volume updates preserve fixed-shape identities and extensive inventory.
- Equal-volume updates are write-free after full preflight.

### Negative

- Callers must manage active-device storage and synchronize before inspection.
- No rollback is guaranteed after an apply writer launches.

### Neutral

- P3+ continue to own transfer admission, transport, and resident integration.

## Implementation

### Required Changes

1. **Concrete direct writer** (`particula/gpu/kernels/communication.py`)
   - Validate primary schemas, domains, nonaliasing, factors, and scaled outputs.
   - Update only volume and particle/gas concentrations in place.
2. **Focused contract coverage** (`particula/gpu/kernels/tests/`)
   - Cover expansion, compression, inventory preservation, no-op, and rejected
     preflight behavior on Warp CPU with optional CUDA skips.
3. **Architecture documentation**
   - Record P2 ownership separately from P1 and P3+ communication phases.

### Testing Strategy

Compare independently calculated concentration scaling and extensive ledgers;
verify container/array identity and protected-field preservation; test invalid
schemas, aliases, domains, overflow, and underflow before writer launch.

### Rollback Plan

Remove the concrete P2 module and its documentation entry. P1 remains unchanged.

## Validation

### Success Criteria

- [x] P2 is concrete-only and direct-import-only.
- [x] P1 remains the separate map declaration/read-only validation boundary.
- [x] Only volume and particle/gas concentrations mutate after successful
  preflight, preserving extensive inventory.
- [x] No hidden transfer, synchronization, fallback, transport, scheduler,
  session binding, or `Runnable` is introduced.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1508](https://github.com/Gorkowski/particula/issues/1508)

## Notes

No prior ADR is superseded.
