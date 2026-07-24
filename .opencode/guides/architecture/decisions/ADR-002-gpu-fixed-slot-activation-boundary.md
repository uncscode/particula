# ADR-002: GPU Fixed-Slot Activation Boundary

**Status:** Accepted
**Date:** 2026-07-24
**Decision Makers:** ADW Development Team
**Technical Story:** [#1419](https://github.com/Gorkowski/particula/issues/1419)

## Context

GPU particle storage has fixed capacity. It needs a direct Warp activation
boundary that has the same deterministic placement semantics as the CPU slot
activation oracle without making transfer, resizing, or diagnostics ownership
implicit.

### Problem Statement

Expose a supported GPU activation entry point that places each selected request
prefix in the corresponding ascending free slot while preserving caller-owned
device storage and atomic rejection before mutation.

### Forces

**Driving Forces:**
- Fixed-capacity particle storage needs deterministic slot placement.
- Direct GPU calls need explicit ownership and no hidden host-device transfer.
- CPU activation provides an independent behavioral oracle for parity tests.

**Restraining Forces:**
- P3 diagnostics must remain concrete-module-only rather than expanding the
  package API.
- Validation must not inspect protected density or volume fields.
- Failure atomicity can only be guaranteed before a device writer launches.

## Decision

Export only `activate_slots_gpu` from `particula.gpu.kernels`, backed by
`particula.gpu.kernels.slot_management`. Keep P3 diagnostics concrete-module-
only and retain fixed-capacity, caller-owned Warp data as the sole execution
model.

### Chosen Option

**Option 2: Package-exported direct activation with concrete P3 diagnostics**

The P4 entry point will:

1. Map selected request-prefix ranks to ascending free fixed slots.
2. Read and write only particle masses, concentration, and charge.
3. Use caller-owned same-device `int32` activation and diagnostic sidecars.
4. Complete read-only preflight before launching the mutation writer.

## Alternatives Considered

### Option 1: Keep activation concrete-module-only

**Description:** Require callers to import the activation API directly from
`slot_management`.

**Pros:**
- Avoids growing the package export surface.

**Cons:**
- Treats the supported P4 operation like an internal diagnostic helper.
- Obscures the intended direct-kernel boundary.

**Reason for Rejection:** P4 is a stable direct execution boundary, unlike P3
diagnostics.

---

### Option 2: Package-exported direct activation with concrete P3 diagnostics
(chosen)

**Description:** Export the P4 mutation boundary lazily through
`particula.gpu.kernels`, but retain read-only P3 diagnostics in its concrete
module.

**Pros:**
- Delineates supported execution from lower-level inspection helpers.
- Preserves explicit device ownership and fixed-capacity behavior.

**Cons:**
- Callers needing P3 diagnostics use a longer concrete-module import.

**Reason for Selection:** It makes the P4 contract discoverable without
committing implementation diagnostics to the public package API.

---

### Option 3: Add activation to conversion or a high-level runnable

**Description:** Hide activation behind CPU↔GPU conversion or orchestration.

**Pros:**
- Could offer a shorter high-level workflow.

**Cons:**
- Blurs transfer, schema, and kernel-boundary responsibilities.
- Would introduce implicit data movement or broader lifecycle commitments.

**Reason for Rejection:** The supported scope is a direct, device-resident
kernel boundary.

## Rationale

Direct package export signals a supported low-level operation, while keeping
P3 diagnostics concrete-only prevents accidental API expansion. Read-only
preflight validates metadata, aliases, existing slots, selected request
prefixes, and capacity before mutation. The writer then uses precomputed
ascending free slots, preserving deterministic placement without changing
container capacity.

### Trade-offs Accepted

1. **Explicit sidecars:** Callers allocate and retain all `int32` count and
   diagnostic buffers.
2. **Bounded atomicity:** Rejected calls are mutation-free, but rollback is not
   promised after the writer starts.

## Consequences

### Positive

- A supported direct API has clear ownership and import boundaries.
- GPU placement is deterministic and comparable with the CPU oracle.
- Density and volume remain unobserved by activation.

### Negative

- Callers manage fixed capacity, device placement, synchronization, and
  sidecar allocation.
- The boundary supplies no resizing, compaction, hidden transfer, fallback, or
  runnable integration.

### Neutral

- P3 diagnostics remain available only from their concrete module.

## Implementation

### Required Changes

1. **Slot-management boundary**
   - Implement P4 preflight, placement, and caller-owned sidecar writes in
     `particula/gpu/kernels/slot_management.py`.
2. **Kernel export**
   - Lazily export only `activate_slots_gpu` from
     `particula/gpu/kernels/__init__.py`.
3. **Contract coverage**
   - Add CPU-oracle parity, sidecar identity, and preflight atomicity coverage
     under `particula/gpu/kernels/tests/`.

### Testing Strategy

Compare fixed-slot mutation and diagnostic sidecars with the CPU oracle across
multi-box, multi-species, zero-prefix, zero-capacity, repeated, and
exact-capacity cases. Verify rejected schema, state, count, request, capacity,
and alias inputs leave accessible caller-owned data unchanged. Warp CPU is the
baseline; CUDA is optional when available.

### Rollback Plan

Remove the lazy package export and P4 entry point. P3 diagnostics and the CPU
oracle remain independent.

## Validation

### Success Criteria

- [x] `activate_slots_gpu` is the only slot-management symbol exported by
  `particula.gpu.kernels`.
- [x] P3 diagnostics remain concrete-module-only.
- [x] P4 uses fixed-capacity caller-owned device data and `int32` sidecars.
- [x] Preflight rejection is write-free and successful placement is
  deterministic by ascending free index.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1419](https://github.com/Gorkowski/particula/issues/1419)

## Notes

No prior ADR is superseded.
