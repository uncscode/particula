# ADR-001: GPU Wall-Loss Direct Boundary

**Status:** Accepted
**Date:** 2026-07-23
**Decision Makers:** ADW Development Team
**Technical Story:** [#1403](https://github.com/Gorkowski/particula/issues/1403)

## Context

The GPU package needs a direct wall-loss entry point without prematurely
coupling host configuration, validation, neutral coefficient calculation,
particle removal, or RNG lifecycle.

### Problem Statement

Define a stable direct boundary that accepts neutral and charged
particle-resolved wall-loss inputs, rejects invalid calls without caller
mutation, and executes fixed-slot removal without changing the callable
interface.

### Forces

**Driving Forces:**
- Direct GPU APIs require deterministic, read-only preflight.
- Neutral and charged coefficient formulas already belong to focused device
  helpers.
- Fixed-slot execution needs a stable callable contract.

**Restraining Forces:**
- P3 must not create hidden transfers, fallback, or a runnable API.
- Configuration must not become an unintended package-level API.

## Decision

Use `particula.gpu.kernels.wall_loss` as the direct neutral-and-charged
wall-loss boundary. Export only `wall_loss_step_gpu` lazily from
`particula.gpu.kernels`; retain `NeutralWallLossConfig` as a concrete-module
API.

### Chosen Option

**Option 2: Separate boundary and coefficient-helper ownership**

The boundary owns immutable host configuration and ordered, read-only schema,
domain, environment, and RNG-metadata validation. The existing
`particula.gpu.dynamics.wall_loss_funcs` module remains the owner of neutral
and charged device coefficient helpers. The boundary performs frozen
preflight, then one device-resident selection/removal pass without host
readback or particle-sized temporary storage.

## Alternatives Considered

### Option 1: Combine validation and coefficients in the kernel boundary

**Description:** Move neutral and charged coefficient helpers into the
entry-point module.

**Pros:**
- Places all wall-loss code in one module.

**Cons:**
- Blurs device-physics and public-boundary responsibilities.
- Duplicates or relocates established helper ownership.

**Reason for Rejection:** It weakens the existing focused-helper boundary.

---

### Option 2: Separate boundary and coefficient-helper ownership (chosen)

**Description:** Keep preflight and future orchestration separate from device
coefficient calculations.

**Pros:**
- Preserves clear module responsibilities.
- Enables a write-free P3 contract and signature reuse in P4/P5.

**Cons:**
- Callers use a concrete-module import for configuration.

**Reason for Selection:** It preserves stable, explicit GPU API boundaries.

---

### Option 3: Expose configuration from package-level GPU modules

**Description:** Re-export `NeutralWallLossConfig` with the direct step.

**Pros:**
- Provides a shorter import path.

**Cons:**
- Commits a phase-specific configuration type to a broader public API.

**Reason for Rejection:** Only the entry point is currently a supported package
export.

## Rationale

The separation mirrors the GPU architecture: direct kernel entry points own
boundary contracts, while device physics remains in focused helper modules.
Read-only preflight establishes failure atomicity before mutable execution.
For charged work, nonzero-charge slots compose image-charge enhancement and
geometry-resolved field drift; image enhancement remains active at zero wall
potential. Spherical fields retain their signed scalar value before the signed
potential-derived contribution. Rectangular fields are caller-owned `(3,)`
`wp.float64` storage, reduced to Euclidean magnitude before that signed
potential contribution, so individual component signs do not select direction.
Charged zero-charge slots preserve the exact neutral coefficient and RNG path.

### Trade-offs Accepted

1. **Longer configuration import**: Callers import configuration from its
   concrete module.
2. **Bounded functionality**: The direct call mutates fixed particle slots,
   including clearing mass lanes, concentration, and charge for removed slots,
   but does not provide a runnable, hidden transfer, or CPU fallback.

## Consequences

### Positive

- Invalid calls do not mutate particles or supplied RNG sidecars.
- Positive-time execution keeps selection and optional caller-owned RNG state
  device-resident.
- Neutral and charged coefficient helpers retain a single owner.

### Negative

- Execution is limited to particle-resolved fixed-shape neutral and charged
  slots; nonzero charge uses charged composition while zero charge follows the
  neutral fallback.
- Rejected preflight is atomic, but rollback is not promised after a mutation
  kernel launch.

### Neutral

- CPU↔GPU transfers remain explicit and caller-owned.

## Implementation

### Required Changes

1. **Direct boundary**
   - Add configuration, read-only preflight, and fixed-slot execution in
     `particula/gpu/kernels/wall_loss.py`.
2. **Kernel export**
   - Lazily export only `wall_loss_step_gpu` from `particula/gpu/kernels/__init__.py`.
3. **Contract coverage**
   - Add focused Warp tests under `particula/gpu/kernels/tests/`.

### Testing Strategy

Exercise valid neutral and charged spherical/rectangular execution, invalid
schemas and domains, geometry-specific field resolution, zero-charge neutral
fallback, environment forms, slot clearing, and unchanged particle/RNG state
after rejected or true no-op calls. Warp CPU is the baseline; CUDA is optional
and skips cleanly when unavailable.

### Rollback Plan

Remove the lazy export and concrete boundary; existing coefficient helpers are
unchanged.

## Validation

### Success Criteria

- [x] Only `wall_loss_step_gpu` is lazily exported from the kernels package.
- [x] Configuration remains concrete-module-only.
- [x] Preflight is read-only; positive-time neutral and charged execution is
  device-resident and clears selected fixed slots in one pass.
- [x] Charged geometry resolution, zero-potential image enhancement, and exact
  zero-charge neutral fallback retain the bounded direct-only contract.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)

## Notes

No prior ADR is superseded.
