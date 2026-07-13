# Architecture Design

## High-Level Design

```text
condensation_step_gpu(..., reusable scratch)
  -> validate all inputs and scratch before mutation
  -> clear total-transfer accumulator
  -> repeat exactly 4 times (static host orchestration)
       -> refresh E4-F1 thermodynamics for current substep
       -> prepare dynamic viscosity / mean free path in reusable buffers
       -> calculate transfer using dt / 4 and current particles
       -> clamp and apply transfer in place
       -> accumulate applied transfer into total-transfer buffer
  -> return updated particles and caller-owned total transfer by identity
```

The count is a fixed constant, not a user-selected or convergence-dependent
loop. Each iteration observes the state produced by the prior iteration. The
design remains compatible with E4-F2 physics when E4-F4 combines the tracks.

## Data / API / Workflow Changes

- **Data model:** No fields are added to `WarpParticleData`, `WarpGasData`, or
  `WarpEnvironmentData`. Scratch arrays are explicit operation sidecars with
  stable `(n_boxes, n_particles, n_species)` or `(n_boxes,)` shapes.
- **API surface:** Extend `condensation_step_gpu()` with optional reusable
  scratch through typed, keyword-only operation sidecars while preserving
  existing positional compatibility. Keep the lazy export in
  `particula.gpu.kernels` unchanged.
- **Return semantics:** The transfer buffer records the sum of applied transfer
  across the full call, not only the fourth substep.
- **Workflow hooks:** E4-F1 is a hard gate. E4-F2 may proceed in parallel;
  E4-F4 consumes both tracks and must retain per-substep refresh placement.

## Validation and Mutation Ordering

Resolve dimensions and validate all environment inputs, E4-F1 configuration,
and supplied scratch arrays before clearing accumulators or launching a kernel.
Reject mismatched shape, dtype, device, species order, or physical parameters
with `ValueError`; particle and caller-owned state must remain unchanged.

## Security & Compliance

No new permissions, serialization, or network surface is introduced. Resource
robustness depends on bounded static work, explicit device matching, and no
hidden host-device transfers. fp64 remains the supported numeric contract.
