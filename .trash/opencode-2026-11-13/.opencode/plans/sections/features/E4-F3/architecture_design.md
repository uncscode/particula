# Architecture Design

## High-Level Design

```text
condensation_step_gpu(..., scratch_buffers=...)
  -> validate all inputs and scratch before mutation
  -> normalize environment and resolve omitted sidecars after the gate
  -> clear resolved total once
  -> repeat exactly four times:
       refresh vapor pressure and environment properties
       calculate one raw transfer into work from current mass
       clamp, apply, and accumulate into total
  -> return the resolved total buffer by identity
```

Issue #1293 delivered the fixed-four static loop. Each interval uses
`time_step / 4.0`; thermodynamic and environment properties refresh in every
iteration before transfer calculation. The design remains compatible with
E4-F2 physics when E4-F4 combines the tracks.

## Data / API / Workflow Changes

- **Data model:** No fields are added to `WarpParticleData`, `WarpGasData`, or
  `WarpEnvironmentData`. Scratch arrays are explicit operation sidecars with
  stable `(n_boxes, n_particles, n_species)` or `(n_boxes,)` shapes.
- **API surface:** Extend `condensation_step_gpu()` with optional reusable
  scratch through typed, keyword-only operation sidecars while preserving
  existing positional compatibility. Keep the lazy export in
  `particula.gpu.kernels` unchanged.
- **Return semantics:** Work retains the final raw, pre-clamp proposal. Total is
  cleared once and accumulates only the mass-clamped applied transfers across
  all four substeps; the resolved total is returned by identity. Legacy
  `mass_transfer` remains that total when scratch transfer fields are omitted.
- **Workflow hooks:** E4-F1 is a hard gate. E4-F2 may proceed in parallel;
  E4-F4 consumes both tracks and must retain per-substep refresh placement.

## Validation and Mutation Ordering

Once dimensions and active device are known, validate all environment inputs,
E4-F1 configuration, and every supplied scratch field before environment
normalization, fallback allocation, refresh, clear, launch, or mutation. Reject
wrong sidecar type, shape, dtype, device, and overlap between `mass_transfer`
and a supplied scratch transfer field with `ValueError`; particle and
caller-owned state remain unchanged.

## Security & Compliance

No new permissions, serialization, or network surface is introduced. Resource
robustness depends on bounded static work, explicit device matching, and no
hidden host-device transfers. fp64 remains the supported numeric contract.
