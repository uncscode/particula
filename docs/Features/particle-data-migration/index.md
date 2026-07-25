---
title: ParticleData and GasData Migration
---

# ParticleData and GasData Migration Guide

Use this section to migrate from the legacy `ParticleRepresentation` and
`GasSpecies` facades to the data-first `ParticleData` and `GasData` workflow.
The facades remain available for backward compatibility, but they are
deprecated and emit migration guidance in their logs.

If you arrived from the legacy path `docs/migration/particle-data.md`, that
page redirects here.

## Migration topics

- [Container and facade migration](containers.md): before/after examples,
  field mappings, gradual migration, and conversion helpers.
- [Dynamics migration](dynamics.md): current CPU support boundaries,
  single-box and multi-box guidance, and direct GPU condensation inputs.
- [Troubleshooting](troubleshooting.md): shape, environment-input, sidecar,
  synchronization, and reproduction guidance.

## Overview

Before adding container fields or changing CPU-to-GPU conversion behavior,
start with the canonical
[Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
guide for the shipped schema, shape, helper, and support-boundary contract.

For roadmap policy and planned follow-on work, review the roadmap's
[authoritative field ownership decisions](../Roadmap/data-oriented-gpu.md#authoritative-field-ownership-decisions),
[canonical shape conventions for container workflows](../Roadmap/data-oriented-gpu.md#canonical-shape-conventions-for-container-workflows),
and [final downstream handoff map for sibling
features](../Roadmap/data-oriented-gpu.md#final-downstream-handoff-map-for-sibling-features).
Treat the [Mass Precision Recommendation
Report](../Roadmap/mass-precision-study.md) as the canonical reference before
changing particle mass dtype or schema behavior.

The migration moves **data** into dedicated containers and leaves **behavior**
in strategies and runnables:

- `ParticleData` stores per-particle arrays with an explicit batch dimension.
- `GasData` stores gas species arrays with an explicit box dimension; it does
  not own per-box thermodynamic state.
- `EnvironmentData` owns CPU-side per-box thermodynamic state with
  `temperature -> (n_boxes,)`, `pressure -> (n_boxes,)`, and
  `saturation_ratio -> (n_boxes, n_species)`.
- `ParticleData.volume` remains the authoritative per-box simulation-volume
  owner.
- `ParticleRepresentation` and `GasSpecies` remain as compatibility facades.

!!! note
    The explicit environment-state transfer boundary is
    `particula.gpu.WarpEnvironmentData`,
    `particula.gpu.to_warp_environment_data()`, and
    `particula.gpu.from_warp_environment_data()`. See
    [Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
    for the authoritative container and transfer contract.

!!! warning
    GPU-to-CPU gas restore is intentionally lossy unless ordered species
    metadata is preserved outside `WarpGasData`. GPU-only helper state such as
    `vapor_pressure` is also dropped on CPU restore.

## Why migrate

- **Clear data/behavior split**: data containers keep state, strategies keep
  physics.
- **Multi-box ready**: batch dimensions make CFD and multi-box simulations
  first-class.
- **Fewer implicit conversions**: attributes are explicit arrays rather than
  getter methods.

## Deprecation timeline

- **v0.3.0**: `ParticleRepresentation` and `GasSpecies` are deprecated and
  emit log warnings.
- **v1.0**: planned removal of the legacy facades.

## Related references

- `ParticleData` implementation: `particula/particles/particle_data.py`
- `GasData` implementation: `particula/gas/gas_data.py`
- Legacy facades: `particula/particles/representation.py` and
  `particula/gas/species.py`
