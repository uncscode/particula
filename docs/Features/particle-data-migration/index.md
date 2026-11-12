---
title: Native aerosol migration
---

# Native aerosol migration

Migrate supported workflows to the flat native `Aerosol`, holding `ParticleData`,
full ordered `GasData`, and `EnvironmentData`. Use the selected E9-native checkout
for these examples; their presence here does not announce a published wheel or
release. M6 owns remaining legacy deletion and release readiness after the M5
validation handoff.

## Migration topics

- [Construction and replacement](construction.md): three-container assembly,
  native initializers, distribution measures, units and ownership.
- [Container access](containers.md): raw arrays, detached properties, individual
  and coordinated replacement, and historical API terminology.
- [Dynamics](dynamics.md): native single-box CPU processes, explicit species
  maps, and direct/resident GPU boundaries.
- [Troubleshooting](troubleshooting.md): validation, gas order, normalization,
  storage replacement and explicit transfer errors.

The executable examples preserve useful initialization and scientific content.
Historical facade names are reference material only; wrapping containers in a
legacy facade is not a native migration technique.

## State and behavior

`ParticleData.volume` owns represented volume. `GasData` retains all gas lanes,
including nonpartitioning species. `EnvironmentData` owns per-box temperature
and pressure and full gas-order saturation ratios. Activity, surface,
vapor-pressure and other physical configuration belong to processes.

Batched storage does not imply general multi-box CPU process support. The
native CPU runnable path remains single-box. Direct Warp and resident execution
use explicit preparation and transfer, with their documented process-specific
limits. Warp CPU is the installed-Warp baseline for uncaptured examples;
native graph capture requires CUDA and has no CPU fallback.

Start with [container foundations](../data-containers-and-gpu-foundations.md)
for the schema and [native access](containers.md) for units. GPU gas restore
requires externally retained names for semantic fidelity and drops device-only
vapor-pressure state. Checkpoint inspection carriers are lossy; canonical device
bytes, not those inspection carriers, govern explicit compatible restart.

## Historical release context

The E8 closeout planned coexistence in 0.2.13 and a breaking native migration in
0.3.0. These are release plans, not publication claims. This guide documents the
native migration target; temporary legacy implementation still awaiting M6 is
not endorsed for supported new examples. No performance qualification or
cross-device continuation is implied.
