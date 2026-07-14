# Architecture Design

Issue #1302 implemented the P1 preflight and gate boundary. Before mutable
condensation work, `condensation_step_gpu()` validates an active-device
`wp.int32` binary `gas.partitioning` array with exact shape
`(n_boxes, n_species)` and all supplied optional P2 fp64 `(n_boxes, n_species)`
sidecars. A status-only Warp validation launch observes binary values; failures
raise `ValueError` without clearing outputs, refreshing vapor pressure, or
mutating particles, gas, or caller-owned scratch.

For each of exactly four substeps, production code computes a raw per-particle
transfer, then a private Warp gate zeros it for a disabled
`partitioning[box, species]` entry or a particle slot with zero concentration
before application. Enabled active entries retain the existing clamp and
accounting behavior. `gas.concentration` remains unchanged.

Issue #1303 implements the first four operations below as private kernels and
`_finalize_inventory_limited_mass_transfer(...)`, a direct-test-only helper.
Its already-gated fp64 proposal is read-only; it validates all inputs before
launch, bounds evaporation, reduces in fixed particle-index order without
atomics, writes finalized transfer and resolved P2 sidecars, and mutates only
particle masses. Gas remains read-only. `condensation_step_gpu()` does not call
this helper or launch its kernels, so coupled orchestration remains P3--P4 work:

1. Reduce concentration-weighted positive and negative transfer independently
   into `(n_boxes, n_species)` fp64 scratch.
2. Add same-substep evaporation to available gas and calculate a `[0, 1]`
   positive-transfer scale for each box/species.
3. Apply positive scales and derive the exact finalized aggregate transfer.
4. In P3--P4, update gas concentration with the opposite aggregate, accumulate
   finalized transfer for return and E4-F4 energy bookkeeping, and refresh
   gas-dependent physics before the next substep.

The public P1 path validates P2 sidecars but neither allocates, reads, clears,
nor writes them; issue #1303's helper resolves only its three P2 reduction,
release, and scale sidecars. Later public reductions must retain this
transactional preflight boundary. Gas concentration remains kg/m3 and particle
mass remains per particle, so future concentration weighting and box-volume
conventions must match the CPU reference explicitly.
