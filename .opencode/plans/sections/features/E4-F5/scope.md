# Scope

## Delivered in P1 / issue #1302
- Validate `WarpGasData.partitioning` as active-device, binary `wp.int32` with
  exact `(n_boxes, n_species)` shape before any mutable condensation work.
- Validate supplied optional P2 reduction, release, and scale sidecars as
  active-device fp64 `(n_boxes, n_species)` metadata, without using them.
- Apply `WarpGasData.partitioning` and zero-concentration slot gates to raw
  proposals before application; retain no mutation of `gas.concentration`.
- Cover atomic failures and CPU↔Warp partitioning conversion with focused
  regression tests.

## Remaining in scope
- Bound evaporation by per-particle and aggregate particle inventory.
- Bound condensation by current per-box/per-species gas inventory, accounting
  for same-substep evaporation before scaling positive uptake.
- Update particle masses and gas concentration from one finalized fp64 transfer
  over exactly four E4-F3 substeps.
- Feed updated gas state into every subsequent thermodynamic refresh.
- Preserve E4-F4 latent-energy bookkeeping from the finalized transfer.
- Support single/multiple boxes and species, inactive slots, Warp CPU, and
  optional CUDA without hidden host transfers or container schema changes.
- Land the production hook and issue #1272 conservation regression together.

## Out of scope
- Gas inventory limiting, coupled gas mutation, and conservation claims in P1.
- New gas/container fields, adaptive substepping, or CPU/GPU synchronization.
- New activity, Kelvin, vapor-pressure, or latent-heat models (E4-F1/F2/F4).
- Performance diagnostics or debug-only public API (diagnostics: none).
- Claiming the complete E4 production envelope before E4-F6/E4-F7 gates pass.
