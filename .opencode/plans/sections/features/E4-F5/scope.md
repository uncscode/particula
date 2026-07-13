# Scope

## In scope
- Apply `WarpGasData.partitioning` before transfer reductions and mutation.
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
- New gas/container fields, adaptive substepping, or CPU/GPU synchronization.
- New activity, Kelvin, vapor-pressure, or latent-heat models (E4-F1/F2/F4).
- Performance diagnostics or debug-only public API (diagnostics: none).
- Claiming the complete E4 production envelope before E4-F6/E4-F7 gates pass.
