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

## Delivered in P2 / issue #1303
- Add a private direct-test-only fp64 finalization helper in
  `particula/gpu/kernels/condensation.py` for already P1-gated proposals.
- Bound negative proposals by per-particle mass, reduce positive demand and
  negative release in fixed particle-index order, and scale positive uptake by
  gas inventory plus permitted release for each `(box, species)`.
- Apply and return the finalized direct-helper transfer while keeping
  `gas.concentration` read-only; atomically reject invalid proposals or P2
  sidecars before caller-owned state changes.
- Add independent NumPy-oracle, atomic-preflight, and public-P1-isolation
  coverage in `particula/gpu/kernels/tests/_condensation_test_support.py`.

## Remaining in scope
- Update particle masses and gas concentration from one finalized fp64 transfer
  over exactly four E4-F3 substeps.
- Feed updated gas state into every subsequent thermodynamic refresh.
- Preserve E4-F4 latent-energy bookkeeping from the finalized transfer.
- Support single/multiple boxes and species, inactive slots, Warp CPU, and
  optional CUDA without hidden host transfers or container schema changes.
- Land the production hook and issue #1272 conservation regression together.

## Out of scope
- Public invocation of P2 finalization, gas mutation, and conservation claims
  until P3--P4.
- New gas/container fields, adaptive substepping, or CPU/GPU synchronization.
- New activity, Kelvin, vapor-pressure, or latent-heat models (E4-F1/F2/F4).
- Performance diagnostics or debug-only public API (diagnostics: none).
- Claiming the complete E4 production envelope before E4-F6/E4-F7 gates pass.
