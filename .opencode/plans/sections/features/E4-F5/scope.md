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

## Delivered in P3 / issue #1304
- Run exactly four equal public substeps in
  `particula/gpu/kernels/condensation.py`: validate each fresh P1-gated raw
  proposal, P2-finalize it, apply it to particles, accumulate the finalized
  total, and couple its concentration-weighted opposite delta to gas.
- Resolve/validate supplied P2 and property scratch once after aggregate
  preflight; allocate omitted fallback buffers once per successful call and
  reuse them across all substeps.
- Clear total and optional energy output once after preflight. Return the
  caller-owned total buffer by identity and derive signed energy from the
  finalized whole-call total.
- Add focused Warp wrapper/support regressions for oracle coupling, four-cycle
  ordering, empty/single-particle limits, scratch identity, and atomic
  preflight/proposal failure boundaries.

## Delivered in P4 / issue #1305
- Add regression-only concentration-weighted particle-plus-gas inventory checks
  to `particula/integration_tests/condensation_particle_resolved_test.py`:
  H2O and NH4HSO4 are accounted independently and gas-only N2 is exactly
  invariant.
- Add deterministic fp64 two-box public `condensation_step_gpu()` coverage in
  `particula/gpu/kernels/tests/_condensation_test_support.py`, with wrapper
  export checks in `particula/gpu/kernels/tests/condensation_test.py`.
- Verify per-box/per-species bookkeeping at `rtol=1e-12, atol=1e-30` separately
  from CPU-oracle particle/gas parity at `rtol=2e-10, atol=1e-30`; exercise
  uptake, evaporation, disabled partitioning, zero gas, and inactive slots on
  Warp CPU plus guarded optional CUDA.

## Delivered in P5 / issue #1306
- Update `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/data-containers-and-gpu-foundations.md` with the verified
  bounded P1-P4 direct-kernel contract, including active-device gas ownership,
  binary partitioning, P2-finalized inventory-limited accounting, fixed
  four-substep coupling, scratch mutability/identity, and no hidden transfers.
- Qualify #1305/#1272 evidence as direct-kernel and production-hook regression
  only; retain E4-F6/E4-F7 as gates for broader cross-device and final support
  claims.

## Out of scope
- Exporting P2 internals or adding a public entry point.
- New gas/container fields, adaptive substepping, or CPU/GPU synchronization.
- New activity, Kelvin, vapor-pressure, or latent-heat models (E4-F1/F2/F4).
- Performance diagnostics or debug-only public API (diagnostics: none).
- Claiming the complete E4 production envelope before E4-F6/E4-F7 gates pass.
