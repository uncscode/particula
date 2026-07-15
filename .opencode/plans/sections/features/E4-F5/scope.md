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

## Remaining in scope
- Production-hook conservation regression, CPU reference evidence, and backend
  parity are P4 work.
- Development/user documentation is P5 work.

## Out of scope
- Exporting P2 internals or adding a public entry point.
- New gas/container fields, adaptive substepping, or CPU/GPU synchronization.
- New activity, Kelvin, vapor-pressure, or latent-heat models (E4-F1/F2/F4).
- Performance diagnostics or debug-only public API (diagnostics: none).
- Claiming the complete E4 production envelope before E4-F6/E4-F7 gates pass.
