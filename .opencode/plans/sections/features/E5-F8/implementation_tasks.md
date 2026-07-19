# Implementation Tasks

## Example and Evidence Code

- [x] Add `docs/Examples/gpu_condensation_parity_walkthrough.py` with immutable
  physical fixture constants and separate NumPy/Warp builders.
- [x] Implement the independent NumPy fixed-four-substep oracle without reading
  mutated Warp buffers or calling production Warp kernels for expected values.
- [x] Build a compact fp64 multi-box/multi-species fixture containing uptake,
  evaporation, gas coupling, and nonzero latent heat.
- [x] Run `condensation_step_gpu` on Warp CPU with explicit scratch,
  thermodynamics, latent-heat, and energy sidecars and explicit synchronization.
- [x] Keep optional CUDA execution additive and skip cleanly when unavailable.
- [x] Test no-Warp/force-disabled control flow, fake enabled sidecars and sync,
  runtime failure propagation and fresh-source recovery guidance, Warp CPU, and
  optional CUDA.

## Documentation

- [x] Add `docs/Features/Roadmap/condensation-parity-walkthrough.md` as the P3
  bounded-evidence and deferred-capability ownership record.
- [x] Add a complete 14-row deferred-capability ownership table with owner,
  entry gate, and explicit E5-F8 non-claim for every carry-forward item.
- [x] Assign phase-aware surface tension and BAT activity to a separately
  approved condensation-physics expansion rather than implying current support.
- [x] Assign thermal feedback/adaptive stepping to a future approved
  condensation numerical-method plan; assign high-level backend/`Runnable`
  work and general workflow parity to Epic G, capture and performance work to
  Epic H, and broad autodiff to Epic I.
- [ ] Link the artifact from canonical condensation, foundations, examples, and
  both roadmap files without duplicating conflicting support language.

## Tooling / Tests

- [x] Add
  `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py` for import,
  deterministic fixtures, category separation, output, and device policy.
- [x] Add `particula/tests/condensation_parity_walkthrough_docs_test.py` for
  required ownership rows, evidence boundary, non-claims, commands, and link
  targets.
- [x] Add separate category-isolation and all-results reporting tests for P2,
  including isolated vapor-pressure, energy-sidecar, and conservation-input
  mutations plus a multi-failure case.
- [ ] Run the focused test suite with `-Werror`, optional CUDA selection when
  available, and the repository coverage validation.
