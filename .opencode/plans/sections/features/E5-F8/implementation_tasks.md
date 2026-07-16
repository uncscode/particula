# Implementation Tasks

## Example and Evidence Code

- [ ] Add `docs/Examples/gpu_condensation_parity_walkthrough.py` with immutable
  physical fixture constants and separate NumPy/Warp builders.
- [ ] Implement the independent NumPy fixed-four-substep oracle without reading
  mutated Warp buffers or calling production Warp kernels for expected values.
- [ ] Build a compact fp64 multi-box/multi-species fixture containing uptake,
  evaporation, gas coupling, and nonzero latent heat.
- [ ] Run `condensation_step_gpu` on Warp CPU with explicit scratch,
  thermodynamics, latent-heat, and energy sidecars and explicit synchronization.
- [ ] Emit machine-testable, human-readable physics, conservation, and energy
  result blocks and return failure when any required category fails.
- [ ] Keep optional CUDA execution additive and skip cleanly when unavailable.

## Documentation

- [ ] Add `docs/Features/Roadmap/condensation-parity-walkthrough.md` explaining
  the independent setup, equations, thresholds, commands, and bounded claim.
- [ ] Add a complete deferred-capability ownership table with owner, entry gate,
  and explicit E5-F8 non-claim for every carry-forward item.
- [ ] Assign phase-aware surface tension and BAT activity to a separately
  approved condensation-physics expansion rather than implying current support.
- [ ] Assign thermal feedback/adaptive process work to Epic F, high-level
  backend/`Runnable` work and general workflow parity to Epic G, capture and
  performance work to Epic H, and broad autodiff to Epic I.
- [ ] Link the artifact from canonical condensation, foundations, examples, and
  both roadmap files without duplicating conflicting support language.

## Tooling / Tests

- [ ] Add
  `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py` for import,
  deterministic fixtures, category separation, output, and device policy.
- [ ] Add `particula/tests/condensation_parity_walkthrough_docs_test.py` for
  required ownership rows, thresholds, non-claims, commands, and link targets.
- [ ] Include mutation tests that independently perturb each expected category
  and verify only the corresponding result fails.
- [ ] Run focused tests with `-Werror`; run optional CUDA selection locally when
  available; do not lower configured coverage.
