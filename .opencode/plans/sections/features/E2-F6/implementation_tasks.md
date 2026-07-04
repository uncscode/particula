# E2-F6 Implementation Tasks

## Phase E2-F6-P1

- Inventory the current reference touchpoints in
  `particula/particles/particle_data.py`,
  `particula/particles/particle_data_builder.py`, and the existing GPU
  condensation tests before creating new study helpers.
- Add deterministic case builders in
  `particula/gpu/tests/mass_precision_case_helpers.py` only if the helper logic
  cannot stay under roughly 100 LOC inside
  `particula/gpu/tests/mass_precision_cases_test.py`.
- Define reproducible cases for NPF clusters, 5-10 nm particles,
  accumulation-mode particles, and droplet-scale particles with shapes matching
  `(n_boxes, n_particles, n_species)`.
- Add tests in `particula/gpu/tests/mass_precision_cases_test.py` proving the
  generated arrays are finite, nonnegative, and stable across reruns.

## Phase E2-F6-P2

- Implement study-only projection helpers in
  `particula/gpu/tests/mass_precision_metrics_test.py` or an adjacent helper,
  keeping production container defaults untouched.
- Compare `fp32`, mixed-precision, and any representation alternative against
  baseline `fp64` mass/radius reconstruction using explicit NumPy assertions.
- Inspect `ParticleData`, `ParticleDataBuilder`, `WarpParticleData`, and
  conversion helpers to confirm public defaults remain `fp64` after the study
  code lands.
- Record unsupported candidates directly in
  `docs/Features/Roadmap/mass-precision-study.md` rather than widening runtime
  APIs to accommodate them.

## Phase E2-F6-P3

- Add conservation/fidelity assertions in
  `particula/gpu/tests/mass_precision_metrics_test.py` against CPU reference
  calculations for representative multi-box cases.
- Measure small-particle mass and radius fidelity while large-droplet bins
  coexist in the same arrays so cancellation risk is visible.
- Calculate per-candidate memory footprints in the report using explicit
  `n_boxes`, `n_particles`, and `n_species` examples rather than prose-only
  estimates.
- Run skip-safe throughput checks next to existing GPU condensation tests when
  hardware is available, or record the missing-runtime limitation in the report.

## Phase E2-F6-P4

- Write `docs/Features/Roadmap/mass-precision-study.md` with a recommendation,
  evidence tables, and the exact pytest/benchmark commands used to regenerate
  results.
- Link the report from `docs/Features/Roadmap/data-oriented-gpu.md` so later
  dtype/schema work has a single canonical reference.
- State explicit follow-up constraints for future schema or dtype changes,
  including which candidates were rejected and why.
- Validate documentation links and rerun the lightweight study tests before
  closing the phase.
