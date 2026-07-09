# E3-F2 Implementation Tasks

## P1 Tasks

- [x] Create a mixed NPF/droplet `ParticleData` fixture in
  `particula/gpu/kernels/tests/coagulation_test.py` with explicit `np.float64`
  masses/radii spanning nanometer to droplet scales.
- [x] Add a test-local acceptance metric helper in
  `particula/gpu/kernels/tests/coagulation_test.py` that reads attempted-pair
  and accepted-pair counts from the existing call path without changing the
  public kernel API or adding hidden CPU transfers.
- [x] Cover the helper with focused tests for attempted/accepted count parity,
  finite non-negative acceptance fractions, and sparse-box zero-collision
  behavior.
- [x] Assert that the baseline metric is finite, positive, and stable enough for
  seeded stochastic checks on Warp CPU; reuse the same fixture under CUDA when
  available instead of creating a second mixed-scale dataset.

## P2 Tasks

- Prototype fixed-bin majorant or stratified pair-selection logic in
  `particula/gpu/kernels/coagulation.py` using `brownian_kernel_pair_wp(...)`.
- Keep the change inside the pair-selection portion of
  `coagulation_step_gpu(...)`; do not rewrite collision application, particle
  transfer helpers, or unrelated Warp launch structure in the same phase.
- Preserve collision-pair buffer shapes, `n_collisions` semantics, and
  `max_collisions` handling, with before/after assertions in
  `particula/gpu/kernels/tests/coagulation_test.py` for the mixed-scale fixture.
- Add co-located tests for invalid/self-pair rejection, sparse-bin fallback,
  and mass conservation, ideally as three focused tests rather than one large
  scenario to keep each phase near the 100-LOC review target.

## P3 Tasks

- Compare aggregate collision counts from repeated seeded runs in
  `particula/gpu/kernels/tests/coagulation_test.py` against CPU Brownian
  expected means using the repository's existing stochastic tolerance pattern.
- Verify conservation and repeated-step behavior with caller-owned
  `rng_states` so E3-F2 does not regress the E3-F1 persistence contract.
- Record in `docs/Features/Roadmap/data-oriented-gpu.md` whether the selected
  design materially improves mixed-scale acceptance or only documents a bounded
  limitation, including the exact pytest command used for the evidence.

## P4 Tasks

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the final design
  decision, measured acceptance bounds, and focused reproduction commands such
  as the exact `coagulation_test.py -k mixed_scale` invocation used during
  validation.
- If diagnostic helpers remain test-only, say so explicitly and document that
  production GPU helpers still require caller-owned explicit transfer steps.
- Confirm docs remain consistent with the landed test names and with any new
  mixed-scale fixture/helper names introduced in `coagulation_test.py`.
