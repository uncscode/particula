# Testing Strategy

Every implementation phase ships with its own tests. The configured coverage
threshold is never lowered; changed code must retain at least 80% coverage.
Tests use `*_test.py`, run on Warp CPU whenever Warp is installed, and
parameterize optional CUDA with a clean skip when unavailable.

## Per-Phase Approach

- **P1 — Pair-property unit tests:** Add probe kernels to
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`. Compare Warp outputs
  with independently evaluated CPU formulas for reduced quantities, Coulomb
  potential, kinetic/continuum limits, and diffusive properties. Include zero
  charge, equal particles, opposite/same signs, the `-200` repulsive clip,
  extreme-but-supported scales, and finite non-negative outputs.
- **P2 — Model parity tests:** For each approved model, construct radius, mass,
  charge, temperature, and pressure fixtures independently on CPU and Warp.
  Compare scalar pair rates with explicit per-fixture `rtol`/`atol`; include
  symmetry and neutral-limit checks. Do not import the Warp implementation into
  the expected-value calculation.
- **P3 — Validation/regression tests:** Extend
  `particula/gpu/kernels/tests/coagulation_test.py` with wrong shape, wrong dtype,
  wrong device, NaN, and infinity cases. Snapshot masses, concentration, charge,
  collision buffers, counts, and persistent RNG to prove validation fails before
  mutation or RNG advancement. Retain existing Brownian API tests.
- **P4 — Merge/conservation tests:** Launch `apply_coagulation_kernel` with
  deterministic disjoint pair buffers. Assert recipient mass by species and
  charge sums, donor fields are all zero, untouched/inactive slots are stable,
  and no-collision input is a no-op. Add one- and multi-box step cases and assert
  species mass and total charge separately per box with tight fp64 tolerances.
- **P5 — Documentation validation:** Check links, imports, support wording, and
  focused reproduction commands. Documentation must say pair physics and merges
  are foundational, not claim E5-F3 execution before it ships.

## Coverage and Evidence Boundaries

Deterministic pair parity and direct merge tests are release-blocking. This
feature does not require stochastic charged collision counts or exact CPU/Warp
pair replay; those belong to E5-F3/F7. Existing Brownian, persistent-RNG,
preallocated-buffer, sparse/inactive, and mixed-scale tests remain green.
