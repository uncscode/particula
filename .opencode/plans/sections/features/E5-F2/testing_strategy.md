# Testing Strategy

Every implementation phase ships with its own tests. The configured coverage
threshold is never lowered; changed code must retain at least 80% coverage.
Tests use `*_test.py`, run on Warp CPU whenever Warp is installed, and
parameterize optional CUDA with a clean skip when unavailable.

## Per-Phase Approach

- **P1 — Pair-property unit tests (completed, #1336):** Co-located probe
  kernels in `particula/gpu/dynamics/tests/coagulation_funcs_test.py` compare
  Warp outputs to independent local oracles for reduced values, Coulomb
  potential, kinetic/continuum limits, and diffusive Knudsen number. Coverage
  includes neutral charge, equal particles, opposite/same signs, the `-200`
  repulsive clip, mixed scales, zero/negative guarded inputs, and the
  sub-`1e-80` kinetic-limit branch. Tests are marked `gpu_parity`, retain the
  module Warp marker, and use shared Warp-CPU/optional-CUDA device discovery.
- **P2 — Model parity tests (completed, #1337):**
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py` uses an independent
  NumPy hard-sphere oracle and fp64 Warp probe to cover valid neutral,
  same-sign, opposite-sign, mixed-scale, temperature, and pressure lanes.
  Deterministic Warp-CPU/optional-CUDA evidence checks oracle parity,
  pair-order symmetry, neutral behavior, the exact extreme-repulsion zero, and
  exhaustive non-finite/zero/negative safe-zero inputs for pair state, charge,
  and all eight scalar constants. Valid positive rates use `rtol=1e-6,
  atol=0`; invalid and extreme-repulsion results are exact zero.
- **P3 — Validation/regression tests:** Extend
  `particula/gpu/kernels/tests/coagulation_test.py` with wrong shape, wrong dtype,
  wrong device, NaN, and infinity cases. Snapshot masses, concentration, charge,
  collision buffers, counts, and persistent RNG to prove validation fails before
  mutation or RNG advancement. Retain existing Brownian API tests.
- **P4 — Merge/conservation tests (completed, #1339):**
  `particula/gpu/kernels/tests/coagulation_test.py` launches
  `apply_coagulation_kernel` with deterministic disjoint signed-charge pairs.
  A multi-box/multi-species test compares complete mass, concentration, and
  charge results to an independent NumPy expected state, then separately checks
  per-box species-mass and total-charge inventory with `rtol=1e-12,
  atol=1e-30`. Merge, zero-count, self-pair, and empty-pair cases cover donor
  clearing and no-op paths. Step-level zero-charge and signed-charge tests retain
  return and supplied-sidecar identity assertions; signed charge is conserved
  per box rather than incorrectly required to remain slotwise unchanged.
- **P5 — Documentation validation:** Check links, imports, support wording, and
  focused reproduction commands. Documentation must say pair physics and merges
  are foundational, not claim E5-F3 execution before it ships.

## Coverage and Evidence Boundaries

Deterministic pair parity and direct merge tests are release-blocking. This
feature does not require stochastic charged collision counts or exact CPU/Warp
pair replay; those belong to E5-F3/F7. Existing Brownian, persistent-RNG,
preallocated-buffer, sparse/inactive, and mixed-scale tests remain green.
