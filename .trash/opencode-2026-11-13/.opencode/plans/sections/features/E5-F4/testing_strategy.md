# Testing Strategy

Every implementation phase ships its tests in the same change. Coverage
thresholds are never lowered; changed code must maintain at least 80% coverage.
GPU tests use the repository's `*_test.py` convention and collect cleanly when
Warp is absent.

## Per-Phase Coverage

- **P1 -- physics helpers (complete):** Direct Warp probe kernels and
  independent NumPy safe-zero oracles are in
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`. They cover
  one-/multi-species effective density, nanometer-to-droplet Stokes/Cunningham
  settling, SP2016 parity and symmetry, equal-velocity zero, and batched
  invalid/overflow/underflow exact-zero cases. Finite nonzero fp64 parity uses
  `rtol=1e-12, atol=0.0`; safe-zero branches use exact equality. An AST check
  verifies that the pair helper has exactly its four physics arguments and no
  collision-efficiency argument.
- **P2 -- majorant and dispatch (complete):** Co-located direct-kernel
  tests cover the exhaustive compact active-pair majorant, the existing bounded
  scheduler/RNG path, cleared private settling-velocity scratch, and mixed
  sedimentation-mask rejection behavior.
- **P3 -- end-to-end execution (complete):** Sedimentation-only one-box and multi-box/
  multi-species cases with differing temperature and pressure. Check particle
  and species-mass conservation, donor mass/concentration clearing, inactive
  slots, accepted-count capacity, caller buffer identity, RNG reuse/reset, and
  aggregate or sigma-bounded collision behavior over deterministic seeds.
  Snapshot masses, concentration, charge, pair/count buffers, and RNG state to
  prove invalid and unsupported calls fail before mutation. Physical-domain
  regressions cover nonfinite/negative mass and concentration and nonfinite/
  nonpositive density, including accepted zero mass/concentration boundaries.
- **P4 -- documentation (complete):** The canonical direct-kernel invocation
  and its support limits are recorded in the user-facing GPU contract guide.
- **Issue #1350 documentation follow-up (complete):** The focused evidence
  description now explicitly covers singleton sedimentation configuration,
  direct and environment inputs, caller-owned output/RNG behavior, conservation,
  and rejected-call state safety. No executable behavior or test is added.

## Device and Numerical Policy

- Warp CPU is required when Warp is installed and is the release baseline.
- CUDA is parametrized when available and skips cleanly otherwise.
- Deterministic fp64 fixtures declare explicit `rtol`/`atol` based on scale;
  conservation remains tight and is asserted per box and species.
- Stochastic validation uses repeated-run means, sigma bounds, invariants, and
  legal-pair checks, never exact CPU/Warp pair replay.
- Expected pair/property values are computed independently from public CPU
  formulas or direct NumPy equations; tests do not share the new Warp helper.

## Coverage Impact

P1 helper coverage is colocated with the helpers in
`particula/gpu/dynamics/tests/coagulation_funcs_test.py`; later execution
coverage remains in the direct-kernel test module. No slow or performance
marker is needed for required correctness evidence.

Issue #1350 retains the focused warning-clean commands
`particula/gpu/dynamics/tests/coagulation_funcs_test.py` and
`particula/gpu/kernels/tests/coagulation_test.py`, plus the documented
`-m "warp and gpu_parity"` smoke selection. Warp CPU is the installed-Warp
baseline; CUDA remains optional cleanly-skipped additive evidence.
