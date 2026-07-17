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
- **P2 -- majorant and private dispatch (complete):** Co-located direct-kernel
  tests cover the exhaustive compact active-pair majorant, the existing bounded
  scheduler/RNG path, cleared private settling-velocity scratch, and mixed
  sedimentation-mask no-op behavior. Public preflight rejection remains a
  separate preserved boundary.
- **P3 -- end-to-end execution:** Run sedimentation-only one-box and multi-box/
  multi-species cases with differing temperature and pressure. Check particle
  and species-mass conservation, donor mass/concentration clearing, inactive
  slots, accepted-count capacity, caller buffer identity, RNG reuse/reset, and
  aggregate or sigma-bounded collision behavior over deterministic seeds.
  Snapshot masses, concentration, charge, pair/count buffers, and RNG state to
  prove invalid and unsupported calls fail before mutation.
- **P4 -- documentation:** Validate markdown links, references, import paths,
  support-table wording, and executable snippets where present.

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
