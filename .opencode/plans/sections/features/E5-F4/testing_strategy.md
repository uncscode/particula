# Testing Strategy

Every implementation phase ships its tests in the same change. Coverage
thresholds are never lowered; changed code must maintain at least 80% coverage.
GPU tests use the repository's `*_test.py` convention and collect cleanly when
Warp is absent.

## Per-Phase Coverage

- **P1 -- physics helpers:** Add small Warp probe kernels in
  `particula/gpu/kernels/tests/coagulation_test.py`. Compare effective density,
  settling velocity, and scalar SP2016 rates against independently evaluated
  NumPy equations for single/multiple species and multiple size/density scales.
  Assert symmetry, units-consistent non-negative values, zero rate for equal
  settling velocity, and collision efficiency exactly 1.
- **P2 -- majorant and dispatch:** Enumerate all active unordered fixture pairs
  on the test side and prove each rate is finite, non-negative, and no greater
  than the device majorant. Cover zero/one/two active slots, equal velocities,
  inactive gaps, bounded trial scheduling, sorted/in-range/disjoint accepted
  pairs, one acceptance draw, and persistent RNG advancement.
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

Primary coverage remains in
`particula/gpu/kernels/tests/coagulation_test.py`, colocated with the existing
direct coagulation path. If helper volume makes that module unwieldy, a focused
`particula/gpu/kernels/tests/sedimentation_coagulation_test.py` may be added,
still using the `*_test.py` suffix and shared device fixtures. No slow or
performance marker is needed for required correctness evidence.
