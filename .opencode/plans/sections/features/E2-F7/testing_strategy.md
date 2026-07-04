# E2-F7 Testing Strategy

## Co-located Testing Policy

Each phase that adds helper functions, prototypes, or production hooks must add
tests in the same phase. There is no standalone testing phase.

## Fast Tests

- Add or extend `particula/gpu/kernels/tests/condensation_test.py` for:
  - stress-case fixture construction,
  - fixed-shape and dtype validation,
  - preallocated `mass_transfer` buffer reuse,
  - explicit timestep stability classification,
  - candidate integration invariants.
- Keep tests compatible with Warp CPU execution by default.
- Use CUDA-specific execution only behind existing availability guards.

## Target Test and Evidence Files

- `particula/gpu/kernels/tests/condensation_test.py` for all fast P1-P3
  executable coverage.
- `particula/gpu/kernels/tests/condensation_stiffness_helpers.py` only if the
  shared stress-case builders or metric helpers outgrow the main test file.
- `docs/Features/Roadmap/condensation-stiffness-study.md` for the published
  timestep tables, reproduction notes, and final recommendation evidence.
- `docs/Features/Roadmap/warp-autodiff-limitations.md` only for cross-links and
  documented constraints in P4; do not move executable assertions into docs.

## Reference Tests

- Reuse CPU reference functions from `mass_transfer.py` and
  `mass_transfer_utils.py` to validate rate and limiter expectations.
- Reuse patterns from staggered stability and mass-conservation tests, but avoid
  requiring slow benchmarks for default CI.

## Metrics to Assert

- Particle masses remain finite and non-negative.
- Explicit fractional mass changes stay below documented thresholds for a
  timestep classified as stable.
- Buffer shapes remain fixed and preallocated buffers are reused.
- Candidate integrators are deterministic for repeated runs.
- Documentation accurately marks GPU particle-only behavior versus full
  gas-particle conservation.

Map those assertions to phases: P1 covers fixture/metric correctness, P2 covers
current explicit-step stability bounds, P3 covers candidate sub-step or
semi-implicit invariants, and P4 is the documentation-only exception that must
rerun the same focused fast tests before publishing conclusions.

## Slow/Benchmark Coverage

- Extended timestep sweeps may be marked slow or kept as documentation
  generation scripts if they are too expensive for default CI.
- Slow tests should report enough context to reproduce the stiffness map but
  should not gate normal development unless runtime remains small.

If a sweep is too expensive for default CI, keep at least one representative
fast assertion in `condensation_test.py` that proves the recommendation is tied
to executable behavior rather than report-only analysis.

## Verification Commands

```bash
pytest particula/gpu/kernels/tests/condensation_test.py
pytest particula/dynamics/condensation/tests/staggered_stability_test.py -m "not slow"
ruff check particula/ --fix && ruff format particula/ && ruff check particula/
```
