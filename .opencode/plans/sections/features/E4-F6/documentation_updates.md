# Documentation Updates

- Update the issue 1272 condensation feature/roadmap document with a matrix for
  Warp CPU, optional CUDA, one/multi-box parity, conservation, capture, and
  bounded autodiff evidence.
- Update `docs/Features/Roadmap/warp-autodiff-limitations.md` with observed
  capture/tape results, clamp boundaries, in-place access constraints, and
  explicit non-claims.
- Update `docs/Features/Roadmap/condensation-stiffness-study.md` to distinguish
  final production qualification tolerances from the earlier `5e-2` candidate
  study bound.
- Update `.opencode/guides/testing_guide.md` only if new reusable graph/autodiff
  test conventions are established.
- Add focused pytest commands for Warp CPU and optional CUDA evidence.
- Update these plan sections with final phase status and measured results after
  implementation.

## Completed P1 Documentation Record

- Issue #1308 changed only the two GPU condensation test/support files; no
  user-facing documentation was added or revised.
- This plan records the delivered P1 matrix, its independent NumPy oracle,
  Warp-CPU requirement, optional-CUDA behavior, and focused coverage. P5 still
  owns user-facing evidence-matrix and roadmap updates.

## Completed P2 Documentation Record

- Issue #1309 changed only `particula/gpu/kernels/tests/`
  `_condensation_test_support.py` and `condensation_test.py`; no user-facing
  documentation was added or revised.
- This plan records the strict accounting, mutation-contract, and deterministic
  fresh-run regressions. P5 still owns user-facing evidence-matrix and roadmap
  updates.

## Completed P3 Documentation Record

- Issue #1310 added only
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py`; no
  user-facing documentation was added or revised.
- This plan records the graph-capture readiness evidence and its bounded
  non-claims. P5 still owns user-facing evidence-matrix and roadmap updates.

## Completed P4 Documentation Record

- Issue #1311 added only
  `particula/gpu/kernels/tests/condensation_autodiff_test.py`; no production
  code, published documentation, or public API was revised.
- This plan records the bounded out-of-place raw-rate Tape evidence,
  centered-fp64 reference, access-verification cleanup, optional CUDA behavior,
  and forward-only P2 clamp/inventory/in-place-mutation non-claims. P5 still
  owns user-facing evidence-matrix and roadmap updates.
