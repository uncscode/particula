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
