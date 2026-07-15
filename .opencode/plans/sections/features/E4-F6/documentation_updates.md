# Documentation Updates

- Issue #1312 shipped the P1--P4 evidence matrix and focused commands in
  `docs/Features/condensation_strategy_system.md`.
- `docs/Features/Roadmap/warp-autodiff-limitations.md` records P4's one-box
  raw-rate interior Tape evidence, capability skips, state-restoration behavior,
  and P2 forward-only boundaries.
- `docs/Features/Roadmap/condensation-stiffness-study.md` distinguishes the
  historical `5e-2` candidate comparison from P1 parity/P2 strict conservation
  and records P3 as unsupported capture rather than replay support.
- `.opencode/guides/testing_guide.md` lists all discoverable wrappers and the
  separate parity, conservation, capture-limit, and derivative conventions.
- The documented commands use Warp CPU for supported probes and optional CUDA
  selections; P3 CPU skip/CUDA strict-xfail remain limitation evidence.

## Completed P1 Documentation Record

- Issue #1308 changed only the two GPU condensation test/support files; no
  user-facing documentation was added or revised.
- This plan records the delivered P1 matrix, its independent NumPy oracle,
  Warp-CPU requirement, optional-CUDA behavior, and focused coverage.

## Completed P2 Documentation Record

- Issue #1309 changed only `particula/gpu/kernels/tests/`
  `_condensation_test_support.py` and `condensation_test.py`; no user-facing
  documentation was added or revised.
- This plan records the strict accounting, mutation-contract, and deterministic
  fresh-run regressions.

## Completed P3 Documentation Record

- Issue #1310 added only
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py`; no
  user-facing documentation was added or revised.
- This plan records the graph-capture limitation: CPU is capability-skipped and
  CUDA public-step replay is strict-xfailed because host validation readbacks
  are not capture-safe.

## Completed P4 Documentation Record

- Issue #1311 added only
  `particula/gpu/kernels/tests/condensation_autodiff_test.py`; no production
  code, published documentation, or public API was revised.
- This plan records the bounded out-of-place raw-rate Tape evidence,
  centered-fp64 reference, access-verification cleanup, optional CUDA behavior,
  and forward-only P2 clamp/inventory/in-place-mutation non-claims.

## Completed P5 Documentation Record

- Issue #1312 updated the feature, roadmap, and testing-guide records named
  above. The matrix maps each P1--P4 result to its discoverable wrapper and
  explicitly separates parity, strict conservation, unsupported P3 capture,
  and bounded P4 raw-rate autodiff evidence.
