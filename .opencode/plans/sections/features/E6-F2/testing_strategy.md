# Testing Strategy

Every implementation phase ships its tests in
`particula/gpu/kernels/tests/dilution_test.py`; coverage thresholds remain at
least the configured 80% and are never lowered.

## Per-Phase Approach

- **P1 contract tests:** Assert public signature/import, scalar and same-device
  `(n_boxes,)` coefficient forms, units, return identity, and deterministic
  validation ordering. Reject host arrays and ambiguous shapes.
- **P2 kernel tests:** Compare one-box and multi-box/multi-species outputs with
  an independent NumPy T1 oracle. Cover zeros, inactive particle slots,
  nonuniform coefficients, repeated steps, exact no-ops, and unchanged masses,
  charge, density, volume, gas metadata, vapor pressure, and partitioning.
- **P3 validation tests:** Parameterize negative/nonfinite coefficient or time,
  wrong rank/shape/dtype/device, inconsistent box dimensions, and invalid
  concentration state. Snapshot every caller field and assert no launch,
  allocation where contractually avoidable, or mutation occurs on failure.
- **P4 parity tests:** Require Warp CPU for scalar/per-box, single/multi-box,
  particle/gas, and edge-case fixtures. Record explicit `rtol`/`atol` for
  float64 CPU parity; run the same deterministic matrix on CUDA when available
  and skip cleanly otherwise.
- **P5 docs validation:** Verify links, import snippets, support/deferred tables,
  and focused commands.

## Regression and Coverage

- Keep existing `particula/dynamics/tests/dilution_test.py` and E6-F1 tests green.
- Add an import smoke test for `from particula.gpu.kernels import dilution_step_gpu`.
- Use `pytest particula/gpu/kernels/tests/dilution_test.py -q -Werror` as the
  focused command; include relevant conversion/container tests when boundaries
  change.
- No exact CPU/GPU bitwise claim is required unless the chosen T1 operation is
  proven identical; acceptance uses documented deterministic float64 tolerances.
