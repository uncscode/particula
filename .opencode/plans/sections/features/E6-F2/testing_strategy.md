# Testing Strategy

Every implementation phase ships its tests in
`particula/gpu/kernels/tests/dilution_test.py`; coverage thresholds remain at
least the configured 80% and are never lowered.

## Per-Phase Approach

- **P1 contract tests (shipped, #1395):** Assert concrete-module signature and
  no-package-export boundary, scalar and same-device `(n_boxes,)` coefficient
  forms, scalar broadcast/per-box identity, units, zero-box and no-write
  identity returns, and deterministic validation ordering. Reject host arrays,
  ambiguous shapes, invalid dtype/device metadata, and invalid scalar domains.
- **P2 kernel tests (shipped, #1396):** Exercise one-box and
   multi-box/multi-species in-place decay with an independent NumPy E6-F1
   oracle; cover zeros, inactive particle slots, nonuniform coefficients,
   repeated steps, zero-time/scalar-zero and per-box-zero paths, zero extents,
   package export, and protected-field/container identity invariants.
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
- P1 tests assert direct concrete-module import; P2 adds the package import
  smoke test because it exports only `dilution_step_gpu` from
  `particula.gpu.kernels`.
- Use `pytest particula/gpu/kernels/tests/dilution_test.py -q -Werror` as the
  focused command; include relevant conversion/container tests when boundaries
  change.
- No exact CPU/GPU bitwise claim is required unless the chosen T1 operation is
  proven identical; acceptance uses documented deterministic float64 tolerances.
