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
- **P3 validation tests (shipped, #1397):** Parameterize invalid coefficient or
  time types/domains, host/Warp rank/shape/dtype/device errors, inconsistent
  dimensions, and negative/nonfinite coefficient or concentration values.
  Full-state snapshots and allocation/launch spies prove rejected calls and
  valid scalar-zero/zero-time no-ops allocate and launch nothing; conflict cases
  lock the documented validation precedence.
- **P4 parity tests (shipped, #1398):** A test-local independent NumPy oracle
  checks particle and gas concentrations separately at `rtol=1e-12`, `atol=0`.
  Required Warp CPU covers scalar/per-box, one/multi-box, multi-species gas,
  zero/inactive cells, repeated nonuniform calls, identities, protected state,
  caller-owned coefficient preservation, and exact scalar-zero/zero-time
  no-ops. The identical finite-step matrix runs on CUDA when available and
  otherwise skips cleanly.
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
