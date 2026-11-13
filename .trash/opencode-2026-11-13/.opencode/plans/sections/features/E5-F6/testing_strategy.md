# Testing Strategy

Every code phase ships co-located tests in the same change. Coverage thresholds
are never lowered; changed code maintains at least 80% coverage. Test files use
the repository's `*_test.py` convention and collect cleanly without Warp.

## Per-Phase Coverage

- **P1 — Combination contract (implemented in #1357):** Matrix tests cover the
  four singleton, six pair, and four-term recognized masks, reverse pair order,
  and all four rejected three-term masks. Validation and atomicity tests cover
  mask-driven turbulent/charged/sedimentation preflight, non-turbulent ignoring
  turbulent arguments, the stable deferred error, downstream-helper bypass, and
  unchanged caller particle/output/RNG state for valid deferred calls.
- **P2 — Additive math and bound (implemented in #1358):** Independent
  deterministic fp64 NumPy oracles calculate Brownian, charged, sedimentation,
  and ST1956 component rates without calling production aggregate helpers.
  Private Warp probes cover all recognized two-way/four-way masks across sparse
  mixed-scale fixtures with non-coincident component maxima, asserting finite
  non-negative totals, explicit-tolerance parity, and checked summed-majorant
  bounds. Regressions cover invalid/overflowed aggregates, zero totals, capped
  scheduling, eight-ULP permitted overshoot versus ninth-ULP rejection, and no
  acceptance draw or selector mutation on rejected ratios. These tests carry
  `warp` and `gpu_parity` markers; Warp CPU is the required backend and CUDA is
  optional.
- **P3 — Single-pass integration (implemented):** Existing public-path tests
  exercise executable masks `1`, `2`, `4`, `8`, `3`, `5`, `6`, `9`, `10`, `12`,
  and `15`; test-local diagnostics cover shared selection and apply behavior,
  caller-buffer identity, persistent RNG, conservation, inactive slots, and
  deferred-mask atomicity for `7`, `11`, `13`, and `14`.
- **P4 — Documentation (implemented):** Final validation is source/signature and
  Markdown inspection of links, direct imports, SI units, exact matrix rows,
  and the absence of changed executable snippets, plus ruff and the existing
  focused coagulation suite. These checks do not constitute E5-F7 release or
  cross-mechanism closeout.

## Deterministic and Stochastic Oracles

- Deterministic expected values come from public CPU formulas or direct NumPy
  equations evaluated independently per component. Tests must not call the new
  Warp aggregate helper to generate expected totals.
- Pair-rate comparisons use explicit `np.float64` fixtures and declared
  scale-appropriate `rtol`/`atol`. Majorant checks include a separately reported
  numerical margin; conservation tolerances remain tight per box/species.
- Stochastic behavior uses repeated fresh seeded runs with expected aggregate
  rates and declared sigma/confidence bounds. Exact CPU/Warp pair replay is not
  required. Deterministic invariants are asserted on every run.
- Preserve legacy omitted/explicit Brownian equivalence and E5-F3's
  Brownian-plus-charged regression behavior.

## Device and Coverage Policy

- Primary tests live in `particula/gpu/kernels/tests/coagulation_test.py`; no
  additional additive test module is needed for this documentation-only phase.
- Warp CPU is required when Warp is installed. CUDA reuses parameterized
  fixtures and skips cleanly when unavailable; it is additive evidence only.
- Focused correctness tests require no slow/performance marker. Run the full
  coagulation suite after focused tests to detect ownership and API regressions.
