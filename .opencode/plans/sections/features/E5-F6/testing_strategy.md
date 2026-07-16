# Testing Strategy

Every code phase ships co-located tests in the same change. Coverage thresholds
are never lowered; changed code maintains at least 80% coverage. Test files use
the repository's `*_test.py` convention and collect cleanly without Warp.

## Per-Phase Coverage

- **P1 — Combination contract:** Table-drive all canonical single, approved
  two-way, and full four-way masks plus unsupported rows. Assert order
  independence, duplicate/unknown rejection, required mechanism inputs, and
  preflight snapshots of masses, concentration, charge, output buffers, and RNG.
- **P2 — Additive math and bound:** Independently calculate each component pair
  matrix, its valid majorant, and component sums. Across every active unordered
  pair assert finite non-negative values, total-rate parity, and
  `total_rate <= sum(term_majorants)`. Cover zero terms, different term maxima,
  mixed scales, zero-total early return, defensive roundoff, and bounded trials.
- **P3 — Single-pass integration:** Exercise approved two-way rows and the full
  four-way row on one-box and heterogeneous multi-box fixtures. Use test-local
  diagnostics to prove one proposal/acceptance stream and one apply pass. Check
  sorted, in-range, disjoint pairs; capacity; caller-buffer identity; persistent
  RNG reuse/reset; inactive slots; per-box/per-species mass conservation; and
  separate total-charge conservation.
- **P4 — Documentation:** Validate Markdown links, direct import paths,
  signature/configuration names, SI units, support-table rows, and executable
  snippets where present.

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

- Primary tests live in
  `particula/gpu/kernels/tests/coagulation_test.py`; a focused
  `particula/gpu/kernels/tests/additive_coagulation_test.py` may be introduced
  if matrix clarity warrants it.
- Warp CPU is required when Warp is installed. CUDA reuses parameterized
  fixtures and skips cleanly when unavailable; it is additive evidence only.
- Focused correctness tests require no slow/performance marker. Run the full
  coagulation suite after focused tests to detect ownership and API regressions.
