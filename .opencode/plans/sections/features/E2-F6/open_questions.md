# E2-F6 Open Questions

## Resolved Answers

- Use `fp64` CPU parity as the baseline. Initial acceptance should target finite,
  non-negative masses/radii and relative error within documented tolerances for
  representative NPF, accumulation, and droplet regimes; exact numeric thresholds
  should be recorded by E2-F6 after measuring current behavior.
- Place the final report under `docs/Features/Roadmap/`, where it can support
  downstream GPU roadmap decisions without presenting exploratory precision work
  as a stable user feature.
- The most relevant mixed-precision candidate is `fp32` storage with `fp64`
  accumulation. Treat other splits as secondary unless measurements show this
  candidate is unsuitable.
- Represent gas-mass conservation with explicit gas concentration deltas and
  total particle-plus-gas mass checks. Do not depend on environment containers to
  carry gas mass.
- Treat log-mass or reference-mass scaling as a quantitative candidate only if
  baseline `fp64` evidence shows absolute mass representation is insufficient;
  otherwise document it as future work.
- Keep the executed P2 comparison layer study-only: candidate helpers and
  fidelity checks live in focused GPU tests and docs, while production dtype
  defaults remain unchanged.

## Newly Resolved by P1 (#1208)

- The baseline case set is now fixed in
  `particula/gpu/tests/mass_precision_cases_test.py` as `npf_cluster`,
  `five_to_ten_nm`, `accumulation_mode`, and `cloud_droplet`.
- The baseline documentation location is now confirmed as
  `docs/Features/Roadmap/mass-precision-study.md`, with cross-links from the
  roadmap index and the data-oriented GPU roadmap.
- The current baseline policy remains unchanged after implementation: absolute
  per-species `np.float64` on CPU and `wp.float64` on Warp mirrors.

## Newly Resolved by P2 (#1209)

- The shipped executable candidate set is now fixed in
  `particula/gpu/tests/mass_precision_metrics_test.py` as
  `fp32_absolute_mass`, `mixed_precision_mass_plus_density`, and
  `fp32_total_mass_fp32_mass_fraction`.
- Invalid candidate ids now fail with a tested `ValueError`, and the
  total-mass-plus-fraction path reconstructs zero-total-mass particles to
  zeros without warning-driven behavior.
- Candidates that require runtime schema expansion remain unsupported and are
  documented without widening production APIs.
- The P2 implementation confirmed again that production defaults remain
  absolute per-species `np.float64` on CPU and `wp.float64` on Warp mirrors.

## Newly Resolved by P3 (#1210)

- The shipped P3 executable surface is now split cleanly between fast metrics
  in `particula/gpu/tests/mass_precision_metrics_test.py`, optional throughput
  timing in `particula/gpu/tests/benchmark_test.py`, and fast helper coverage
  in `particula/gpu/tests/benchmark_helpers_test.py`.
- Conservation-sensitive candidate review is now grounded in executable
  CPU-reference `get_mass_transfer(...)` comparisons with both per-particle and
  aggregate species-total delta assertions.
- Mixed-scale smallest-particle thresholds, zero-total-mass handling,
  zero-volume warning-clean paths, and clamp accounting terminology are now all
  fixed in the shipped test and roadmap surfaces.
- The roadmap page now records explicit analytic memory-footprint examples and
  documents the benchmark path as an opt-in, skip-safe throughput surface.

## Remaining Questions

- Whether the final recommendation should keep absolute `fp64` storage or
  endorse a later migration path remains open pending those broader tradeoff
  results.
