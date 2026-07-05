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

## Newly Resolved by P1 (#1208)

- The baseline case set is now fixed in
  `particula/gpu/tests/mass_precision_cases_test.py` as `npf_cluster`,
  `five_to_ten_nm`, `accumulation_mode`, and `cloud_droplet`.
- The baseline documentation location is now confirmed as
  `docs/Features/Roadmap/mass-precision-study.md`, with cross-links from the
  roadmap index and the data-oriented GPU roadmap.
- The current baseline policy remains unchanged after implementation: absolute
  per-species `np.float64` on CPU and `wp.float64` on Warp mirrors.

## Remaining Questions

- Whether `fp32`, mixed precision, or alternate mass representations meet
  conservation and small-particle fidelity needs remains for P2-P4.
