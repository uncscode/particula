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
