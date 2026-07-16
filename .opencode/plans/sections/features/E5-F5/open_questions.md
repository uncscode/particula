# Open Questions

- [ ] Should dissipation equal to zero be accepted as a disabled-rate box or
  rejected under the epic's positive-finite physical-input policy?
  - Proposed resolution: reject zero for an enabled turbulent-shear term. A
    caller that does not want the term should omit it from the mechanism set;
    this avoids ambiguous mixed-box capability semantics.
- [ ] Should the initial majorant use the tight two-largest-active-radii proof
  or an exhaustive active-pair scan?
  - Proposed resolution: use the O(n) tight bound because the per-box ST1956
    prefactor is constant and `(2r_i + 2r_j)^3` is monotone for non-negative
    radii; retain exhaustive all-pairs evaluation in tests. Fall back to the
    exhaustive implementation if Warp control-flow review makes the compact
    extrema path less reliable.
- [ ] What should happen when turbulent inputs are supplied but the mechanism
  is not enabled?
  - Open: follow the excess-input policy finalized by E5-F1. Prefer rejection
    over silent ignore so configuration mistakes fail explicitly.
- [x] Does E5-F5 include DNS turbulence models?
  - Resolved 2026-07-15: No. Only the ST1956 turbulent-shear equation is in
    scope; DNS models and claims are explicitly excluded.
- [x] Are dissipation and fluid density fields added to shared environment data?
  - Resolved 2026-07-15: No. They remain explicit call-specific scalar or
    per-box inputs, preserving shared data-container scope and ownership.
