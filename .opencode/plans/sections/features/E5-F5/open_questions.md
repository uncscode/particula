# Open Questions

- [x] Should dissipation equal to zero be accepted as a disabled-rate box or
  rejected under the epic's positive-finite physical-input policy?
  - Resolved 2026-07-16: reject zero when ST1956 is enabled. Require positive
    finite dissipation and fluid density in every enabled box; omit ST1956 from
    the mechanism set to disable the term.
- [x] Should the initial majorant use the tight two-largest-active-radii proof
  or an exhaustive active-pair scan?
  - Resolved 2026-07-16: use the O(n) two-largest-distinct-active-radii maximum.
    The box prefactor is constant and the cubic radius-sum term is monotone for
    nonnegative radii. Tests exhaustively compare every active pair to the bound.
- [x] What should happen when turbulent inputs are supplied but the mechanism
  is not enabled?
  - Resolved 2026-07-16: reject either argument as excess input, reject partial
    input pairs, and fail before allocation, RNG initialization, or mutation.
- [x] Does E5-F5 include DNS turbulence models?
  - Resolved 2026-07-15: No. Only the ST1956 turbulent-shear equation is in
    scope; DNS models and claims are explicitly excluded.
- [x] Are dissipation and fluid density fields added to shared environment data?
  - Resolved 2026-07-15: No. They remain explicit call-specific scalar or
    per-box inputs, preserving shared data-container scope and ownership.
