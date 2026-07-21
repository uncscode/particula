# Open Questions

1. **Finite-step rule:** Should the canonical step use the exact first-order
   solution `c_new = c * exp(-alpha * dt)` or the repository's explicit rate
   update with a documented nonnegative limiter? P1 must resolve this before
   container code; the choice becomes E6-F2's CPU parity oracle.
2. **Strategy construction:** Should the strategy accept a precomputed
   coefficient, or chamber volume plus inlet flow and derive the coefficient?
   Prefer one unambiguous public constructor; avoid redundant mutable sources.
3. **Rate return shape:** Should `rate(aerosol)` return a named pair/tuple of
   particle and gas rates or a small immutable result type? Match existing
   process conventions while keeping both concentration domains observable.
4. **Container mutation API:** Is direct volume-aware particle concentration
   assignment the current sanctioned path, or should this feature add/reuse a
   public concentration setter on `ParticleRepresentation`? Do not use the
   deprecated distribution-merging `add_concentration()` path accidentally.
5. **Builder/factory surface:** The issue requires a strategy and runnable, not
   builders/factories. Add those only if maintainers identify a concrete
   consistency requirement before P4.
6. **Free-function validation compatibility:** Which currently accepted NumPy
   broadcasting cases are contractual? Preserve them unless physical-domain
   validation requires an explicitly documented correction.

All questions are non-blocking for plan drafting but questions 1–4 must be
closed in E6-F1-P1 before mutation APIs are implemented.
