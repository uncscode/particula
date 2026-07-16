# Open Questions

The following decisions are resolved without broadening issue #1320 scope:

- [x] Which CPU charged variants are approved for the first GPU support matrix?
  - Resolved 2026-07-16: only the complete hard-sphere system-state model is
    approved, with identifier `charged_hard_sphere`. Dyachkov 2007, Gatti 2008,
    Gopalakrishnan 2012, and Chahl 2019 remain explicitly unsupported.
- [x] What configuration representation preserves Brownian call compatibility
  while supporting scalar and per-box mechanism inputs?
  - Resolved 2026-07-16: use a frozen concrete-module
    `CoagulationMechanismConfig` and keyword-only `mechanism_config=None`, where
    `None` preserves Brownian behavior. Normalize unique mechanism sets to
    canonical order. Turbulent inputs remain explicit scalar or active-device
    `(n_boxes,)` call arguments and validate before allocation, RNG work, or
    mutation.
- [x] What analytic or conservative construction proves the charged and total
  combined majorants across supported mixed-scale inputs?
  - Resolved 2026-07-16: use exhaustive unique-active-pair maxima for charged
    hard-sphere and SP2016 terms, and the proved two-largest-active-radii
    maximum for ST1956. Sum valid component majorants for additive execution.
    If that bound reaches the trial cap in an approved fixture, use the
    exhaustive maximum of the summed pair rate rather than an unproved bound.
- [x] What deterministic tolerances and stochastic confidence bounds apply to
  each mechanism and combined fixture?
  - Resolved 2026-07-16: Brownian uses `rtol=1e-7, atol=0`; positive charged,
    SP2016, ST1956, and additive rates use `rtol=1e-6, atol=0`; the extreme
    repulsive charged fixture uses `atol=1e-30`; exact-zero rates are asserted
    exactly. Stochastic fixtures use 100 fresh seeds and a predeclared
    `3 * sqrt(expected_mean)` bound. Mass and charge conservation remain
    separate checks at `rtol=1e-12` with explicit physical zero floors.
- [x] Which downstream epic or track owns each deferred condensation capability
  identified by the E4 walkthrough?
  - Resolved 2026-07-16: E5 does not assign these out-of-scope capabilities to
    an existing epic. Thermal feedback and adaptive stepping require a future
    condensation numerical-method plan. Phase-aware surface tension and BAT
    activity require a future condensation-physics expansion plan. Plan IDs
    are assigned when those future plans are approved, before implementation.
    Existing roadmap ownership remains Epic G for high-level backend/runnable
    integration, Epic H for graph/performance work, and Epic I for broad
    autodiff.

Classifier diagnostics: none.
