Mark questions as resolved with dates when answers are found.

1. [x] Should `CondensationFactory` expose the new public strategy under only
   `"latent_heat"`, or should it also support a clearer alias such as
   `"non_isothermal"`? (reviewer: plan-review-architecture)
   - Resolved 2026-06-06: P2 shipped the final `"latent_heat"` key only.
     Factory registration preserved the generic builder path, and no alias was
     added in this phase.
