Mark questions as resolved with dates when answers are found.

1. [ ] Should `CondensationFactory` expose the new public strategy under only
   `"latent_heat"`, or should it also support a clearer alias such as
   `"non_isothermal"`? (reviewer: plan-review-architecture)
   - Open: Factory key naming affects user-facing API docs, examples, and
     backward-compatible configuration patterns.
