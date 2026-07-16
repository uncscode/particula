# Open Questions

The following decisions must be resolved within the owning child track without
broadening issue #1320 scope:

- [ ] Which CPU charged variants are approved for the first GPU support matrix?
  - Owner: E5-F1/E5-F2. Record excluded variants explicitly before pair-port
    acceptance criteria are finalized.
- [ ] What configuration representation preserves Brownian call compatibility
  while supporting scalar and per-box mechanism inputs?
  - Owner: E5-F1. Resolution must preserve fail-before-mutation validation.
- [ ] What analytic or conservative construction proves the charged and total
  combined majorants across supported mixed-scale inputs?
  - Owner: E5-F3/E5-F6. Resolve before stochastic execution is enabled.
- [ ] What deterministic tolerances and stochastic confidence bounds apply to
  each mechanism and combined fixture?
  - Owner: E5-F7. Record per fixture; do not use exact collision-pair replay.
- [ ] Which downstream epic or track owns each deferred condensation capability
  identified by the E4 walkthrough?
  - Owner: E5-F8/E5-F9. Every row requires an owner before closeout.

Classifier diagnostics: none.
