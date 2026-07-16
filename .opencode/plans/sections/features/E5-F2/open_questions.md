# Open Questions

- [ ] Which CPU charged variants are approved for the first GPU support matrix?
  - Owner: E5-F1/E5-F2-P2.
  - Decision gate: freeze identifiers and require the hard-sphere charged
    baseline before P2 implementation; explicitly defer or include Dyachkov
    2007, Gatti 2008, Gopalakrishnan 2012, and Chahl 2019 individually.
- [ ] What exact `rtol`/`atol` applies to each approved pair model and extreme
  repulsive fixture?
  - Owner: E5-F2-P1/P2, with E5-F7 consuming the recorded fixture tolerances.
  - Constraint: tolerances must be justified per formula and cannot be replaced
    by exact stochastic pair replay.
- [ ] Should invalid duplicate recipient indices in a caller-supplied collision
  buffer be defensively ignored, rejected, or remain a documented private-kernel
  precondition?
  - Owner: E5-F2-P4.
  - Constraint: the production selector emits disjoint pairs; do not add an
    O(n²) validation or serialize the normal path without evidence.
- [ ] Should charge finite-value validation reuse a shared active-device helper
  or add a coagulation-local helper with non-positive values allowed?
  - Owner: E5-F2-P3.
  - Constraint: validation must not copy charge to the host and must finish
    before RNG initialization or particle mutation.
- [ ] Does the final model decision require concrete-module exports for pair
  helpers, or should they remain internal until E5-F3 integrates execution?
  - Owner: E5-F2-P2/E5-F3. Default is internal helpers to avoid premature API.

Classifier diagnostics: none.
