# Open Questions

- [ ] What exact public identifier and parameterization does E5-F2 approve for
  the first charged GPU model?
  - Blocking point: E5-F3 capability registration and documentation must use the
    frozen E5-F1/F2 decision and must not invent an alias or expose all CPU
    charged strategies.
- [ ] Does E5-F1 expose term-majorant dispatch as one helper or inline branches
  in the sampling kernel?
  - Implementation rule: extend the shipped interface in place; do not create a
    second selector or parallel configuration abstraction.
- [ ] Is an exhaustive unique-active-pair scan the accepted initial charged
  majorant after P1 evidence is reviewed?
  - Proposed answer: yes, because it is safe for arbitrary supported charge
    signs/magnitudes. A tighter bound is allowed only with a mathematical proof
    and pairwise regression coverage.
- [ ] Which deterministic tolerances and repeated-seed sigma multiplier are
  justified by the approved charged model's independent fixtures?
  - Resolve in P1/P2 and record values beside each test; do not inherit a blanket
    tolerance from unrelated Brownian cases.
- [ ] Should the mechanism configuration type be re-exported above the concrete
  coagulation module?
  - Open but non-blocking: follow E5-F1's stable API decision. E5-F3 does not add
    a second export path.

Resolved scope boundaries: particle-resolved only; one stochastic pass;
caller-owned buffers/RNG; Warp CPU required when Warp is installed; CUDA
optional; no high-level runnable or general performance redesign. Classifier
diagnostics: none.
