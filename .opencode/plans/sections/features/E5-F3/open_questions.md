# Open Questions

- [x] What exact public identifier and parameterization does E5-F2 approve for
  the first charged GPU model?
  - Resolved 2026-07-16: use `charged_hard_sphere` for the complete CPU hard-
    sphere system-state chain. It has no tuning parameter and consumes existing
    particle radius, mass, signed charge counts, temperature, and pressure.
- [x] Does E5-F1 expose term-majorant dispatch as one helper or inline branches
  in the sampling kernel?
  - Resolved 2026-07-16: use one internal mask-driven component-majorant
    dispatcher paired with the component-rate dispatcher. Extend the shared
    selector in place and do not create a second selector or configuration type.
- [x] Is an exhaustive unique-active-pair scan the accepted initial charged
  majorant after P1 evidence is reviewed?
  - Resolved 2026-07-16: yes. It is the correctness-first bound for arbitrary
    supported signs, magnitudes, and mixed scales. A tighter replacement needs a
    mathematical proof and exhaustive pairwise regressions.
- [x] Which deterministic tolerances and repeated-seed sigma multiplier are
  justified by the approved charged model's independent fixtures?
  - Resolved 2026-07-16: use `rtol=1e-6, atol=0` for positive finite pair rates,
    `rtol=1e-6, atol=1e-30` for extreme repulsion, and exact-zero assertions for
    zero rates. Use 100 fresh seeds and a predeclared `3-sigma` aggregate bound.
- [x] Should the mechanism configuration type be re-exported above the concrete
  coagulation module?
  - Resolved 2026-07-16: no. Import it from
    `particula.gpu.kernels.coagulation`; E5-F3 adds no second export path.

Resolved scope boundaries: particle-resolved only; one stochastic pass;
caller-owned buffers/RNG; Warp CPU required when Warp is installed; CUDA
optional; no high-level runnable or general performance redesign. Classifier
diagnostics: none.
