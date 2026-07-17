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
- [x] Can charged-only reuse the shared selector and apply kernels without a
  separate collision buffer or a per-pair species reduction?
  - Resolved 2026-07-17 by #1343: yes. A private fp64 total-mass scratch array
    is populated once per valid active slot, then consumed by the compact O(A²)
    majorant and charged candidate-rate path; one selector and one apply launch
    preserve the existing collision-buffer and merge ownership contracts.
- [x] Is finite-charge validation mandatory for charged-only requests even when
  Brownian's optional validation flag is omitted?
  - Resolved 2026-07-17 by #1343: yes. Charged-only preflight scans charge before
    output/RNG work or mutation; Brownian retains opt-in finite-charge validation.

- [x] How is the combined additive majorant formed without a second selection
  pass or double-counting the charged bound?
  - Resolved 2026-07-17 by #1344: normalize either canonical requested order to
    the combined mask and scan each compact active pair once with
    `_total_pair_rate(actual_mask)`. The finite non-negative maximum bounds the
    summed candidate rate and feeds the existing single selector/apply path.
- [x] Does combined execution retain charged preflight and caller resource
  ownership?
  - Resolved 2026-07-17 by #1344: yes. Finite-charge and charged-physics
    preflight precede output allocation, RNG initialization, and mutation; the
    existing caller buffers and persistent RNG stream are reused.

Resolved scope boundaries: particle-resolved only; one stochastic pass;
caller-owned buffers/RNG; Warp CPU required when Warp is installed; CUDA
optional; no high-level runnable or general performance redesign. Classifier
diagnostics: none.
