# Open Questions

- [x] Which particle representations are enabled in the first public
  communication capability?
  - Resolved 2026-07-27: Enable particle-resolved fixed-slot transport only and
    reject binned/discrete and continuous-PDF requests without conversion or
    fallback.
  - Rationale: Epic G's required multi-box regression is particle-resolved, and
    the roadmap warns that representation-specific slot semantics need distinct
    kernels.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1503` - the exit coverage calls
      for larger particle-resolved multi-box cases.
    - `docs/Features/Roadmap/data-oriented-gpu.md:1582` - particle slot contents
      require explicit transport rules and potentially distinct kernels.
  - Resolved by: plan-question-resolver

- [ ] Are edge weights transfer fractions per scheduler step or physical rates
  multiplied by `time_step`?
  - Open: The roadmap does not define edge-weight units, and choosing fractions
    versus rates changes public semantics and timestep dependence.
  - Recommendation: **A - Use physical rates with explicit inverse-time units**
  - Suggested answer: Choose **A** because existing transport-like dilution
    integrates an inverse-time coefficient over `time_step` and avoids a weight
    whose meaning changes with scheduler cadence.
  - Options:
    - [ ] A. Physical rates with explicit inverse-time units, integrated over
      `time_step` (Recommended)
    - [ ] B. Bounded transfer fractions already integrated per scheduler step
    - [ ] C. Two separately named declaration types for rates and fractions
  - Evidence considered:
    - `particula/gpu/kernels/dilution.py:323` - the existing finite-step sink
      uses `alpha` in s^-1 and integrates it over `time_step`.

- [ ] Should volume updates occur before or after edge amount construction?
  - Open: Both orderings are implementable, but they produce different
    unequal-volume mixing semantics and no canonical transport oracle exists.
  - Recommendation: **A - Construct amounts from pre-step volume**
  - Suggested answer: Choose **A** because a pre-step snapshot supports one
    auditable extensive ledger before normalization by prescribed final volume.
  - Options:
    - [ ] A. Build all edge amounts from pre-step volume, then apply final volume
      and normalize once (Recommended)
    - [ ] B. Apply prescribed volume updates first, then construct edge amounts
      from post-update volume
  - Evidence considered:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1565` - `ParticleData.volume`
      owns simulation-volume evolution but no relative ordering is specified.

- [x] Which open-boundary source/sink forms are part of T7?
  - Resolved 2026-07-27: Support prescribed dilution/outflow sinks for particle
    and gas concentrations; exclude nonzero-composition inlets, implicit
    reservoirs, and general source terms.
  - Rationale: Dilution is explicitly in T7 and has a shipped concentration-only
    finite-step contract, while no external source or reservoir schema exists.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1498` - T7 explicitly includes
      prescribed advection, dilution, expansion, and simple mixing.
    - `particula/gpu/kernels/dilution.py:323` - the shipped sink mutates only
      particle and gas concentrations using a finite-step rate.
  - Resolved by: plan-question-resolver
