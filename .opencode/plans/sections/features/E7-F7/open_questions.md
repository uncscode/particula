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

- [x] Are edge weights transfer fractions per scheduler step or physical rates
  multiplied by `time_step`?
  - Resolved 2026-07-27: Edge weights are physical inverse-time rates integrated
    over `time_step`.
  - Rationale: This matches existing finite-step transport-like semantics and
    avoids a weight whose meaning changes with scheduler cadence.
  - Evidence:
    - `particula/gpu/kernels/dilution.py:323` - the existing finite-step sink
      uses `alpha` in s^-1 and integrates it over `time_step`.
  - Resolved by: PR #1452 decision

- [x] Should volume updates occur before or after edge amount construction?
  - Resolved 2026-07-27: Construct all edge amounts from pre-step volume, then
    apply final volume and normalize once.
  - Rationale: A pre-step snapshot supports one auditable extensive ledger and
    deterministic unequal-volume mixing semantics before final normalization.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1565` - `ParticleData.volume`
      owns simulation-volume evolution but no relative ordering is specified.
  - Resolved by: PR #1452 decision

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
