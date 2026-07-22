# Open Questions

All planning questions were reviewed against the current CPU and direct-Warp
contracts on 2026-07-21. Implementation measurements may tighten a recorded
tolerance, but must not silently weaken these decisions.

- [x] Which bounded nucleation-rate model and critical-cluster parameterization
  becomes the public CPU reference?
  - Decision: E6-F7 ships the caller-calibrated empirical source laws `J=A*C`
    and `J=K*C^2`, with SI-normalized coefficients, mandatory provenance and
    closed validity bounds. The caller supplies formation-particle composition,
    size, and any survival factor. This is explicitly a bounded particle-source
    reference, not a predictive critical-cluster model; CNT and the full
    Vehkamaki binary parameterization remain out of scope.
- [x] Which distribution moments must default resampling preserve?
  - Decision: preserve represented number, per-species represented mass,
    represented charge, and weighted particle volume (`radius^3`) at
    `rtol=1e-12`, `atol=1e-30`. Weighted mean radius and weighted `radius^2`
    use relative error; mixing state uses the absolute change in
    `get_mixing_state_index(w[:, None] * mass)`. Limits are caller/domain-owner
    policy. Canonical fixtures start with 2%, 2%, and 0.02 respectively, but
    those values are not universal defaults.
- [x] Should half-active particle slots be rejected or normalized?
  - Decision: a shared helper rejects contradictory records before mutation.
    Active means positive finite weight and positive finite total mass; free
    means all species mass, weight, and charge are exactly zero. This rule
    governs new E6 slot, exhaustion, and nucleation APIs; retrofitting legacy
    process entry points is outside E6.
- [x] What statistical evidence is required for neutral and charged wall loss?
  - Decision: use scale-stratified deterministic coefficient checks and 4,096
    Bernoulli observations per homogeneous survival stratum. Evaluate each
    stratum with an exact binomial interval using family-wise alpha `1e-6`.
    Neutral evidence has eight strata (four radii by two geometries); charged
    evidence has eight (four radii by image-only spherical and field-plus-
    potential rectangular cases), so each uses per-stratum alpha `1.25e-7`.
    Prefer survival probability near 0.5. Full CPU/Warp coefficients start at
    `rtol=1e-6`, `atol=0`, while zero branches and neutral fallback are exact.
    CPU/Warp RNG-sequence equality is not required.
- [x] Which diagnostics fields are reusable across activation, exhaustion, and
  nucleation?
  - Decision: caller-owned fixed-shape per-box diagnostics retain active/free,
    requested/released/activated slot counts, policy code, scale factor, and
    number/charge/volume/per-species-mass residuals. Nucleation additionally
    owns potential, gas-admitted, represented, gas-limited, representation-
    reduction, and residual event counts, plus limiting-species index and gas
    mass removed. No diagnostics become container fields.

## Scheduling and Metadata Decisions

- [x] Calendar dates remain unassigned until execution planning. Dependency
  gates, not fabricated dates, are authoritative while E6 is Draft.
- [x] Phase issue numbers remain `TBD`/`null` until each implementation phase is
  scheduled; each issue must be linked before that phase starts.
- [x] Epic `child_plans` metadata is synchronized with E6-F1 through E6-F9.
- [x] E6-F2 and E6-F4 metadata record their mandatory E6-F1 and E6-F3
  dependencies respectively.
