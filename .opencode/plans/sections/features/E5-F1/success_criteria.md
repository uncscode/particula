# Success Criteria

- [ ] Existing calls that omit mechanism configuration execute the same
  Brownian particle-resolved path and preserve positional compatibility.
- [ ] Explicit Brownian configuration is seed-for-seed equivalent to the
  omitted-configuration path for accepted pairs, particle updates, and RNG.
- [ ] Configuration rejects empty, duplicate, unknown, unavailable, and
  non-particle-resolved requests with actionable errors.
- [ ] All caller-controlled configuration validation completes before
  allocation, Warp launch, particle mutation, output-buffer mutation, or RNG
  advancement.
- [ ] Pair rates and safe term majorants have one documented extension
  interface; enabled terms are summed before one acceptance pass.
- [ ] No mechanism can add a separate candidate stream, acceptance pass,
  collision buffer, or RNG update through the E5-F1 interface.
- [ ] The initial capability matrix distinguishes Brownian support from
  reserved E5 terms and names the downstream owner of each reserved term.
- [ ] `WarpParticleData` and CPU/Warp conversion schemas remain unchanged.
- [ ] Required Warp CPU tests pass, CUDA tests skip cleanly when unavailable,
  and Ruff/mypy checks pass without reducing coverage requirements.
- [ ] Documentation explicitly states the particle-resolved-only boundary and
  does not claim downstream mechanisms are already executable.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Explicit mechanism configurations | 0 | Brownian default plus stable reserved E5 identifiers | Resolver unit tests |
| Sampling passes per combined step | Undefined | Exactly 1 | Dispatch/sampling tests |
| RNG streams advanced per box per step | 1 Brownian stream | 1 for any supported combination | Persistent-RNG tests |
| Rejected configurations mutating caller state | Not specified | 0 | Preflight snapshot tests |
| Supported GPU distribution modes | Implicit particle-resolved | 1 explicit (`particle_resolved`) | Capability tests/docs |
| Legacy Brownian regression mismatches | N/A | 0 | Seeded equivalence tests |
