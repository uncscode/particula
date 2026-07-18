# Success Criteria

- [x] The executable capability matrix explicitly registers every supported
  two-way row and the full Brownian+charged+sedimentation+turbulent-shear row;
  canonical ordering is equivalent and unsupported rows fail closed.
- [x] For every deterministic active-pair fixture, the private device total rate equals
  the independently calculated sum of enabled component rates within declared
  fp64 tolerances.
- [x] Every component and total majorant is finite and non-negative, and every
  tested pair satisfies `total_pair_rate <= sum(component_majorants)`.
- [x] Private selector diagnostics show each valid scheduled proposal uses one
  active-pair stream and at most one acceptance draw, independent of enabled
  term count; rejected/materially unbounded ratios draw and mutate nothing.
- [x] Two-way and full four-way aggregate collision counts satisfy declared
  stochastic bounds without requiring exact CPU/Warp pair replay.
- [x] Accepted pairs are sorted, in range, disjoint, and capacity bounded; one
  merge pass conserves each species' mass and total charge separately.
- [x] Caller-owned pair/count buffers are returned by identity and persistent
  RNG state advances/reinitializes only under the existing explicit contract.
- [x] Invalid aggregate arithmetic and material rate-over-majorant violations
  fail closed before selector/output/RNG mutation; P1 retains its existing
  preflight atomicity contract for deferred masks and invalid inputs.
- [x] Existing Brownian-only, charged-only, Brownian-plus-charged,
  sedimentation-only, and turbulent-shear-only regressions remain passing.
- [x] Warp CPU tests pass when Warp is installed; CUDA tests pass when available
  and skip cleanly otherwise; coverage thresholds remain unchanged.
- [x] Documentation states the exact additive matrix, required inputs, safe
  total-majorant rule, single-pass semantics, and support exclusions.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Acceptance passes per additive call | Undefined / risk of one per term | Exactly 1 | Test-local sampler diagnostics |
| Pair buffers and per-box RNG streams | Risk of one per term | 1 pair buffer and 1 RNG stream | Buffer identity and RNG tests |
| Deterministic additive pair-rate error | No additive GPU evidence | Within fixture-specific fp64 `rtol`/`atol` | Independent NumPy/CPU matrix tests |
| Majorant violations over enumerated pairs | No combined bound | 0 | All-active-pair bound tests |
| Species-mass and charge conservation failures | No combined evidence | 0 | Per-box conservation assertions |
| Preflight mutation failures | No combined contract | 0 mutated state/buffer/RNG cases | Snapshot tests |
| Required device evidence | Brownian/mechanism-local | Warp CPU pass; CUDA optional | Focused pytest device matrix |
