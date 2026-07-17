# Success Criteria

- [ ] The E5-approved charged identifier resolves as executable alone and with
  Brownian; unsupported variants and distributions fail before mutation.
- [x] The private charged majorant is finite and non-negative and bounds 100% of
  unique compact-active charged pair rates in every deterministic P1 proof
  fixture; invalid and zero candidates safely produce zero.
- [ ] Brownian-plus-charged pair rates are the sum of enabled terms, and the sum
  of term majorants bounds every tested combined pair rate.
- [ ] Each call performs one candidate/acceptance pass, writes one collision
  pair/count buffer set, and advances one per-box RNG stream.
- [ ] Charged-only and combined repeated-seed collision counts stay within the
  declared aggregate or sigma-based bounds of independent expected rates.
- [ ] Every accepted pair is distinct, in bounds, initially active, sorted, and
  disjoint within its box.
- [ ] Every successful call conserves each species' mass and total particle
  charge per box; donor mass, concentration, and charge clear after merge.
- [ ] Caller-owned collision pairs, collision counts, and persistent RNG buffers
  are returned/reused by identity with no hidden reseeding.
- [ ] Invalid configurations and inputs leave particles, output buffers, and RNG
  states unchanged.
- [ ] Existing omitted/explicit Brownian regressions pass unchanged.
- [ ] Warp CPU evidence passes when Warp is installed; optional CUDA passes or
  skips cleanly; changed-code coverage is at least 80% without threshold changes.
- [ ] Documentation states exact supported mechanisms, ownership, devices,
  limitations, and focused reproduction commands without broader claims.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Executable charged configurations | 0 | charged; Brownian + charged | Capability tests |
| Deterministic active pairs exceeding charged majorant | Not covered | 0 | Majorant proof tests |
| Stochastic passes per combined call | Brownian only | Exactly 1 | RNG/selector integration tests |
| Accepted merges conserving per-box species mass | Brownian baseline | 100% | Kernel integration tests |
| Accepted merges conserving per-box total charge | Pair/merge only in E5-F2 | 100% | Charged execution tests |
| Caller-owned output/RNG identity preservation | Brownian baseline | 100% | Ownership regression tests |
| Validation failures after state or RNG mutation | 0 tolerated | 0 | Snapshot tests |
| Required Warp CPU focused tests passing | Brownian baseline | 100% | Focused pytest run |

## P1 Completion Boundary

P1 satisfies only the charged-majorant criterion above and Brownian dispatcher
regression coverage. Executable charged configurations, selection/acceptance,
merge behavior, public API support, and stochastic metrics remain P2/P3 work.
