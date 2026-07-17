# Success Criteria

- [x] The E5-approved charged identifier resolves as executable alone for
  particle-resolved execution; unsupported variants and distributions fail before
  mutation.
- [ ] The charged identifier resolves with Brownian in one combined pass (P3).
- [x] The private charged majorant is finite and non-negative and bounds 100% of
  unique compact-active charged pair rates in every deterministic P1 proof
  fixture; invalid and zero candidates safely produce zero.
- [ ] Brownian-plus-charged pair rates are the sum of enabled terms, and the sum
  of term majorants bounds every tested combined pair rate.
- [x] Charged-only calls perform one candidate/acceptance pass, write one
  collision pair/count buffer set, and advance one per-box RNG stream.
- [x] Charged-only repeated-seed collision counts meet independent aggregate
  bounds; combined evidence remains P3.
- [x] Charged accepted pairs are distinct, in bounds, initially active, sorted,
  and disjoint within each box.
- [x] Charged-only successful calls conserve per-box species mass and total
  charge; donor mass, concentration, and charge clear after merge.
- [x] Charged-only caller collision buffers and persistent RNG buffers preserve
  identity and reuse without hidden reseeding.
- [x] Invalid charged configurations and inputs leave particles, output buffers,
  and RNG states unchanged.
- [ ] Existing omitted/explicit Brownian regressions pass unchanged.
- [ ] Warp CPU evidence passes when Warp is installed; optional CUDA passes or
  skips cleanly; changed-code coverage is at least 80% without threshold changes.
- [ ] Documentation states exact supported mechanisms, ownership, devices,
  limitations, and focused reproduction commands without broader claims.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Executable charged configurations | 0 | charged-only; combined deferred | Capability tests |
| Deterministic active pairs exceeding charged majorant | Not covered | 0 | Majorant proof tests |
| Stochastic passes per combined call | Brownian only | Exactly 1 | RNG/selector integration tests |
| Accepted merges conserving per-box species mass | Brownian baseline | 100% | Kernel integration tests |
| Accepted merges conserving per-box total charge | Pair/merge only in E5-F2 | 100% | Charged execution tests |
| Caller-owned output/RNG identity preservation | Brownian baseline | 100% | Ownership regression tests |
| Validation failures after state or RNG mutation | 0 tolerated | 0 | Snapshot tests |
| Required Warp CPU focused tests passing | Brownian baseline | 100% | Focused pytest run |

## P2 Completion Boundary

P2 satisfies charged-only execution, selection/acceptance, merge, preflight,
ownership, conservation, and stochastic criteria. Brownian-plus-charged
additivity and combined stochastic evidence remain P3 work.
