# Success Criteria

- [x] Every executable single and approved additive mechanism row from E5-F3
  through E5-F6 appears in one canonical validation table.
- [x] Independent fp64 pair/property values match Warp results at explicit,
  scale-appropriate tolerances; no expected value calls the Warp helper under
  test.
- [x] Every explicit active unordered pair has a finite non-negative rate bounded by its
  independently checked mechanism or summed majorant.
- [ ] Repeated-run collision aggregates fall within a predeclared statistical
  confidence or sigma bound without exact pair replay requirements.
- [ ] Per-box/per-species mass is conserved at tight physical tolerances for all
  executable rows and total charge is separately conserved for charge-bearing
  rows.
- [ ] Multi-box isolation, inactive slots, zero/one/two/many active particles,
  mixed scales, zero rates, and collision capacity boundaries pass.
- [ ] Caller-owned output buffers retain identity; persistent RNG state advances
  or resets only as requested; invalid inputs preserve particles, buffers, and
  RNG state.
- [ ] Warp CPU executes all required cases when Warp is installed. CUDA runs the
  applicable matrix when available and otherwise skips cleanly.
- [ ] Focused validation and the full coagulation suite pass without lowering
  coverage or requiring slow/performance benchmarks.
- [ ] Published documentation states tested rows, tolerances, statistical
  bounds, device policy, limitations, and reproduction commands.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Executable mechanism rows represented | Fragmented by feature | 100% of E5-F3-F6 approved rows | Canonical case table |
| Deterministic pair checks with independent oracle | Mechanism-local/partial | 100% of represented rows | `coagulation_validation_test.py` |
| Majorant violations over enumerated fixture pairs | Not consolidated | 0 | Deterministic matrix |
| Mass conservation failures | Not consolidated | 0 per box/species | End-to-end matrix |
| Charge conservation failures on charged rows | Not consolidated | 0 | End-to-end matrix |
| Stochastic rows outside declared bound | Not consolidated | 0 | Repeated-seed matrix |
| Required device coverage | Uneven | Warp CPU for every row when Warp is installed | Pytest results |
| CUDA behavior | Uneven/local | Pass when available; clean skip otherwise | `cuda` marker results |
