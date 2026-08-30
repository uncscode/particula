# Success Criteria

- [ ] Small and medium representative workloads have stable IDs and record
  boxes, particles, species, replay count, process order, communication mode,
  warmup, sample count, and fixed fixture provenance.
- [ ] Every measured artifact records date, exact command, CUDA device name and
  architecture, driver/runtime, Warp, Python, profiler version, timer, and
  synchronization method.
- [ ] Captured and uncaptured runs use identical prepared fixture contracts and
  retain raw host launch and synchronized elapsed samples separately.
- [ ] Setup, allocation, graph construction, warmup, serialization, and profiler
  startup are excluded from replay measurements or separately labeled.
- [ ] Dominant kernels have duration and invocation evidence plus occupancy and
  memory-access metrics where the qualified tool exposes them; missing metrics
  are explicitly unavailable and never synthesized.
- [ ] Kernel contributions and host-launch contribution reconcile within the
  documented method limits, with unattributed time retained rather than hidden.
- [ ] Each recommendation cites workload, machine, metric, raw artifact, and
  confidence/limitation, and no recommendation changes scientific contracts.
- [ ] Missing CUDA, capture, profiler tools, permissions, or counters cleanly
  records unavailable evidence without CPU fallback.
- [ ] Fast tests, the untargeted repository coverage runner, documentation
  contract tests, and `mkdocs build --strict` pass without lowering thresholds.
- [ ] T7/E8-F7 versus stale E8-F8 parent mapping is reconciled before closeout.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Workload classes profiled | 0 resident classes | Small and medium, or explicit unavailable rows | Normalized profile artifact |
| Launch-cost categories | Aggregate elapsed only | Uncaptured dispatch, captured graph launch, synchronized elapsed | Raw timing samples |
| Kernel attribution | None for resident loop | Dominant kernels plus explicit unattributed share | Nsight/Warp export |
| Occupancy/memory evidence | Not recorded | At least one supported occupancy and one memory-access metric per dominant kernel, or unavailable reason | Nsight Compute export |
| Recommendation provenance | Informal | 100% cite machine, workload, metric, and artifact | Decision table tests |
| CPU fallback in CUDA profile | Not applicable | 0 | Device metadata and routing tests |
| Default-suite benchmark execution | Disabled | Remains disabled | Pytest collection tests |
