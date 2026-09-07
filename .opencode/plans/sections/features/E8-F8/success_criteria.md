# Success Criteria

- [x] P1 (#1595) canonical example demonstrates supported setup, one explicit RNG
  initialization, capture, exactly two replays, synchronization, structural
  invalidation, retirement, renewal, fresh capture, and ordered teardown using
  concrete-only production boundaries.
- [x] P1 (#1595) hardware-free contract tests and an optional native-CUDA smoke
  row cover the example without substituting CPU or Warp-CPU capture evidence.
- [x] P2 (#1596) adds a hardware-free runbook contract; qualified CUDA execution
  remains optional evidence and is never replaced by CPU fallback or an
  unsupported captured-evidence claim.
- [x] The runbook distinguishes every structural recapture trigger from mutable
  payload changes that remain valid in pinned arrays.
- [x] The runbook explicitly excludes dynamic shapes/order/maps, automatic
  recapture, migration, fallback, retry, rollback, graph serialization,
  checkpointed handles, and portable performance guarantees.
- [ ] Closeout records date, revision, software/runtime/device identity, derived
  executable targets, exact commands, literal output, artifact links/checksums,
  and required/optional disposition.
- [ ] Every consumed E8-F6/E8-F7 normalized artifact matches the committed
  closeout manifest by schema, source revision, workload IDs, provenance, and
  SHA-256; local-only raw reports are not treated as retrievable closeout inputs.
- [ ] Every Epic H success metric links to current evidence; any failed,
  unavailable, missing, or stale required row leaves Epic H Active.
- [ ] One designated qualified CUDA device completes every required measured exit
  row at the final revision; clean skips, historical runs, and supplemental
  devices do not satisfy a missing required row.
- [ ] Focused tests, linters, untargeted repository coverage, changed-module
  coverage where applicable, plan validation, and `mkdocs build --strict` pass
  without lowering thresholds.
- [ ] The changed-module coverage list is frozen from shipped implementation
  records plus the final production diff, and every listed module has a literal
  per-target result.
- [ ] Parent/sibling labels consistently assign profiling to E8-F7 and the T8
  example, runbook, limitations, documentation reconciliation, roadmap, and
  closeout to E8-F8; T7 also owns machine-bounded recommendations.
- [x] P4 documentation reconciliation is implemented: focused
  closeout/runbook documentation contracts passed (16 passed), `mkdocs build
  --strict` passed (exit 0), and active split-plan validation is recorded for
  this worktree. This is documentation-only, non-promoting evidence and is not
  CUDA evidence.
- [ ] P3's `UNSHIPPED/BLOCKED` ledger remains non-promoting until E8-F2, E8-F6,
  E8-F7/T7, and each final-revision H1--H11 row passes on one designated
  qualified CUDA device.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Runnable full-loop graph examples | 0 | 1 canonical tested example | Example contract test |
| Recapture trigger categories documented | Fragmented plan text | 100% of compatibility-signature fields plus terminal lifecycle events | Runbook trigger matrix |
| Epic success metrics with evidence disposition | Draft checklist | 100% linked and pass/block classified | Closeout report |
| Required command rows with literal output | None | 100%; no inferred passes | Closeout schema test |
| Hidden CPU fallback in CUDA rows | Forbidden | 0 | Device metadata and dispatch tests |
| Full-package coverage | Repository configured | Meets or exceeds normal threshold | `.opencode/tools/run_pytest.py` |
| Documentation build failures | Unknown | 0 | `mkdocs build --strict` |
