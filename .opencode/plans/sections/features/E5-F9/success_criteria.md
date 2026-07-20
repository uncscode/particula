# Success Criteria

- [x] Canonical guides accurately enumerate shipped mechanisms/combinations and
  explicitly exclude high-level integration, fallback, required CUDA, and
  deferred physics/runtime claims.
- [x] The direct example imports without Warp and runs on Warp CPU when installed
  with explicit transfers, caller buffers, persistent RNG, and valid restored
  conservation checkpoints.
- [x] P3: Both roadmap files contain E5 and each ID E5-F1 through E5-F9 exactly
  once in the pre-closeout inventory, with canonical statuses and resolving
  artifact links. The issue #1374 reconciliation is recorded as shipped.
- [x] E5-F1 through E5-F8 are shipped and all required E5-F7/E5-F8 evidence,
  example, docs, link, lint, and fast-test gates pass.
- [x] A negative closeout test proves any failed prerequisite blocks closeout.
- [x] After all gates passed, E5 is consistently shipped and Epic F active.
- [ ] Classifier diagnostics remain recorded as `none`.

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Assigned E5 child IDs in roadmap matrix | 0 | 9/9 | Docs test |
| Required artifact links resolving | incomplete | 100% | Link validation |
| Direct example Warp CPU result | absent | pass when Warp installed | Example test |
| Required release gates passing | not run | 100% | P4 checklist/CI |
| Premature closeout paths | possible manual drift | 0 | Negative gate test |
| Coverage threshold | >=80% | unchanged, >=80% changed code | pytest-cov |
