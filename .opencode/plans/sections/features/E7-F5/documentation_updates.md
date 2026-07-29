# Documentation Updates

- [x] Added the published concrete-only scheduler/process-order section to the
  feature guide, including resolver/profile dependence, virtual freshness edges,
  state authority, and the closed diagnostics protocol.
- [x] Updated the roadmap with shipped E7-F5 evidence without claiming E7-F7
  transport, E7-F8 final RNG semantics, or E7-F9 closeout.
- [x] Updated the feature and checkpoint pages with resident state authority,
  explicit checkpoint boundaries, and no-hidden-transfer rules.
- [ ] Deferred to E7-F9: extend a scheduler-focused example only when dependent
  public APIs exist. `docs/Examples/gpu_complete_process_sequence.py` remains an
  illustrative explicit-transfer five-call sequence, not a resident scheduler
  example.
- [x] Updated `AGENTS.md` with resolver-produced process order, focused
  validation commands, freshness invalidations, and unsupported boundaries.
- P1 (#1492) required no user-facing documentation because it is an unexported,
  declaration-only internal boundary. E7-F5 plan sections now record its
  implementation and retained non-scheduler boundary.
- P2 (#1493) likewise required no user-facing documentation: its scheduler
  foundation is direct-import-only, declaration-only, and unexported. The
  module and record docstrings document canonical ordering, P1-first validation,
   closure, and no-launch semantics.
- P4 (#1495) required no user-facing documentation or export change because
  `particula.execution.state_updates` is concrete-only and direct-import-only.
  Its module and focused tests document exact graph-node binding, preflight,
  protected fields, and empty-schema no-op behavior without claiming scheduling,
   derived refresh, transport, host transfer, fallback, or lifecycle behavior.
- P5 (#1496) added concrete-only architecture documentation in
  `.opencode/guides/architecture_reference.md` and
  `.opencode/guides/architecture/architecture_outline.md`. It records exact
  identity binding, caller-reported successful nodes, cursor/virtual-writer
  ordering, stale-marker partial-failure semantics, identity preservation, the
  delegated configuration-fingerprint-read caveat, and exclusions for
   lifecycle, transfer, fallback, full scheduler dispatch, and general process
   dispatch. No public documentation or export changed.
- P6 (#1497) updated `.opencode/guides/architecture_reference.md` and
  `.opencode/guides/architecture/architecture_outline.md` with the
  direct-import-only diagnostics and resident scheduler boundaries: two closed
  snapshots, caller-owned output/nonaliasing and canonical empty no-ops, exact
  ten-node scheduling, identity-bound lifecycle, consumer-window refreshes, and
  no-transfer/no-fallback/no-rollback limits. No public export changed.
- [x] Added focused Warp-independent documentation regression assertions and ran
  documentation regressions and `mkdocs build --strict`; no paired notebook was
  touched.
