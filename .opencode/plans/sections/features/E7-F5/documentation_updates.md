# Documentation Updates

- Add a user-facing scheduler/process-order section to the appropriate
  `docs/Features/` execution or GPU-resident simulation guide, including the
  canonical dependency diagram and supported process matrix.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` Track T5 status and evidence
  without claiming E7-F7 transport, E7-F8 final RNG semantics, or E7-F9 closeout.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with resident
  state authority, explicit checkpoint boundaries, and no-hidden-transfer rules.
- Extend a scheduler-focused example only when the dependent public APIs exist;
  keep `docs/Examples/gpu_complete_process_sequence.py` labeled illustrative.
- Update `AGENTS.md` with canonical process order, focused validation commands,
  environment/gas freshness rules, and explicit unsupported boundaries.
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
- Run documentation regressions and `mkdocs build --strict`; if a paired notebook
  is touched, edit its Jupytext `.py`, sync, execute, and commit both files.
