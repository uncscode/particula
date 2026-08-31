# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-08-30 | Started E8-F1-P4 for issue #1550: added bounded developer-contract documentation and hardware-free documentation/export regressions. Focused checks passed (17 passed); the untargeted runner passed with 6381 passed, 9 skipped, 1 xfailed, and 93.59% coverage. `mkdocs build --strict` is unavailable because no supported MkDocs runner is available, so P4 is not delivered and no parent handoff is claimed | implementation |
| 2026-08-30 | Delivered E8-F1-P3 for issue #1549: added exact direct-module-only `ResidentGraphCaptureBinding` lifecycle/gating in `particula/execution/graph_capture.py`, optional request binding and pre-token scheduler gate in `particula/execution/resident_scheduler.py`, and graph-capture/full-loop regression coverage; no user documentation was changed | implementation |
| 2026-08-30 | Delivered E8-F1-P2 for issue #1548: added Warp-free immutable host lifecycle metadata and explicit transitions in `particula/execution/graph_capture.py`, with hardware-free lifecycle/import-boundary coverage in `particula/execution/tests/graph_capture_test.py`; native capture/replay and resident-binding integration remain P3 work | implementation |
| 2026-08-30 | Delivered E8-F1-P1 for issue #1547: added the Warp-import-free concrete graph-capture capability/signature declarations and focused unit tests; capture/replay and lifecycle remain deferred to P2-P3 | implementation |
| 2026-08-30 | Resolved graph teardown to always retire the Particula owner and call native release only through a documented, version-qualified public Warp API | user decision |
| 2026-08-30 | Created first-pass E8-F1 plan with four issue-sized phases covering graph-capture capability, compatibility signatures, lifecycle/invalidation, recapture gates, co-located tests, and documentation | plan-feature-drafter |
| 2026-08-30 | Preserved classifier diagnostics (`none`) and linked the feature to parent E8 and sibling tracks E8-F2 through E8-F8 | plan-feature-drafter |
