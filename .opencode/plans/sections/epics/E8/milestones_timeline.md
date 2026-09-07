# Milestones and Timeline

Calendar dates require owner scheduling; ordering and exit evidence are fixed.

| Milestone | Planned Date | Actual Date | Status | Notes |
|-----------|--------------|-------------|--------|-------|
| Capture lifecycle established | TBD | 2026-08-30 | Shipped | E8-F1 host-side contract handoff; no captured fixed-loop smoke test has shipped. #1550 focused checks passed (2 graph-document tests and 16 export tests), the untargeted runner passed (6382 passed, 9 skipped, 94% coverage), and `mkdocs build --strict` passed (exit 0). |
| Prepared enqueue boundary shipped | TBD | 2026-09-01 | In Progress | E8-F2 P1--P6/P8 shipped; P7 and E8-F3 resource work remain pending. |
| Graph capture and guarded replay established | TBD | 2026-09-04 | Shipped | E8-F4 P1--P5; capture, replay, invalidation, private handle provenance, and CUDA smoke evidence |
| Three-way correctness gate passes | TBD | 2026-09-05 | Shipped | E8-F5 #1579: focused, coverage, and documentation assertions passed; approved strict-equivalent worktree validation passed with exit status 0. |
| Scaling and memory evidence published | TBD | - | Not Started | E8-F6; dated artifacts with environment metadata |
| Profiling evidence published | TBD | - | Not Started | E8-F7/T7; native-CUDA profiling and machine-bounded recommendations. |
| User workflow and closeout accepted | TBD | - | Blocked | E8-F8/T8 delivered the native-CUDA example, runbook, limitations, non-promoting P3 `UNSHIPPED/BLOCKED` ledger, and P4 documentation reconciliation. P4 focused closeout/runbook contracts passed (16 passed), and `mkdocs build --strict` passed (exit 0). Epic H remains Active/unshipped: E8-F2, E8-F6, E8-F7/T7, and designated-device final-revision H1--H11 rows still block closeout; no CUDA evidence is claimed. |

No milestone is considered shipped from benchmark output alone. Each milestone
must include its implementation tests, focused validation command, and any
required documentation in the same child-plan delivery.
