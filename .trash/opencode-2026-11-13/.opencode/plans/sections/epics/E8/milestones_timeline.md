# Milestones and Timeline

Maintainer closeout, 2026-09-27: E8 and all feature plans are Shipped.
The dated milestone history below preserves evidence available at the time;
unavailable measurement milestones are not promoted by plan closure.

Calendar dates require owner scheduling; ordering and exit evidence are fixed.

| Milestone | Planned Date | Actual Date | Status | Notes |
|-----------|--------------|-------------|--------|-------|
| Capture lifecycle established | TBD | 2026-08-30 | Shipped | E8-F1 host-side contract handoff; no captured fixed-loop smoke test has shipped. #1550 focused checks passed (2 graph-document tests and 16 export tests), the untargeted runner passed (6382 passed, 9 skipped, 94% coverage), and `mkdocs build --strict` passed (exit 0). |
| Prepared enqueue boundary shipped | TBD | 2026-09-27 | Shipped | E8-F2 and E8-F3 closed by maintainer decision. |
| Graph capture and guarded replay established | TBD | 2026-09-04 | Shipped | E8-F4 P1--P5; capture, replay, invalidation, private handle provenance, and CUDA smoke evidence |
| Three-way correctness gate passes | TBD | 2026-09-05 | Shipped | E8-F5 #1579: focused, coverage, and documentation assertions passed; approved strict-equivalent worktree validation passed with exit status 0. |
| Scaling and memory evidence published | TBD | - | Not Started | E8-F6; dated artifacts with environment metadata |
| Profiling evidence published | TBD | - | Not Started | E8-F7/T7; native-CUDA profiling and machine-bounded recommendations. |
| User workflow and closeout accepted | TBD | 2026-09-27 | Shipped | Maintainer closes E8-F8 and Epic H. The P3 measured-evidence ledger remains `UNSHIPPED/BLOCKED`; no CUDA evidence is inferred. |

No milestone is considered shipped from benchmark output alone. Each milestone
must include its implementation tests, focused validation command, and any
required documentation in the same child-plan delivery.
