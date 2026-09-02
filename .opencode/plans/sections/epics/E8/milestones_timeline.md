# Milestones and Timeline

Calendar dates require owner scheduling; ordering and exit evidence are fixed.

| Milestone | Planned Date | Actual Date | Status | Notes |
|-----------|--------------|-------------|--------|-------|
| Capture lifecycle established | TBD | 2026-08-30 | Shipped | E8-F1 host-side contract handoff; no captured fixed-loop smoke test has shipped. #1550 focused checks passed (2 graph-document tests and 16 export tests), the untargeted runner passed (6382 passed, 9 skipped, 94% coverage), and `mkdocs build --strict` passed (exit 0). |
| Prepared enqueue boundary shipped | TBD | 2026-09-01 | In Progress | E8-F2 P1--P6/P8 shipped; P7 and E8-F3 resource work remain pending. |
| Graph capture and guarded replay established | TBD | - | Not Started | E8-F4; capture, replay, invalidation, and CUDA smoke evidence |
| Three-way correctness gate passes | TBD | - | Not Started | E8-F5; CPU, uncaptured GPU, and captured GPU evidence |
| Scaling and memory evidence published | TBD | - | Not Started | E8-F6; dated artifacts with environment metadata |
| Profiling evidence published | TBD | - | Not Started | E8-F7; machine-bounded launch and kernel evidence |
| User workflow and closeout accepted | TBD | - | Not Started | E8-F8; example, runbook, docs, and full validation |

No milestone is considered shipped from benchmark output alone. Each milestone
must include its implementation tests, focused validation command, and any
required documentation in the same child-plan delivery.
