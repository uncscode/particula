# Child Plans

Maintainer closeout, 2026-09-27: E8 and all eight feature plans are Shipped.
Measured CUDA evidence remains unavailable in the separate closeout ledger.

### Feature Tracks

The issue defines the following eight ordered tracks. Each implementation track
must ship its own co-located unit and contract tests.

| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E8-F1 | Graph-Capture Capability and Lifecycle Contracts | Shipped | P1--P4 delivered; #1550 focused checks passed (2 graph-document tests and 16 export tests), the untargeted runner passed (6382 passed, 9 skipped, 94% coverage), and `mkdocs build --strict` passed (exit 0). This is a host-side contract handoff only: no native capture/replay or user example shipped. |
| E8-F2 | Capture-Ready Device Enqueue Paths | Shipped | All phases closed by maintainer decision. |
| E8-F3 | Registry Preallocation, Identity Reuse, and Byte Accounting | Shipped | All phases closed by maintainer decision. |
| E8-F4 | Resident Graph Capture and Guarded Replay Lifecycle | Shipped | P1--P5 delivered; private handle provenance, guarded replay, teardown, and three-way validation are covered. |
| E8-F5 | Captured Full-Loop Parity and Lifecycle Validation | Shipped | P1--P5 shipped; #1579 focused, coverage, documentation, and approved strict-equivalent worktree validation passed. |
| E8-F6 | Multi-Box Scaling Benchmarks and Memory-Budget Evidence | Shipped | Plan closed; reviewed CUDA scaling/memory artifacts remain unavailable. |
| E8-F7 | CUDA Profiling and Machine-Bounded Performance Decisions | Shipped | Plan closed; T7 profiling and machine-bounded recommendations remain unmeasured. |
| E8-F8 | Graph-Capture Example, Runbook, Limitations, and Closeout | Shipped | Example, runbook, limitations, and closeout delivered. Plan closure does not certify designated-device H1--H11 evidence. |

### Maintenance Tracks

Maintenance Tracks: none
