# Child Plans

### Feature Tracks

The issue defines the following eight ordered tracks. Each implementation track
must ship its own co-located unit and contract tests.

| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E8-F1 | Graph-Capture Capability and Lifecycle Contracts | Shipped | P1--P4 delivered; #1550 focused checks passed (2 graph-document tests and 16 export tests), the untargeted runner passed (6382 passed, 9 skipped, 94% coverage), and `mkdocs build --strict` passed (exit 0). This is a host-side contract handoff only: no native capture/replay or user example shipped. |
| E8-F2 | Capture-Ready Device Enqueue Paths | In Progress | P1--P6/P8 shipped; P7 composition remains pending. |
| E8-F3 | Registry Preallocation, Identity Reuse, and Byte Accounting | Pending | Preallocate and pin process, communication, diagnostic, and RNG sidecars |
| E8-F4 | Resident Graph Capture and Guarded Replay Lifecycle | Shipped | P1--P5 delivered; private handle provenance, guarded replay, teardown, and three-way validation are covered. |
| E8-F5 | Captured Full-Loop Parity and Lifecycle Validation | Shipped | P1--P5 shipped; #1579 focused, coverage, documentation, and approved strict-equivalent worktree validation passed. |
| E8-F6 | Multi-Box Scaling Benchmarks and Memory-Budget Evidence | Pending | Measure scaling and publish reproducible memory-budget evidence behind opt-in CUDA gates |
| E8-F7 | CUDA Profiling and Machine-Bounded Performance Decisions | Pending | Record occupancy, memory access, and captured-versus-uncaptured launch overhead |
| E8-F8 | Graph-Capture Example, Runbook, Limitations, and Closeout | Pending | Publish the runnable workflow and close the epic with literal evidence |

### Maintenance Tracks

Maintenance Tracks: none
