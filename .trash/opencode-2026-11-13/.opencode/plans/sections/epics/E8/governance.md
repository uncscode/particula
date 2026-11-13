# Governance

| Date | Decision | Owner | Impact |
|------|----------|-------|--------|
| 2026-08-30 | Preserve eight ordered feature tracks E8-F1 through E8-F8 | Epic issue | Child plans follow the roadmap decomposition without merging or reordering |
| 2026-08-30 | Build capture around the shipped resident scheduler rather than introducing a second process-order authority | E8 maintainers | Captured and uncaptured execution share one semantic order |
| 2026-08-30 | Require explicit setup, replay, teardown, and recapture boundaries | E8 maintainers | No hidden allocation, fallback, migration, or automatic recapture |
| 2026-08-30 | Keep capture and profiling evidence CUDA-gated; retain Warp CPU for uncaptured parity | E8 maintainers | Unsupported environments skip cleanly and never masquerade as captured evidence |
| 2026-08-30 | Keep benchmarks opt-in and hardware-qualified | E8 maintainers | Default CI remains correctness-focused; performance claims remain reproducible and bounded |

## Review and Approval Process

- Every feature plan must identify its compatibility boundary, recapture
  triggers, mutation/rollback semantics, and focused validation commands.
- Changes to resident scheduler ordering, registry ownership, RNG continuation,
  checkpoint schema, or public exports require architecture and correctness
  review before implementation approval.
- Performance conclusions require review of raw artifacts and environment
  metadata; summaries alone are insufficient.
- E8-F8 may close the epic only after all earlier child tracks are shipped and
  the exit metrics in `success_metrics.md` have recorded evidence.
