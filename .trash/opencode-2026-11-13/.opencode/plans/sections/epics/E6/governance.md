# Governance

| Date | Decision | Owner | Impact |
|------|----------|-------|--------|
| 2026-07-21 | Preserve nine ordered feature tracks E6-F1 through E6-F9 | Epic owner | Child plans and dependency reviews use the issue ordering. |
| 2026-07-21 | CPU references precede corresponding GPU parity features | Physics and GPU maintainers | E6-F1 gates F2; E6-F7 gates F8. |
| 2026-07-21 | Keep GPU APIs low-level, fixed-shape, and explicit-transfer | GPU maintainers | Prevents hidden fallback, resizing, and Epic G scope leakage. |
| 2026-07-21 | Require Warp CPU; treat CUDA as optional evidence | Test maintainers | Delivery is reproducible without mandatory CUDA hardware. |
| 2026-07-21 | Resampling defaults on and volume scaling defaults off | Physics maintainers | Defines E6-F6 exhaustion semantics and pre-mutation failure behavior. |

Cross-cutting contract changes require review from a physics owner, a GPU
owner, and a test owner. Changes to container fields or public exports require
an explicit compatibility review. E6-F9 closes the epic only after every child
acceptance signal and both roadmap cross-links are verified.
