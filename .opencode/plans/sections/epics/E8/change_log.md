# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-09-04 | Reconciled #1579 E8-F5-P5 as In Progress. Focused and documentation assertions plus untargeted coverage passed (6634 passed, 24 skipped, 1 xfailed, 92.92%); optional CUDA cleanly skipped. `mkdocs build --strict` is unavailable in the isolated worktree, so E8-F5 and the correctness milestone remain unshipped. | adw-build |
| 2026-09-01 | Reconciled #1559 E8-F2-P8 documentation closeout: P1--P6 and P8 are shipped; P7 remains pending. Focused assertions passed (22 passed in 0.09s) and `mkdocs build --strict` passed (exit 0 in 14.67s). E8 remains In Progress with E8-F2 included; E8-F3--E8-F8 remain pending. | plan-update-full |
| 2026-08-30 | Shipped the E8-F1 #1550 P4 validation handoff: focused documentation checks passed (2 passed) and export checks passed (16 passed), the untargeted runner passed (6382 passed, 9 skipped, 94% coverage), and `mkdocs build --strict` passed (exit 0). E8-F1 is complete; at that time E8-F2--E8-F8 remained unshipped, including native capture/replay and the user example. | implementation |
| 2026-08-30 | Fixed the launch-overhead benchmark matrix at 1, 10, 100, and 1000 repeated timesteps with explicit unavailable rows for budget failures | user decision |
| 2026-08-30 | Required captured-graph teardown before checkpoint or finalize; continuation and restart always require fresh capture | user decision |
| 2026-08-30 | Initial Epic H plan drafted from the authoritative roadmap and eight ordered E8-F1--E8-F8 feature tracks; classifier diagnostics preserved as none | plan-epic-drafter |
