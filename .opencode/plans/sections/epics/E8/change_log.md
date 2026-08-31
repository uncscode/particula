# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-08-30 | Shipped the E8-F1 #1550 P4 validation handoff: focused documentation checks passed (2 passed) and export checks passed (16 passed), the untargeted runner passed (6382 passed, 9 skipped, 94% coverage), and `mkdocs build --strict` passed (exit 0). E8-F1 is complete; E8-F2--E8-F8 remain unshipped, including native capture/replay and the user example. | implementation |
| 2026-08-30 | Fixed the launch-overhead benchmark matrix at 1, 10, 100, and 1000 repeated timesteps with explicit unavailable rows for budget failures | user decision |
| 2026-08-30 | Required captured-graph teardown before checkpoint or finalize; continuation and restart always require fresh capture | user decision |
| 2026-08-30 | Initial Epic H plan drafted from the authoritative roadmap and eight ordered E8-F1--E8-F8 feature tracks; classifier diagnostics preserved as none | plan-epic-drafter |
