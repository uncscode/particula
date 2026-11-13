# Governance

| Date | Decision | Owner | Impact |
|------|----------|-------|--------|
| 2026-07-26 | Use issue #1451 and roadmap Epic G as scope authority | Epic owner | Preserves the ordered nine tracks and exit bar |
| 2026-07-26 | Keep CPU behavior as the independent reference | Feature owners | Backend adapters require explicit parity evidence |
| 2026-07-26 | Prohibit silent fallback and hidden transfers | API reviewers | Every device transition is requested and observable |
| 2026-07-26 | Preserve fixed-shape state and deliberate exports | GPU maintainers | Integration cannot widen low-level APIs accidentally |
| 2026-07-26 | Defer performance/graph capture and autodiff | Epic owner | Maintains Epic H and Epic I boundaries |

## Review and Approval Process

- E7-F1's capability matrix and protocol require architecture and public-API
  review before E7-F6 freezes errors, fallback, and exports.
- Every child plan documents supported and unsupported configurations, mutation
  and identity behavior, device/transfer behavior, and failure atomicity.
- Physics or numerical changes require domain-owner approval and are normally
  redirected out of E7 because this epic integrates shipped physics.
- Public exports are reviewed against `particula/gpu/tests/kernel_exports_test.py`.
- Dependency gates in `dependency_map.md` control merge readiness. Parallel
  tracks may merge independently only after their shared gate is shipped.
- E7-F9 may close the epic only when the success checklist and roadmap exit bar
  have reproducible commands and published evidence.
