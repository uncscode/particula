# Governance

| Date | Decision | Owner | Impact |
|---|---|---|---|
| 2026-07-12 | Preserve seven ordered feature tracks and dependency gates | Epic maintainers | Prevents gas coupling or evidence work from bypassing physics prerequisites |
| 2026-07-12 | Use fixed four-substep integration | GPU maintainers | Stable launches, bounded work, graph-oriented design |
| 2026-07-12 | Keep transfers explicit and schemas fp64/fixed-shape | GPU maintainers | Maintains E1-E3 ownership and precision contracts |
| 2026-07-12 | Require tests in every implementation track | Reviewers | T6 consolidates rather than postpones evidence |

Child-plan owners propose API or numerical changes in their plan/PR. Changes to
model support, units, ownership, diagnostic semantics, or dependency order
require GPU and condensation-domain review. Conservation regressions, hidden
synchronization, or unsupported model fallbacks block merge.
