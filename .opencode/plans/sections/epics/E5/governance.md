# Governance

| Date | Decision | Owner | Impact |
|------|----------|-------|--------|
| 2026-07-16 | Treat issue #1320 and its nine feature-track descriptions as authoritative scope | Epic owner | Prevents scope expansion during drafting and review |
| 2026-07-16 | Preserve classifier diagnostics as `none` | Plan automation | Records that classification found no ambiguity requiring correction |
| 2026-07-16 | Keep support particle-resolved and low-level | GPU maintainers | Excludes binned/PDF and high-level runnable integration |
| 2026-07-16 | Add mechanism pair rates before one stochastic sampling pass | Physics reviewers | Aligns GPU combination semantics with CPU kernel addition and avoids sequential-pass bias |
| 2026-07-16 | Require fail-before-mutation and fail-before-RNG-advance validation | API reviewers | Preserves deterministic ownership and error behavior |
| 2026-07-16 | Require Warp CPU evidence; keep CUDA optional and additive | Test maintainers | Keeps routine validation portable without weakening device evidence |
| 2026-07-16 | Use feature tracks and milestones as the epic work breakdown | Plan maintainers | Epic plan schema does not support phase records; each child feature owns implementation phases and co-located tests |

Architecture or physics changes require review against the CPU reference,
declared supported variants, numerical tolerances, and support matrix. E5-F9
may mark the roadmap epic shipped only after E5-F1-F8 done signals pass and
plan, documentation, and artifact links agree.
