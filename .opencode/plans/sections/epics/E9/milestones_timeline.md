# Milestones and Timeline

Target: v0.3.0; no fixed calendar deadline. All dates remain unset until
maintainer scheduling. All implementation milestones are Not Started.

The plan store rejects `add-phase` for epic records. Accordingly, these six
program phases are persisted as six supported **milestones**, not fabricated
`E9-P*` records. Detailed implementation phases belong to the existing child
maintenance plans and must include co-located tests.

| Program phase / milestone | Planned date | Actual date | Status | Exit gate |
|---|---|---|---|---|
| T1 / E9-M1: contracts and accessors (M) | Unscheduled | — | Not Started | Units, ordered alignment and accessor tests validated |
| T2 / E9-M2: construction and flat Aerosol (L) | After validated M1 | — | Not Started | Identity and coordinated replacement validated |
| T3 / E9-M3: CPU scientific processes (L) | After validated M2 | — | Not Started | Scientific regressions and both wall-loss suites validated |
| T4 / E9-M4: runnables and adapters (L) | After validated M3 | — | Not Started | Composition and documented failure semantics validated |
| T5 / E9-M5: examples and documentation (L) | After validated M4 | — | Not Started | Supported examples and notebook pairs execute; strict docs pass |
| T6 / E9-M6: deletion, integration and release docs (L) | After validated M5 | — | Not Started | Legacy audit cleared; full regression/lint/docs and retained GPU boundaries validated |

No milestone has actual completion evidence in this drafting run. A required
unavailable validation blocks the corresponding handoff; optional CUDA clean
skips are recorded separately and never presented as measured GPU success.
