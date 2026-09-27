# Risk Register

Owners below are responsible track roles, not assigned individuals. All risks
remain open until implementation evidence exists.

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|---|---|---|---|---|---|
| Volume normalization silently changes results | High | High | Audit raw/getter semantics by distribution; non-unit-volume independent regression fixtures | M1/M3 | Open |
| Gas ordering or partitioning loses species alignment | High | High | Freeze ordered mapping and test mixed gas categories, permutation/mismatch rejection and per-species outputs | M1/M4 | Open |
| Per-particle mass confused with population inventory | Medium | High | Separate derived-mass assertions from concentration-weighted conservation | M1/M3 | Open |
| Replacement partially publishes incompatible state | Medium | High | Read-only candidate validation, coordinated commit, identity and unchanged-state rejection tests | M2 | Open |
| Physics strategies become a new state facade | Medium | High | Process-owned configuration review; no duplicate storage or compatibility API at closeout | M1/M3/M6 | Open |
| Dilution or nucleation failure semantics regress | Medium | High | Migrate preflight/rollback and per-substep atomicity assertions with fixtures | M4 | Open |
| CPU wall-loss cases are missed by default collection | High | High | Explicitly execute adjacent suite plus normally collected suite | M3/M6 | Open |
| Early deletion removes scientific coverage or supported presets | Medium | High | Consumer/test/preset inventories; require replacement evidence before M6 deletion | M2/M6 | Open |
| Tutorials/notebook pairs remain broken | High | Medium | Complete supported inventory with source-first sync, execution and strict docs gate | M5 | Open |
| CPU replacement rebinds GPU state or transfer helpers are deleted | Medium | High | Protected conversion/export/resident regression matrix and explicit no-auto-rebind assertions | M4/M6 | Open |
| Temporary bridges survive release | Medium | High | Inventory owners/removal gates; zero open compatibility items at closeout | M6 | Open |
| Tool schema cannot store epic phases | Certain | Medium | Persist six supported milestones and expose exact limitation to orchestrator; child phases remain separate | Orchestrator | Open |
