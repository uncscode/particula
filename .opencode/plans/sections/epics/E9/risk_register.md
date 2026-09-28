# Risk Register

Owners below are responsible track roles; Kyle/Gorkowski is final approver.
Implementation risks remain open until evidence exists.

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
| Tool schema cannot store epic phases | Certain | Medium | Accepted supported milestones with detailed child phases in review | E9 | Resolved design choice |
| Shared metadata is silently lost or mismatched | High | High | D1–D2 copy/transfer/checkpoint propagation, versioned interpretation and pre-mutation process compatibility tests | M1/M4/M6 | Open |
| PDF samples mistaken for integrated counts | High | High | Radius-only dN/dr, explicit quadrature/Jacobians and independent moment tests | M1/M3 | Open |
| CPU fix leaves direct/prepared/resident GPU normalization inconsistent | High | High | Per-family V=0.25/1/4 matrix, raw-preserving transfer and cross-layer conservation | M3/M4/M6 | Open |
| Expanded normalization audit exceeds original phase estimates | High | Medium | Re-size and split serial bounded implementation PRs before issue generation; preserve all gates | E9 | Open |
