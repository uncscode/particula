| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Builder validation may expose a parameter contract that does not match existing latent-heat strategy constructor expectations | Medium | High | Lock builder setter names in P1 and add failure-path tests before factory registration begins | TBD | Open |
| Factory registration could publish an unstable or unclear public key (`"latent_heat"` vs alias) that later forces docs churn or compatibility shims | Medium | Medium | Resolved by shipping only the final `"latent_heat"` key and documenting that same path in the feature docs | TBD | Mitigated |
| Namespace exports or docs examples may drift from the final shipped builder path | Medium | Medium | Mitigated by coupling P3 export smoke tests with the completed `docs/Features/condensation_strategy_system.md` sync and updating plan sections after docs-validator drift surfaced | TBD | Mitigated |
