| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Builder validation may expose a parameter contract that does not match existing latent-heat strategy constructor expectations | Medium | High | Lock builder setter names in P1 and add failure-path tests before factory registration begins | TBD | Open |
| Factory registration could publish an unstable or unclear public key (`"latent_heat"` vs alias) that later forces docs churn or compatibility shims | Medium | Medium | Resolve the public key before P4 and cover the accepted key in factory regression tests | TBD | Open |
| Namespace exports or docs examples may drift from the final shipped builder path | Medium | Medium | Keep P3 export smoke tests and P4 docs updates coupled, then rerun the scoped `pytest -Werror` command after doc edits | TBD | Open |
