| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Epic scope remains broader than the only child feature, leaving validation or acceleration-readiness work unplanned | High | High | Either add follow-on child plans before implementation starts or narrow the epic title/scope to match delivered work | TBD | Open |
| Public builder/factory names diverge between builder setters, factory keys, and docs examples | Medium | High | Lock the builder API in `E1-F1-P1`, decide the factory key before docs ship, and keep export tests plus docs examples aligned | TBD | Open |
| Latent-heat public API ships without enough regression coverage for import stability and warning-free factory usage | Medium | High | Require feature phases to land tests with code changes and rerun the scoped `pytest -Werror` condensation suite before ship | TBD | Open |
