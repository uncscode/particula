# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Empirical coefficient used outside observational domain | Medium | High | Require explicit closed bounds, fail closed, and document provenance/units. | Physics owner |
| Number and mass concentration are confused | Medium | High | Centralize SI conversion and test hand-calculated dimensions. | Physics owner |
| Species-wise limiting changes source composition | Medium | High | Use one shared admission factor and verify each species. | Process owner |
| Full slots cause source loss or gas-only depletion | Medium | Critical | Use E6-F5/F6 complete plan/commit; reject before writes and expose residual. | Process owner |
| Slot packaging changes represented totals | Medium | High | Separate physical events from computational slots and test number/mass. | Particle owner |
| Multi-box error leaves partial update | Low | High | Validate/plan every box before commit and snapshot all state. | Process owner |
| Scope expands into CNT, Vehkamäki, chemistry, or GPU | Medium | Medium | Keep supported/deferred tables and E6-F8/Epic G boundaries. | E6 owner |
| E6-F5/F6 APIs evolve incompatibly | Medium | Medium | Freeze shared records in P1-P3 and coordinate cross-plan changes. | E6 owner |
| Large rates overflow or destabilize stepping | Low | High | Validate finite products, inventory-limit, support explicit substeps. | Physics owner |
| Docs overstate scientific predictiveness | Medium | High | Peer-review equations, citations, domains, and limitations. | Docs owner |
