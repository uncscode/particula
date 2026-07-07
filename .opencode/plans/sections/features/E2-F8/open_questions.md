# E2-F8 Open Questions

## Resolved Answers

1. P1 does not change runtime coagulation semantics. Current CPU coagulation
   still accepts multi-box `ParticleData`, and the new regressions lock in the
   present box-0-only read and mutation behavior for later phases to revisit.
2. Use `IsothermalCondensationStrategy.step()` plus one public pressure-delta or
   mass-transfer path that already flows through `_require_single_box` for
   representative multi-box rejection tests. Implemented choice: public
   `mass_transfer_rate(...)` coverage.
3. No general documentation update was needed in this audit-only phase; keep any
   future caller-managed per-box loop guidance deferred until a later doc phase.
4. Any later CPU dynamics boundary doc can still mention E2-F2 environment
   containers as a short cross-reference, but P1 should not imply CPU dynamics
   have full environment-aware multi-box support.

## Deferred Out-of-Scope Topics

- Full multi-box CPU condensation or coagulation implementation details.
- First-class all-box execution semantics for GPU strategy kernels.
- Container schema changes for representing environment boxes differently.
