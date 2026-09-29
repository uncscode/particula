# Success Criteria

All checks are pending implementation acceptance, not drafting achievements.

- [ ] Approved ordered mapping covers particle/density, all gas categories,
  environment and process parameters, with mismatches rejected explicitly.
- [ ] Every supported distribution has explicit concentration semantics;
  resolved data has independent non-unit-volume regression evidence and
  fixed-V=1 PMF/PDF have independent sum/quadrature evidence.
- [ ] Needed helper inventory identifies reused versus added APIs and their
  units, shapes, validation, copy/view, mutation and empty/zero behavior.
- [ ] No new facade, physics-bearing container or duplicate authoritative state.
- [ ] Whole-container and coordinated replacement specification is approved
  and handed to M2; no aggregate implementation is claimed by M1.
- [ ] Helper rejection leaves protected input state unchanged; direct mutation
  is visible to subsequent derived reads and validation, not hidden by caches.
- [ ] Adjacent tests, retained transfer/export assertions, required lint/type
  checks, untargeted full-suite coverage and bounded strict docs checks pass.
- [ ] M2 start authorization records revision, evidence and reviewer approval;
  unresolved contract questions or required unavailable checks block handoff.

| Metric | Baseline at drafting | Target | Source |
|---|---|---|---|
| Approved alignment/units contract | E9 decisions still open | One reviewed complete contract | P1 decision table and tests |
| Scientific volume evidence | Existing coverage not inventoried | Resolved V=0.25/1/4 m³; PMF/PDF fixed V=1 m³ and invalid-volume rejection | P1 specification/P3 independent fixtures |
| Mixed gas species preservation | Separate legacy gas groups remain | All ordered species retained in data-native helper cases | P4 assertions |
| Necessary helpers with complete contracts/tests | Gap inventory pending | Every added or changed helper | P3/P4 inventory and PR tests |
| New duplicate state authorities | None authorized | Zero introduced | Diff and architecture review |
| Required validation | Not run during drafting | All required checks pass at implementation revision | Dated P5 evidence ledger |
| Coverage policy | Repository runner defaults | Unchanged full-package scope and normal threshold | Untargeted runner output |

Optional CUDA clean skips are not measured GPU success. Full E9 release
readiness, migrated physics and implemented aggregate replacement remain
downstream gates, not M1 completion claims.
