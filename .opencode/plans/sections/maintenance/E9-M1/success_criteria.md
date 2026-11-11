# Success Criteria

Implementation checks pass at `0db7e42b2`; final maintainer handoff approval
remains pending. See `validation_evidence.md` for commands and provenance.

- [x] Approved ordered mapping covers particle/density, all gas categories,
  environment and process parameters, with mismatches rejected explicitly.
- [x] Every supported distribution has explicit concentration semantics;
  resolved data has independent non-unit-volume regression evidence and
  fixed-V=1 PMF/PDF have independent sum/quadrature evidence.
- [x] Needed helper inventory identifies reused versus added APIs and their
  units, shapes, validation, copy/view, mutation and empty/zero behavior.
- [x] No new facade, physics-bearing container or duplicate authoritative state.
- [x] Whole-container and coordinated replacement specification is approved
  and handed to M2; no aggregate implementation is claimed by M1.
- [x] Helper rejection leaves protected input state unchanged; direct mutation
  is visible to subsequent derived reads and validation, not hidden by caches.
- [x] Adjacent tests, retained transfer/export assertions, required lint/type
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

## Current completion boundary (2026-11-10)

The delivered-helper and integration inventory is reconciled in `appendix.md`.
Checklist completion is contingent on `validation_evidence.md` and final
maintainer approval. Historical drafting baselines in the table above remain
historical; they do not override the recorded P1/P2 decisions.
