# Success Criteria

Implementation and required validation are complete; evidence is in
[implementation.md](implementation.md). Maintainer acceptance and M5
authorization remain pending.

- [x] M3 completed-and-validated revision and upstream contracts are accepted.
- [x] All supported CPU runnable rate/execute paths use native containers;
  composed calls preserve ordering, returned aggregate identity and state.
- [x] Multi-box and invalid directly mutated state reject without unauthorized
  writes; ordered alignment and non-unit-volume results match M1/M3 contracts.
- [x] Dilution decays all gas lanes and particle concentrations once per
  substep, preserves protected fields, and retains rejection/commit-failure
  behavior, including custom-strategy delegation boundaries.
- [x] Nucleation reads current gas/environment, preserves independent weighted
  conservation and commits atomically per attempted substep only.
- [x] CPU adapter paths retain supported profile gates, exactly-once dispatch,
  payload/result identity, error propagation and concrete-only export surface.
- [x] Enduring behavioral assertions are migrated, not discarded; each
  temporary compatibility seam has an M6 removal owner and remaining consumers.
- [x] Focused assertions, explicit CPU wall-loss checks, retained GPU boundary
  checks, untargeted coverage, and required lint/type/docs checks pass.
- [ ] Maintainer review accepts the dated final-revision evidence and
  authorizes M5 to start.

| Metric | Baseline at drafting | Target | Source |
|---|---|---|---|
| Native runnable families | Facade access remains | All five listed families | P1–P3 tests and audit |
| Diluted gas categories | Two legacy groups | Every unified gas lane | P2 mixed-partitioning fixtures |
| Preserved failure contracts | Existing legacy fixtures | Every enduring assertion mapped | P2/P3 failure ledger |
| Native CPU adapter families | Existing carriers use Aerosol | Existing condensation and coagulation profiles | P4 adapter/integration tests |
| Hidden transfers/rebinds/new exports | None authorized | Zero introduced | Boundary tests and diff review |
| Required validation | Not run during drafting | All required checks pass | P5 command ledger |
| Coverage policy | Repository defaults | Unchanged full-package scope/normal threshold | Untargeted runner output |

Unavailable mandatory checks block completion. Optional CUDA clean skips are
not measured GPU success. M4 completion is not E9 release/deletion readiness.
