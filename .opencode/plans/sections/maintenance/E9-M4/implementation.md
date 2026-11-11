# M4 single-PR implementation record

## Accepted scope and decisions (2026-11-11)

The maintainer accepted M3 as complete and requested implementation, usage
validation, and one PR for all M4 work. Base: `da2ad5e50`, merged upstream
PR #1632. Branch: `codex/e9-m4-native-integration`; target: upstream `main`;
issue: #1606. This supersedes the old multiple-PR and issue-generation wording.

- PMF stores bin number density; radius PDF stores dn/dr. Both require V=1.
  Resolved populations store represented counts at positive physical volume.
- Native CPU resolved wall loss will clear charge when it removes mass/count,
  making the slot reusable. This explicitly amends M3's retained-charge rule;
  the removal law itself does not change.
- Checkpoint schema v5 will carry population semantics. Old formats reject;
  no checkpoint migration utility or inferred meaning is authorized.
- Native Nucleation reads current aggregate environment, without a constructor
  environment argument. Species order and gas-to-particle mapping are explicit.
- M3 fixed-composition density coagulation and its documented limitations stay.
  Resolved coagulation still requires equal positive active weights.
- M5 owns broad examples/docs; M6 owns final legacy API deletion and release.
  Temporary compatibility must be isolated and inventoried, never constructed
  inside native execution. No hidden transfers, fallback, or automatic rebind.

## Serial commit checkpoints

| Work | State |
|---|---|
| 1. Accepted contracts and delivery record | Recorded |
| 2. Native scientific runnables and CPU slot retirement | Pending |
| 3. CPU/direct/prepared GPU dilution | Pending |
| 4. CPU nucleation, mapping and exhaustion transactions | Pending |
| 5. Direct/prepared GPU nucleation and exhaustion | Pending |
| 6. CPU adapters and remaining orchestration consumers | Pending |
| 7. Communication and physical volume evolution | Pending |
| 8. Native resident/prepared metadata and diagnostics | Pending |
| 9. Schema-v5 checkpoints and exact-device continuation | Pending |
| 10. Complete native composed-process regressions | Pending |
| 11. Final validation and developer handoff | Pending |

Each functional checkpoint includes adjacent tests and literal evidence before
dependent work. The final PR includes all checkpoints; no partial checkpoint
constitutes M4 completion. Required final evidence: untargeted repository
coverage, separate examples/infrastructure, explicit both CPU wall-loss suites,
GPU transfer/export/ownership, resident/full-loop/RNG/restart checks, repository
lint/format/mypy and strict docs. Warp CPU is required when installed; CUDA is
optional pass-or-clean-skip. Thresholds and scientific tolerances stay intact.

## Consumer, assertion and compatibility ledger

| Consumer | Native authority / enduring behavior | Temporary compatibility |
|---|---|---|
| Scientific runnables | Held particles, unified gas, environment; ordered equal substeps, strategy/container identity | Existing explicit legacy aggregate callers until M5/M6 |
| Dilution | Raw population and every gas lane; exponential decay, preflight, rollback and exception propagation | Existing legacy dilution fixtures/consumers until M6 |
| Nucleation | Current gas/environment, explicit mapping; count/volume conversion, per-attempt transaction | Isolated legacy signature only for inventoried remaining callers |
| CPU adapters and Equilibria | Native aggregate recognition; exact dispatch/result and existing solver contracts | Remaining legacy examples until M5/M6 |
| GPU direct/prepared processes | Tagged population and process-owned configuration; caller-owned arrays and explicit transfer | Untagged legacy branches until M6 |
| Resident execution/diagnostics | Frozen semantic metadata, separate species widths, pinned resources, RNG ownership | No automatic reinterpretation or rebinding |
| Checkpoint | v5 canonical bytes and explicit semantics; fresh exact-device restore | Old formats rejected, no migration utility |

## Validation evidence

Implementation checks have not yet run. Append exact command outcomes and
revision references here as work completes. Final acceptance remains pending.
