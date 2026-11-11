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
| 2. Native scientific runnables and CPU slot retirement | Implemented; focused checks pass |
| 3. CPU/direct/prepared GPU dilution | Implemented; focused checks pass |
| 4. CPU nucleation, mapping and exhaustion transactions | Implemented; focused checks pass |
| 5. Direct/prepared GPU nucleation and exhaustion | Implemented; focused checks pass |
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

Focused checkpoint evidence follows. Final acceptance remains pending.

### Checkpoint 2

- `.venv/bin/pytest particula/dynamics/tests/native_runnable_test.py particula/tests/runnable_test.py particula/dynamics/tests/wall_loss_runnable_test.py particula/dynamics/tests/native_wall_loss_test.py particula/integration_tests/native_scientific_processes_test.py -q --no-cov`: 474 passed.
- Both CPU wall-loss suites explicitly: 16 passed and 43 passed.
- Changed-file Ruff check and format: passed.
- Initial full mypy inspection exposes downstream annotation migration work
  in legacy custom fixtures, dilution, nucleation, Equilibria and adapters.
  This is pending M4 work, not a passing comprehensive gate.
- Native successor assertions live in `native_runnable_test.py`; legacy
  runnable fixtures remain for transition coverage. M3 removed-slot charge
  assertions now check zero for removed slots, unchanged survivors, and the
  independently computed charge sink. Read-only charge rejects before draws.

### Checkpoint 3

- CPU native/legacy dilution, runnable, exports and existing direct-GPU suite:
  335 passed, 9 skipped.
- Native direct/prepared GPU dilution and population-boundary suite with
  `WARP_CACHE_PATH=/tmp/particula-m4-warp`: 58 passed, 22 skipped
  (optional CUDA). The first native run failed on the read-only default Warp
  cache; using the explicit writable cache resolved those failures.
- Changed-file Ruff check, format and whitespace checks: passed.
- Native CPU dilution uses equal substeps and restores an attempted substep
  after a write failure. Legacy exact total-duration execution is preserved.
- Native GPU preflight validates population metadata, physical state and
  writer aliases. Prepared repeated enqueue retains its allocation/transfer/
  synchronization-free boundary. Native positive tests replace dilution's
  temporary rejection row in the population-boundary inventory.

### Checkpoint 4

- CPU source finalization/commit, configuration/builders, native and legacy
  runnables, integration, and exhaustion: 381 passed.
- Native CPU nucleation example: 4 passed with `-Werror`; this focused usage
  example was migrated to native containers, non-unit volume and unified inert
  gas. Its import/behavior assertions now verify that native successor.
- Changed-file Ruff/format, focused source mypy, whitespace checks: passed.
- Native events convert density to counts at commit; explicit gas order/pairs
  support unequal widths and particle-only lanes. Native representative volume
  reduction preserves gas concentration and old particle mass density.
- Final diagnostics declare event units. CPU partial writes restore the
  attempted transaction; tests retain earlier successful substeps.
- `LegacyNucleation` is concrete-only in `particle_process.py`; only the
  transitional runnable regression module imports it. Public `Nucleation` has
  two constructor arguments and always uses the current aggregate environment.

### Checkpoint 5

- Native/legacy direct and prepared nucleation/exhaustion plus boundary guards:
  221 passed, 20 skipped (optional CUDA).
- Direct nucleation and complete-process source examples: 26 passed with
  `-Werror`. All Warp runs use `/tmp/particula-m4-warp` for cache storage.
- Changed-file Ruff/format, focused source mypy and whitespace checks: passed.
- Native GPU injection uses explicit uploaded mapping configuration and separate
  gas/particle widths. Existing integer event admission is retained: one
  unit-count slot per event. CPU weighted packing is not silently substituted.
- Native exhaustion scales count demand with physical volume once; it does not
  scale gas concentration. Tests cover resampling/source composition, independent
  density closure, repeated prepared execution, stale bindings, protected gas
  metadata aliasing, and write-free invalid live-state rejection.
- Old concrete `_execute_*` exhaustion and `_preflight_nucleation` legacy
  phase helpers remain fail-closed for tagged direct calls; migrated public and
  prepared boundaries have positive native successor tests.
