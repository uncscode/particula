# Dependencies

**Parent:** E9. **Issue:** #1602. **Track:** T2. Metadata prerequisite: E9-M1.

`E9-M1 -> E9-M2 -> E9-M3 -> E9-M4 -> E9-M5 -> E9-M6`

Each arrow means **completed and validated**, not drafted, started, or merged
with tests pending. T1–T6 map one-to-one to M1–M6. No parallel implementation.
Internally P1 → P2 → P3 → P4 → P5 → P6 follows the same serial evidence gates.

| Related plan | Dependency / ownership |
|---|---|
| E9-M1 | Approves units, normalization, ordered mapping, API/copy-view/replacement specification and shared validators; supplies tested helpers |
| E9-M3 | Starts after M2 completion; consumes flat construction/replacement for scientific migration, owns scientific strategies |
| E9-M4 | Starts after M3; owns runnable composition, dilution/nucleation orchestration and CPU adapters |
| E9-M5 | Starts after M4; owns supported examples/notebooks and broad migration docs |
| E9-M6 | Starts after M5; owns temporary compatibility and legacy API removal, final release-readiness validation |

M1 is still a drafted plan in this planning context; reviewed/completed
contracts are implementation prerequisites, not assumed available now.
Unresolved or changed M1 contracts reopen the relevant entry gates; do not
invent alternate mapping/normalization authority inside Aerosol or its builder.

Existing native builders, distribution utilities and scientific tests supply
the baseline. CPU↔Warp transfer/export/ownership and prepared/resident identity
contracts are protected dependencies, not new implementation work. No new
external package, performance study or hardware prerequisite is introduced.
Epic I follows the migration and is not an upstream dependency.

M2 handoff includes final revision, approved native capability/API inventory,
replacement matrix evidence, migrated-test mapping, compatibility removal
ledger, required validation and reviewer acceptance. Only then may M3 start.
