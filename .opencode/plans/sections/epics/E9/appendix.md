# Appendix

## Authorities and technical references

Paths below are repository-relative. Line references describe the drafting
checkout and may move during migration.

| Reference | Relevance |
|---|---|
| Issue #1602, workflow `52e231a2` issue state | Agreed target contracts and six serial maintenance tracks |
| Prior `plan-scope-analyzer` message | Epic; feature/research tracks none; maintenance auto |
| `particula/aerosol.py:37–49` | Existing atmosphere/particle facade aggregate |
| `particula/particles/representation.py:566–581` | Concentration getter divides storage by volume |
| `particula/gas/gas_data.py:54–80` | CPU-owned ordered names, concentration units and partitioning mask |
| `particula/gas/environment_data.py:22–43` | Temperature, pressure and saturation-ratio authority |
| `particula/dynamics/condensation/condensation_strategies.py:294–364` | Existing native process-owned configuration |
| `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py:49–111` | Native single-box admission and raw concentration helper |
| `particula/dynamics/dilution.py:339–406` | Separate legacy gas groups and normalized particle access |
| `particula/runnable.py:108–137` | Retained composition and ordered sequence |
| `particula/execution/adapters/condensation.py`, `coagulation.py` | CPU adapter migration review points |
| `particula/gpu/conversion.py` | Protected CPU↔Warp transfer boundary, not obsolete facade machinery |
| `docs/Examples/data_containers_and_gpu_foundations.py` | Existing data-first example pattern from issue |
| `docs/Features/particle-data-migration/` | Coexistence guidance to migrate, preserving labeled history |
| `docs/Features/Roadmap/data-oriented-gpu.md` | Migration-before-Epic-I roadmap context |
| `.opencode/guides/testing_guide.md:70–95,234–255` | Focused assertions/full runner and adjacent wall-loss requirements |
| `.opencode/guides/documentation_guide.md:19–35,77–93` | Strict docs and source-first notebook sync/execute |
| `.opencode/guides/linting_guide.md:15–26` | Ruff and mypy completion checks |
| `.opencode/plans/templates/epic/` | Thirteen canonical section structures |

## Rejected approaches

- Mechanical replacement of facade getters by raw fields: units and volume
  normalization differ and require explicit scientific regression evidence.
- Deletion first or parallel tracks: violates migration/validation ordering.
- Replacement compatibility facade: recreates duplicated ownership/behavior.
- Dropping legacy-fixture scientific tests: destroys enduring behavior checks.
- Broad removal of all conversion helpers: breaks supported CPU↔Warp APIs.
- New GPU integration or performance promises: beyond this maintenance scope.

## Drafting limitations

Researcher delegation was attempted but blocked by the runtime subagent-depth
limit. The draft uses direct repository reads, issue references and guides;
full builder/example inventories remain owning-track work. The exposed tools
provide `apply_patch` rather than a full-file `write` tool; section files were
replaced, not appended. Epic `add-phase` is unsupported; six Not Started
milestones are the schema-valid program representation. `schema` unexpectedly
generated four schema files despite being exposed through the read wrapper;
the orchestrator should inspect any generated-file diff before committing.

Canonical returned paths are the thirteen relative E9 section paths with no
traversal, all accessed beneath the designated worktree. No shell/lstat tool
was available for an independent symlink audit. No implementation tests,
linters or examples were executed by this planning-only task.
