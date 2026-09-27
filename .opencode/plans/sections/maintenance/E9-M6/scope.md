# Scope

## Modules and directories

- Audit `particula/`, `docs/`, `readme.md`, `AGENTS.md`, `mkdocs.yml` and active
  developer guidance for imports, aliases, annotations, builders/factories,
  dispatch branches, strings, generated API references and runnable snippets.
- Removal candidates: `particula/particles/representation.py`,
  `representation_builders.py`, `representation_factories.py`, obsolete parts
  of `change_particle_representation.py`, and their exports/tests.
- Removal candidates: `particula/gas/species.py`, `species_builders.py`,
  `species_factories.py`, `atmosphere.py`, `atmosphere_builders.py`, and their
  exports/tests. Delete only capabilities proven obsolete by M2's inventory;
  native initialization and scientific utilities must remain available.
- Clean residual facade-only machinery in `particula/aerosol.py`,
  `aerosol_builder.py`, `runnable.py`, `dynamics/particle_process.py`,
  `dynamics/dilution.py`, migrated condensation/coagulation/wall-loss modules,
  and `execution/adapters/{condensation,coagulation}.py` as the audit requires.
  Remove imports/exports from relevant `__init__.py` files atomically with
  their definitions, not in a later broken intermediate commit.
- Classify tests under `particula/{particles,gas}/tests/`, dynamics adjacent
  suites, `particula/tests/`, `integration_tests/`, `gpu/` and `execution/`.
  Delete facade-only assertions; preserve migrated scientific assertions.
- Final guidance: `docs/Features/particle-data-migration/`,
  `docs/Features/data-containers-and-gpu-foundations.md`,
  `docs/Features/Roadmap/data-oriented-gpu.md`, supported `docs/Examples/`,
  root/developer snippets and API navigation. Proposed readiness record:
  `docs/Features/Roadmap/data-native-v030-closeout.md` (not yet created).

## Interfaces

Remove ParticleRepresentation, GasSpecies, Atmosphere, obsolete constructors,
facade bridges, temporary aliases/branches and facade-only messages. Retain
flat Aerosol, ParticleData, GasData, EnvironmentData, native initialization,
whole-container replacement and runnable composition established by M1–M4.
Preserve `particula/gpu/conversion.py`, GPU exports and concrete-only resident
contracts. Old names may appear in negative boundary tests and explicitly
labeled non-executable historical/migration material, never as supported APIs.

## Out of scope

No new physics, precision/schema changes, general multi-box CPU execution,
permanent substitute facade, new public GPU API, hidden transfer/synchronization
or fallback, automatic prepared-state rebind, resident/graph redesign, new
performance claims, arbitrary coverage targets, Epic I or release publication.
Do not remove unrelated warnings, functional science, useful native builders
or CPU-Warp conversion helpers based on name matching alone. This drafting
task edits only E9-M6 metadata and its twelve section files.
