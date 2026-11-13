# Scope

## Post-M5 single-PR scope — 2026-11-11

The maintainer reports M5 completed and requests M6 scoping as one PR with
multiple commits. GitHub confirms M5 PR #1634 merged at
`fb25078b6c0e8136ed06feeda59f0928b85c1b34`; the clean local `main` inspected
for this scope is that revision. Issue #1607 remains the tracking issue.
Its original per-phase issue-generation workflow is superseded for this
delivery by one implementation PR. No child issues are needed for this scope.

Implementation is in progress on `codex/e9-m6-legacy-removal`. CPU consumers,
legacy aggregate, particle/gas facades and untagged GPU compatibility paths
have been removed, with revision-specific evidence in `implementation.md` and
the assertion ledgers. Full package validation and the complete example output
review and both strict publication builds pass. PR delivery is tracked by
issue #1607; maintainer scientific and release approval remain separate. The serial review units are recorded
in `phase_details.md`; P1–P6 remain acceptance milestones, not separate PRs.

The post-M5 audit established these removal targets (consult the implementation
record for completed items):

- `particula/_legacy_aerosol.py`, including its aggregate and builder.
- `LegacyNucleation`, legacy dilution preflight and aggregate dispatch in
  `dynamics/particle_process.py` and `dynamics/dilution.py`.
- Native/legacy type unions in `runnable.py`, `equilibria/equilibria.py` and
  CPU condensation/coagulation adapters; facade-unwrapping, strategy fallback
  and cached-state synchronization in scientific consumers.
- `from_representation` / `to_representation` in `particles/particle_data.py`
  and `from_species` / `to_species` in `gas/gas_data.py`, plus their exports.
- Native test guards that currently import and monkeypatch deleted classes:
  replace their mechanism while preserving real native execution assertions.

The accepted implementation removes untagged direct/prepared GPU admission
and schema-v5 `legacy_untagged` checkpoint admission, as explicitly assigned
by the M4 handoff. Keep tagged v5 payloads and reject untagged records before
setup/upload; do not infer tags or change schema version. Explicit CPU-Warp
transfer APIs remain, but require supported explicit population metadata.
Incomplete CPU material scratch storage is not executable population state.
Independent shared-mask conversion and numerical utilities remain supported.

The maintainer additionally requires the full publication build to pass.
Repair the production-docstring backlog and rendering warnings without
suppressing warnings, excluding supported APIs, or changing scientific code.

Native resolved kernel binning already exists in
`dynamics/coagulation/coagulation_strategy/_native_step.py` (`kernel_edges`
and `binned_kernel_data`). Trace the old `change_particle_representation.py`
assertions to these native capabilities before removing obsolete wrappers;
do not discard scientific binning or introduce a replacement facade.

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
  root/developer snippets and API navigation. Readiness record:
  `docs/Features/Roadmap/data-native-v030-closeout.md`.

## Interfaces

Remove ParticleRepresentation, GasSpecies, Atmosphere, obsolete constructors,
facade bridges, temporary aliases/branches and facade-only messages. Retain
flat Aerosol, ParticleData, GasData, EnvironmentData, native initialization,
whole-container replacement and runnable composition established by M1–M4.
Preserve `particula/gpu/conversion.py`, GPU exports and concrete-only resident
contracts. Old names may appear in negative boundary tests and explicitly
labeled non-executable historical/migration material, never as supported APIs.

## Out of scope

Validate upstream approved distribution metadata and normalization corrections;
M6 does not defer their implementation until removal. No new physics, further
precision/schema changes, general multi-box CPU execution,
permanent substitute facade, new public GPU API, hidden transfer/synchronization
or fallback, automatic prepared-state rebind, resident/graph redesign, new
performance claims, arbitrary coverage targets, Epic I or release publication.
Do not remove unrelated warnings, functional science, useful native builders
or CPU-Warp conversion helpers based on name matching alone. The original scoping
update edited only E9-M6 metadata; implementation now follows the accepted
serial review units. A version
bump, release tag and publication are separate from this implementation scope.
