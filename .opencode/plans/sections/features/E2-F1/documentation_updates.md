# Documentation Updates

## Primary Documentation Deliverables

- Shipped Issue #1183 as a docs-only inventory update in
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- Published the authoritative current-state schema table for `ParticleData`,
  `GasData`, `WarpParticleData`, and `WarpGasData` in that roadmap document.
- Captured current CPU↔GPU round-trip caveats there, including placeholder gas
  species names on restore and `WarpGasData.vapor_pressure` being dropped on
  return to CPU `GasData`.
- Shipped Issue #1184 as a docs-first ownership-decision update in
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- Added direct links to the new authoritative ownership section from
  `docs/Features/Roadmap/index.md` and
  `docs/Features/particle-data-migration.md`.
- Added a broader discoverability link from `docs/index.md` to the roadmap and
  migration guidance.
- Shipped Issue #1185 as a docs-first shape-conventions update in
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- Added a canonical `Canonical shape conventions for container workflows`
  subsection there with workflow rules, concrete shape examples, explicit field
  tables for `ParticleData`, `GasData`, future `EnvironmentData`,
  `WarpParticleData`, and `WarpGasData`, and a source-backed CPU execution
  boundary caution for condensation and coagulation.
- Updated `docs/Features/particle-data-migration.md` to link directly to the
  canonical shape-conventions anchor.
- Updated `docs/index.md` with one minimal discoverability reference to the
  canonical shape conventions while keeping the roadmap subsection as the
  single source of truth.
- Shipped Issue #1186 as the docs-only downstream handoff publication pass for
  E2-F1-P4.
- Published the final downstream handoff map in
  `docs/Features/Roadmap/data-oriented-gpu.md` with one concrete inheritance
  note each for `E2-F2` through `E2-F9`, plus the roadmap publication note that
  records `python3 .opencode/tools/build_mkdocs.py --validate-only --strict` as
  the validation command for the docs-only release.
- Updated `docs/Features/Roadmap/index.md`,
  `docs/Features/particle-data-migration.md`, and `docs/index.md` so roadmap,
  migration, and top-level feature entry points all link implementers directly
  to the canonical downstream handoff map instead of duplicating contract prose.

## Required Content

- Current schema inventory for `ParticleData`, `GasData`, `WarpParticleData`,
  and `WarpGasData`.
- Exact field shapes, dtypes, storage notes, validation/coercion hooks, and
  CPU↔GPU transfer behavior for the current public stored fields.
- Evidence references back to source files and existing tests for each row.
- Derived-property note separating `ParticleData` stored fields from computed
  accessors.
- Explicit notes for the current lossy gas round-trip behavior.
- An authoritative ownership-decision section covering field owners, shape
  expectations, mutability, and CPU↔GPU round-trip constraints.
- A canonical shape-conventions subsection under that roadmap decision record,
  rather than duplicated shape tables across multiple docs.
- Cross-links from roadmap, migration, and top-level docs entry points so
  implementers can find the ownership record without searching plan prose.
- A final downstream handoff map for sibling features `E2-F2` through `E2-F9`,
  published on the canonical roadmap page with direct entry-point links from
  roadmap, migration, and top-level documentation.
- A single roadmap publication note that states this phase shipped finalized
  P2/P3 contracts rather than new schema semantics and records the exact docs
  validation command used.

## Style and Maintenance

- Prefer concise tables with explicit shapes such as `(n_boxes, n_particles)`.
- Avoid vague language like "maybe per box" without assigning a downstream
  owner or open question.
- Link to code files only as implementation references; the docs should remain
  understandable without reading source.
