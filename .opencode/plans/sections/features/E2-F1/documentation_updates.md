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
- Cross-links from roadmap, migration, and top-level docs entry points so
  implementers can find the ownership record without searching plan prose.

## Style and Maintenance

- Prefer concise tables with explicit shapes such as `(n_boxes, n_particles)`.
- Avoid vague language like "maybe per box" without assigning a downstream
  owner or open question.
- Link to code files only as implementation references; the docs should remain
  understandable without reading source.
