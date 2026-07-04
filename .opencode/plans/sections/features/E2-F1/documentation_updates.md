# Documentation Updates

## Primary Documentation Deliverables

- Create or update a schema decision record for particle, gas, and environment
  state ownership.
- Add a shape convention table covering CPU and GPU containers.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with links to the decision
  record and explicit E2-F1 outcomes.
- Update `docs/Features/particle-data-migration.md` or related feature docs with
  links to the canonical shape rules where appropriate.

## Required Content

- Current schema inventory for `ParticleData`, `GasData`, `WarpParticleData`, and
  `WarpGasData`.
- Ownership table for shared and per-box fields.
- Round-trip semantics for CPU/GPU conversions, including lossy or metadata-only
  fields.
- Shape rules for single-box, multi-box, particle-resolved, and binned
  workflows.
- Handoff notes for E2-F2 through E2-F9.

## Style and Maintenance

- Prefer concise tables with explicit shapes such as `(n_boxes, n_particles)`.
- Avoid vague language like "maybe per box" without assigning a downstream
  owner or open question.
- Link to code files only as implementation references; the docs should remain
  understandable without reading source.
