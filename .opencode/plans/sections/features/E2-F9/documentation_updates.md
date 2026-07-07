# E2-F9 Documentation Updates

## New or Updated User Docs

- Added `docs/Features/data-containers-and-gpu-foundations.md` as the canonical
  reference for:
  - `ParticleData`, `GasData`, `EnvironmentData`, `WarpGasData`, and
    `WarpEnvironmentData` schemas.
  - leading-axis shape conventions for single-box and multi-box workflows.
  - explicit CPU↔GPU transfer helpers in `particula.gpu`.
  - current shipped CPU/GPU support boundaries and intentionally lossy gas
    restore behavior.
- Updated `docs/Features/index.md` with a new
  `Data Containers and GPU Foundations` card placed ahead of the migration
  guide, and revised the migration card copy so it reads as migration-focused.
- Updated `docs/Features/particle-data-migration.md` to add early cross-links
  back to the canonical foundation guide and to keep migration-specific prose
  separate from the authoritative contract page.
- Supporting docs discovery updates may also be present in `docs/index.md` from
  the docs subagent; those are secondary to the feature-doc changes above.

## Examples

- No `docs/Examples/` artifacts were added in issue #1222; runnable examples
  remain deferred to E2-F9-P2.

## Roadmap and Handoff Docs

- The new foundation guide links readers to
  `docs/Features/Roadmap/data-oriented-gpu.md` for future work and planned
  expansions.
- No standalone roadmap-page edits are required to reflect the documentation
  delivered in issue #1222.
