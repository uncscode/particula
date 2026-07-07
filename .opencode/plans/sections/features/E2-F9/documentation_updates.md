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
- `docs/index.md` was reviewed during the final handoff phase and intentionally
  left unchanged; the feature-doc changes above remained the primary discovery
  path for this track.

## Examples

- Added `docs/Examples/data_containers_and_gpu_foundations.py` as the published
  top-level runnable entrypoint required by issue #1223.
- Added `docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py`
  as the topic-directory implementation source of truth for the example logic.
- Added `docs/Examples/Data_Containers/index.md` as the rendered landing page
  with the canonical run command and GitHub source links.
- Updated `docs/Examples/index.md` with a `Data Containers` card so the example
  is discoverable from the examples gallery.
- No notebook artifact was added; the shipped example remains Python-only.

## Roadmap and Handoff Docs

- Updated `docs/Features/Roadmap/data-oriented-gpu.md` so the quick links,
  downstream handoff map, and relevant Epic B/C/D/E sections point readers to
  the shipped foundation guide and runnable example before deeper roadmap
  anchors.
- Updated `docs/Features/Roadmap/warp-autodiff-limitations.md` to link back to
  the shipped guide/example baseline as the starting point for differentiability
  follow-on work.
- Updated `docs/Features/Roadmap/index.md` to surface the shipped guide/example
  alongside the roadmap anchors they now support.
- Reviewed `docs/index.md` and general guides for follow-up discoverability
  changes, but intentionally left them unchanged because the roadmap-facing
  handoff was sufficient.
- Validation evidence for this docs-only phase belongs in PR notes rather than a
  committed repository artifact.
