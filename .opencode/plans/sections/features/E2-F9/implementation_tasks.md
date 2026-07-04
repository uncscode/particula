# E2-F9 Implementation Tasks

## E2-F9-P1: Foundation Guide

- Create `docs/Features/data-containers-and-gpu-foundations.md` as the canonical
  guide unless the open question explicitly resolves in favor of expanding
  `particle-data-migration.md` instead.
- Add named sections for container schemas, shape conventions, transfer helper
  APIs, current limitations, and roadmap handoff so later docs can deep-link to
  anchors instead of prose-searching.
- Cross-link `docs/Features/particle-data-migration.md` rather than duplicating
  all migration content.
- Add a support-boundary table covering single-box CPU condensation, scalar
  temperature/pressure GPU kernels, `WarpGasData` schema drift, fixed-shape
  graph capture, and absent/planned environment state.
- Update `docs/Features/index.md` with a discoverable card/link that points to
  the final guide path.

## E2-F9-P2: Examples

- Add `docs/Examples/data_containers_and_gpu_foundations.py` as the primary
  example file and keep the example small enough to execute in the default dev
  environment.
- In that example, construct or convert documented `ParticleData` and `GasData`
  objects using public APIs.
- Add a minimal GPU transfer section with:
  - `from particula.gpu import WARP_AVAILABLE` guard.
  - `to_warp_particle_data` and `from_warp_particle_data`.
  - `to_warp_gas_data(..., vapor_pressure=...)` and
    `from_warp_gas_data(..., name=...)`.
  - A note that CUDA is optional and Warp CPU/fallback behavior depends on the
    installed environment.
- Link the example from `docs/Examples/index.md` and the new foundation guide.
- If notebooks are used, pair them with the same base filename and follow the
  repository's `.py`/`.ipynb` sync and execution workflow.

## E2-F9-P3: Handoff and Validation

- Update roadmap sections to point users to the new foundation guide.
- Add explicit downstream dependency notes for future Epic B/C/D/E planning.
- Run the repository's docs link checks or mkdocs build when available, and note
  the exact command used in the PR summary.
- Smoke-run `docs/Examples/data_containers_and_gpu_foundations.py` in the
  repository's default development environment.
- Record any optional-Warp validation limitations in the final PR notes.
