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

- Added `docs/Examples/data_containers_and_gpu_foundations.py` as the published
  wrapper entrypoint while keeping the example logic in
  `docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py`.
- The shipped example constructs documented single-box `ParticleData` and
  `GasData` objects using public APIs and prints concise shape-oriented output.
- The shipped GPU transfer section uses:
  - `from particula.gpu import WARP_AVAILABLE` guard.
  - `to_warp_particle_data` and `from_warp_particle_data`.
  - `to_warp_gas_data(..., vapor_pressure=...)` and
    `from_warp_gas_data(..., name=...)`.
  - A clear note that Warp-backed transfers are optional and use the Warp CPU
    backend (`device="cpu"`) when available.
- Linked the example from `docs/Examples/index.md` and published the rendered
  landing page at `docs/Examples/Data_Containers/index.md`.
- No notebook was used for the shipped implementation.

## E2-F9-P3: Handoff and Validation

- Update roadmap sections to point users to the new foundation guide.
- Add explicit downstream dependency notes for future Epic B/C/D/E planning.
- Run the repository's docs link checks or mkdocs build when available, and note
  the exact command used in the PR summary.
- Smoke-run `docs/Examples/data_containers_and_gpu_foundations.py` in the
  repository's default development environment.
- Record any optional-Warp validation limitations in the final PR notes.
- Keep `particula/gpu/tests/data_containers_example_test.py` aligned with the
  published entrypoint and example output contract.
