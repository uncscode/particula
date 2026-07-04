# E2-F9 Infrastructure Reuse

## Documentation Infrastructure

- `docs/Features/index.md` already hosts user-facing feature guide links and
  should link the foundation guide.
- `docs/Features/particle-data-migration.md` provides existing data-container
  explanation and conversion-helper style to reuse or cross-link.
- `docs/Features/Roadmap/data-oriented-gpu.md` is the canonical roadmap source
  for shapes, limitations, and downstream Epic handoff.
- `docs/Features/Roadmap/warp-autodiff-limitations.md` should be linked for
  autodiff/support-boundary caveats.
- `docs/Examples/index.md` is the examples gallery entry point.

## Runtime APIs to Document

- CPU containers:
  - `particula.particles.particle_data.ParticleData`
  - `particula.gas.gas_data.GasData`
- GPU schemas and helpers exported from `particula.gpu`:
  - `WARP_AVAILABLE`
  - `WarpParticleData`, `WarpGasData`
  - `to_warp_particle_data`, `to_warp_gas_data`
  - `from_warp_particle_data`, `from_warp_gas_data`
  - `gpu_context`

## Existing Patterns

- Optional Warp examples should guard imports or execution with
  `particula.gpu.WARP_AVAILABLE`.
- User examples in `docs/Examples/` should follow paired Jupytext notebook
  workflow when notebooks are added; edit `.py`, sync `.ipynb`, and execute.
- Public docs should import GPU helpers from `particula.gpu`, not private
  submodules.
- Keep transfers explicit; do not imply hidden CPU/GPU synchronization.
