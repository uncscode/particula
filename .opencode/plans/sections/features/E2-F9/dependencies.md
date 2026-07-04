# E2-F9 Dependencies

## Required Internal Dependencies

- **E2-F2:** CPU environment/container foundation decisions, especially any
  `EnvironmentData` naming and field ownership if implemented.
- **E2-F3:** `WarpEnvironmentData`, GPU transfer-helper exports, and
  conversion-test behavior for environment state.
- **E2-F4:** Gas/environment boundary and vapor-pressure ownership decisions.
- **E2-F5:** Scalar-to-per-box kernel migration behavior and compatibility
  guarantees for environment inputs.
- **E2-F8:** Current support-boundary docs/errors for multi-box and GPU
  limitations.

## Existing Files and APIs

- `particula/particles/particle_data.py`
- `particula/gas/gas_data.py`
- `particula/gpu/warp_types.py`
- `particula/gpu/conversion.py`
- `particula/gpu/__init__.py`
- `particula/gpu/kernels/condensation.py`
- `particula/dynamics/condensation/condensation_strategies.py`

## External Dependencies

- Optional `warp-lang` for GPU transfer examples.
- CUDA-capable devices are optional for documentation examples; examples must be
  safe when Warp or CUDA is unavailable.
- Existing docs stack, including mkdocs and any notebook/Jupytext validation
  tooling used by `docs/Examples/`.

## Downstream Dependencies

- Future GPU-resident simulation work needs stable schemas and explicit transfer
  caveats from this guide.
- Graph-capture roadmap work depends on fixed-shape/preallocation conventions.
- Autodiff and precision roadmap work depends on limitations being discoverable.

## Sequencing Notes

- `E2-F9-P1` should wait for the shipped contracts from `E2-F2` through `E2-F5`
  and `E2-F8` so the foundation guide documents implemented schemas,
  environment-input rules, and support boundaries rather than drafts.
- `E2-F9-P2` should follow `P1` because examples must teach the same helper
  names, import paths, and limitation wording already published in the guide.
- `E2-F9-P3` should run last and should incorporate `E2-F6`/`E2-F7` roadmap
  evidence only after those features publish accepted precision and condensation
  guidance.
- This feature is downstream-only: it should not introduce new container or
  transfer semantics that would force reverse edits into `E2-F2` through `E2-F5`.
