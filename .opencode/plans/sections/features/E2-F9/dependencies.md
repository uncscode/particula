# E2-F9 Dependencies

## Required Internal Dependencies

- **E2-F2/T2:** CPU environment/container foundation decisions, especially any
  `EnvironmentData` naming and field ownership if implemented.
- **E2-F3/T3:** `WarpEnvironmentData`, GPU transfer-helper exports, and
  conversion-test behavior for environment state.
- **E2-F4/T4:** Gas/environment boundary and vapor-pressure ownership decisions.
- **E2-F5/T5:** Scalar-to-per-box kernel migration behavior and compatibility
  guarantees for environment inputs.
- **E2-F8/T8:** Current support-boundary docs/errors for multi-box and GPU
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
