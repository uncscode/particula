# Dependencies

## Upstream Dependencies

- **E2-F2:** Provides or defines environment containers, including expected
  fields such as temperature and pressure, `n_boxes` shape conventions, and any
  CPU/GPU conversion helpers.
- **E2-F3:** Establishes `WarpEnvironmentData` and the explicit CPU/GPU
  transfer conventions that E2-F5 should reuse for environment inputs.
- **E2-F4:** Vapor-pressure boundary decisions may influence whether
  condensation receives precomputed per-box vapor-pressure state or recomputes
  temperature-dependent terms later.

## Internal Code Dependencies

- `particula/gpu/kernels/condensation.py` and its tests.
- `particula/gpu/kernels/coagulation.py` and its tests.
- `particula/gpu/warp_types.py` for Warp container conventions.
- `particula/gpu/conversion.py` for explicit container transfer patterns.
- `particula/gpu/tests/cuda_availability.py` for CPU/CUDA test parametrization.

## External Dependencies

- Warp runtime availability, including CPU fallback and optional CUDA execution.
- NumPy and existing scientific utility functions used to compute viscosity,
  mean free path, and kernel physics.

## Downstream Dependencies

- Later E2 tracks and physics kernels can consume the environment normalization
  path rather than defining separate scalar/array migration logic.
- Documentation and examples must explain that scalar compatibility is a bridge,
  not the long-term multi-box environment model.

## Sequencing Notes

- `E2-F5-P1` can start once E2-F2 publishes the accepted environment field list,
  but it should treat E2-F3 helper names as provisional until `E2-F3-P2` lands.
- `E2-F5-P2` should wait for `E2-F3-P2` so normalization logic reuses the same
  transfer vocabulary and device-validation path instead of creating a parallel
  contract.
- `E2-F5-P3` should follow `P2` and should incorporate the vapor-pressure
  boundary documented by `E2-F4-P3` before condensation inputs are treated as
  final.
- `E2-F5-P4` should follow `P3` so coagulation adopts the tested normalization
  path after condensation proves the shared helper surface.
- E2-F7 and E2-F9 should consume the `P3`/`P4` environment-input contract rather
  than redefining scalar compatibility rules, which keeps the cross-feature DAG
  one-directional.
