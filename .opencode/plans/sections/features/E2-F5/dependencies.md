# Dependencies

## Upstream Dependencies

- **E2-F2 / T2:** Provides or defines environment containers, including expected
  fields such as temperature and pressure, `n_boxes` shape conventions, and any
  CPU/GPU conversion helpers.
- **E2-F3 / T3:** Establishes `WarpEnvironmentData` and the explicit CPU/GPU
  transfer conventions that T5 should reuse for environment inputs.
- **E2-F4 / T4:** Vapor-pressure boundary decisions may influence whether
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
