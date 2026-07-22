# Infrastructure Reuse

- `get_volume_dilution_coefficient()` and `get_dilution_rate()` in
  `particula/dynamics/dilution.py:9-96` define the existing equations and units.
- The E6-F1 CPU strategy/runnable and its deterministic fixtures are the required
  finite-step oracle; consume them after T1 ships rather than duplicating policy.
- `WarpParticleData` and `WarpGasData` in `particula/gpu/warp_types.py:24-184`
  provide authoritative fixed-shape fields and ownership boundaries.
- Explicit converters in `particula/gpu/conversion.py:113-179` belong at caller
  boundaries only; the new step must not invoke them internally.
- `condensation_step_gpu()` in
  `particula/gpu/kernels/condensation.py:1814-2180` provides entry-point,
  same-device, shape, dtype, and atomic-preflight patterns for particle/gas data.
- `_validate_device_match()` and `_validate_device_arrays()` in
  `particula/gpu/kernels/condensation.py:1643-1677` are patterns to reuse or
  factor only when doing so does not alter established APIs.
- `coagulation_step_gpu()` in
  `particula/gpu/kernels/coagulation.py:2104-2460` demonstrates scalar/per-box
  normalization, fixed-shape launches, caller-owned identity, and validation
  before output allocation or mutation.
- `particula/gpu/tests/cuda_availability.py:14-37` supplies the Warp CPU baseline
  and optional-CUDA parameterization convention.
- `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/coagulation_test.py` provide parity, no-op,
  device-mismatch, invalid-call atomicity, and import-contract test patterns.
- P1 intentionally leaves `dilution_step_gpu` concrete-module-only at
  `particula.gpu.kernels.dilution`; P2 owns any package re-export. Do not add a
  top-level `particula.gpu` runnable or implicit transfer API.
