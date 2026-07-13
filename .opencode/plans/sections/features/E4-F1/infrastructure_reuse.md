# Infrastructure Reuse

- `condensation_step_gpu()` in `particula/gpu/kernels/condensation.py:387-559`
  remains the supported orchestration boundary; insert refresh after environment
  normalization and before its mass-transfer launch.
- `_ensure_environment_arrays()` in
  `particula/gpu/kernels/environment.py:260-388` already owns scalar broadcast,
  `(n_boxes,)` shape/device checks, and positive-finite validation. Reuse its
  normalized temperature rather than creating another owner.
- `WarpGasData.vapor_pressure` in `particula/gpu/warp_types.py:81-146` remains
  mutable GPU helper state and the refresh destination.
- `WarpEnvironmentData` in `particula/gpu/warp_types.py:149-169` remains the
  source of per-box temperature and pressure.
- `ConstantVaporPressureStrategy` in
  `particula/gas/vapor_pressure_strategies.py:189-246` defines constant-model
  semantics.
- `get_buck_vapor_pressure()` in
  `particula/gas/properties/vapor_pressure_module.py:121-172` is the exact
  piecewise CPU parity reference, including the freezing branch.
- Existing shape checks in `particula/gpu/kernels/condensation.py:322-348` and
  transfer behavior in `particula/gpu/conversion.py:265-340` should be extended,
  not duplicated.
- Follow lazy high-level export policy in `particula/gpu/kernels/__init__.py`;
  keep raw Warp device/kernel helpers internal unless a separate public API is
  approved.
- Extend `particula/gpu/kernels/tests/condensation_test.py` and its support
  module, following Warp CPU-required/CUDA-optional conventions.
