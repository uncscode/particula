# Infrastructure Reuse

## Existing GPU Kernel APIs

- `particula/gpu/kernels/condensation.py` exposes `condensation_step_gpu` with
  scalar `temperature` and `pressure`. The launch path computes scalar dynamic
  viscosity and mean free path before launching kernels.
- `particula/gpu/kernels/coagulation.py` exposes `coagulation_step_gpu` with
  scalar `temperature` and `pressure`. The Brownian coagulation kernel computes
  dynamic viscosity and mean free path from those scalar values.
- `particula/gpu/kernels/__init__.py` exports the public low-level kernel APIs;
  export changes must remain backward compatible.

## Validation Patterns to Reuse

- Condensation has strict shape/device validation helpers for particle and gas
  containers. Extend the same style for environment arrays.
- Coagulation already has `_ensure_volume_array(volume, n_boxes, device)`, which
  accepts either a scalar or a `(n_boxes,)` Warp array. Use this as the model for
  scalar-to-per-box temperature/pressure normalization.
- GPU tests already include bad-shape and wrong-device cases; mirror those test
  structures for environment state.

## Container and Conversion Patterns

- `particula/gpu/warp_types.py` contains `WarpParticleData` and `WarpGasData`
  shape conventions using `n_boxes` as the leading dimension. E2-F2 is expected
  to add or define `WarpEnvironmentData` beside these containers.
- `particula/gpu/conversion.py` owns explicit CPU/GPU transfer helpers. If this
  feature needs conversion functions, add them there rather than hiding device
  transfers inside kernels.
- `particula/gas/gas_data.py` intentionally lacks environment fields; do not add
  temperature/pressure to gas data as part of this track.

## Testing Infrastructure

- Use `particula/gpu/tests/cuda_availability.py` to run Warp CPU tests and CUDA
  tests when available.
- Preserve scalar compatibility coverage in existing condensation and
  coagulation kernel tests, then add per-box and mismatch cases next to the
  affected APIs.
