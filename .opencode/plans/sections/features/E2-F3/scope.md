# Scope

## In scope

- Add `WarpEnvironmentData` to `particula/gpu/warp_types.py`.
- Mirror the `E2-F2` CPU `EnvironmentData` schema exactly with
  `temperature` and `pressure` as `(n_boxes,)` `wp.array(dtype=wp.float64)`
  fields and `saturation_ratio` as `(n_boxes, n_species)`
  `wp.array2d(dtype=wp.float64)`.
- Update the module/class docstrings in `particula/gpu/warp_types.py` so the
  environment container and its shape semantics are explicit.
- Add focused Warp CPU tests in `particula/gpu/tests/warp_types_test.py` for
  struct creation, field presence, one-box and multi-box shapes, `float64`
  dtypes, and deterministic NumPy-backed round-trip values.
- Add `to_warp_environment_data(data, device="cuda", copy=True)` in
  `particula/gpu/conversion.py` using the shared Warp-availability and
  device-validation helpers.
- Add focused conversion coverage in `particula/gpu/tests/conversion_test.py`
  for values, shapes, dtypes, invalid-device behavior, Warp-unavailable
  behavior, and CPU copy semantics.

## Out of scope

- Do not redesign the CPU `EnvironmentData` schema except for small alignment
  fixes needed to make GPU transfer unambiguous.
- Do not add Warp-to-CPU environment conversion helpers in this shipped slice.
- Do not update `particula/gpu/__init__.py` exports in this shipped slice.
- Do not migrate existing condensation or coagulation kernels from scalar
  temperature/pressure arguments in this track.
- Do not introduce implicit transfers inside runnable objects or kernel launch
  wrappers.
- Do not store string metadata or Python-only objects in `WarpEnvironmentData`.
- Do not change precision defaults; use `float64` unless a later precision
  decision changes repository policy.
- Do not add optional CUDA-parametrized transfer coverage or broader user docs
  until later phases land the transfer boundary.

## Done signal

`WarpEnvironmentData` exists in `particula/gpu/warp_types.py`,
`to_warp_environment_data` exists in `particula/gpu/conversion.py`, field
shapes and dtypes match the CPU schema, targeted Warp CPU tests pass in
`particula/gpu/tests/warp_types_test.py` and
`particula/gpu/tests/conversion_test.py`, and no reverse helper or export
changes are introduced prematurely.
