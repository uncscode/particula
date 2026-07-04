# Success Criteria

## Functional criteria

- `WarpEnvironmentData` exists and mirrors the CPU `EnvironmentData` numeric
  schema from `E2-F2`.
- `WarpEnvironmentData` keeps `temperature` and `pressure` as `(n_boxes,)`
  arrays and `saturation_ratio` as an `(n_boxes, n_species)` array.
- Warp CPU schema tests preserve `temperature`, `pressure`, and
  `saturation_ratio` values on deterministic NumPy-backed round trips without
  dropping fields or reshaping them implicitly.
- `to_warp_environment_data(data, device="cuda", copy=True)` exists and moves
  `temperature`, `pressure`, and `saturation_ratio` into
  `WarpEnvironmentData` with explicit `wp.float64` transfers.
- No Warp-to-CPU helper or package export changes are introduced before later
  phases define the remaining transfer boundary.
- Existing particle and gas conversion behavior is unchanged.

## Design criteria

- Environment transfer remains explicit rather than implicit; no kernel,
  runnable, or strategy object performs an environment CPU/GPU transfer unless
  callers invoke the helper directly.
- `temperature` and `pressure` remain `(n_boxes,)`, while `saturation_ratio`
  remains `(n_boxes, n_species)`, including single-box simulations.
- The struct declaration uses explicit field names and `wp.float64` array types
  and the shipped CPU-to-Warp helper preserves those explicit fields so later
  phases can build on a stable schema.
- Optional dependency behavior for Warp remains unchanged.

## Verification criteria

- Targeted GPU tests pass.
- `particula/gpu/tests/warp_types_test.py` asserts field presence, shapes,
  `float64` dtypes, and deterministic values for one-box and multi-box inputs.
- `particula/gpu/tests/conversion_test.py` asserts helper values, shapes,
  `float64` dtypes, invalid-device failures, Warp-unavailable behavior, and CPU
  copy semantics.
- Linting passes for changed GPU modules.
- Documentation is limited to updated module/class docstrings plus the new
  helper docstring.
- The implementation does not block downstream kernel migration tracks.
