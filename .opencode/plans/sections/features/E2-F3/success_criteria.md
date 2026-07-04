# Success Criteria

## Functional criteria

- `WarpEnvironmentData` exists and mirrors the CPU `EnvironmentData` numeric
  schema from `E2-F2`.
- CPU-to-Warp and Warp-to-CPU helpers are public through `particula.gpu` when
  Warp is available.
- Environment round trips preserve `temperature`, `pressure`, and
  `saturation_ratio` on the Warp CPU backend without dropping fields or
  reshaping them implicitly.
- CUDA-parametrized round-trip coverage runs automatically when CUDA is
  available and skips cleanly otherwise.
- Existing particle and gas conversion behavior is unchanged.

## Design criteria

- Transfers are explicit and centralized in conversion helpers.
- No kernel, runnable, or strategy object performs an implicit CPU/GPU transfer.
- `temperature` and `pressure` remain `(n_boxes,)`, while `saturation_ratio`
  remains `(n_boxes, n_species)`, including single-box simulations.
- `to_warp_environment_data(..., device="cuda")` is the documented default,
  with validation/tests covering both CPU and optional CUDA execution.
- Optional dependency behavior for Warp remains unchanged.

## Verification criteria

- Targeted GPU tests pass.
- Round-trip tests assert field values, shapes, and `float64` dtypes for one-box
  and multi-box inputs.
- Linting passes for changed GPU/environment modules.
- Documentation describes transfer semantics and no-hidden-transfer behavior.
- The implementation does not block downstream kernel migration tracks.
