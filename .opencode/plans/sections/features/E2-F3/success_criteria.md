# Success Criteria

## Functional criteria

- `WarpEnvironmentData` exists and mirrors the CPU `EnvironmentData` numeric
  schema from `E2-F2`.
- CPU-to-Warp and Warp-to-CPU helpers are public through `particula.gpu` when
  Warp is available.
- Environment round trips preserve every field on the Warp CPU backend.
- CUDA-parametrized round-trip coverage runs automatically when CUDA is
  available and skips cleanly otherwise.
- Existing particle and gas conversion behavior is unchanged.

## Design criteria

- Transfers are explicit and centralized in conversion helpers.
- No kernel, runnable, or strategy object performs an implicit CPU/GPU transfer.
- Field shapes remain `(n_boxes,)`, including single-box simulations.
- Optional dependency behavior for Warp remains unchanged.

## Verification criteria

- Targeted GPU tests pass.
- Linting passes for changed GPU/environment modules.
- Documentation describes transfer semantics and no-hidden-transfer behavior.
- The implementation does not block downstream kernel migration tracks.
