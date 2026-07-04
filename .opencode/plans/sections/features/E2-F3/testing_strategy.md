# Testing Strategy

## Co-located testing policy

Every implementation phase includes unit tests alongside the changed GPU code.
There is no standalone testing-only implementation phase.

## Required test coverage

- `WarpEnvironmentData` struct can be instantiated and populated on the Warp
  CPU backend.
- Every field has the expected `float64` dtype and `(n_boxes,)` shape.
- One-box and multi-box environment containers transfer correctly.
- `to_warp_environment_data(..., device="cpu")` preserves values.
- `from_warp_environment_data(..., sync=True)` reconstructs the CPU container.
- Full CPU round trip preserves all environment fields.
- `sync=False` path is covered where safe.
- Invalid device names raise the same style of `RuntimeError` as existing
  conversion helpers.
- Optional CUDA-parametrized tests run when `warp_devices(wp)` includes
  `"cuda"` and skip otherwise.

## Suggested test locations

- `particula/gpu/tests/warp_types_test.py`
- `particula/gpu/tests/conversion_test.py`
- Reuse `particula/gpu/tests/cuda_availability.py`.

## Commands

```bash
pytest particula/gpu/tests/warp_types_test.py \
  particula/gpu/tests/conversion_test.py
ruff check particula/gpu particula/gas --fix
ruff format particula/gpu particula/gas
ruff check particula/gpu particula/gas
```

## Acceptance threshold

All Warp CPU tests pass. CUDA-specific assertions are active only on machines
where Warp reports CUDA availability.
