# Testing Strategy

## Co-located testing policy

Every implementation phase includes unit tests alongside the changed GPU code.
There is no standalone testing-only implementation phase.

## Required test coverage

- `WarpEnvironmentData` struct can be instantiated and populated on the Warp
  CPU backend.
- `temperature` and `pressure` keep `float64` dtype with `(n_boxes,)` shape,
  while `saturation_ratio` keeps `float64` dtype with `(n_boxes, n_species)`
  shape.
- One-box and multi-box environment containers preserve their leading box axis.
- Field access assertions prove the struct exposes `temperature`, `pressure`,
  and `saturation_ratio` directly.
- Deterministic NumPy-backed construction plus `.numpy()` checks preserve exact
  values for all three fields.
- `to_warp_environment_data` conversion tests cover populated fields, exact CPU
  values, one-box and multi-box shapes, `wp.float64` dtypes, and invalid-device
  failures.
- Conversion tests also cover Warp-unavailable behavior through the shared
  helper path plus `copy=True` independence and conservative CPU `copy=False`
  semantics.

The shipped coverage now lives across
`particula/gpu/tests/warp_types_test.py` and
`particula/gpu/tests/conversion_test.py`; CUDA coverage remains future-phase
work.

## Suggested test locations

- `particula/gpu/tests/warp_types_test.py`
- `particula/gpu/tests/conversion_test.py`

## Commands

```bash
pytest particula/gpu/tests/warp_types_test.py -v -Werror
pytest particula/gpu/tests/conversion_test.py -v -Werror
ruff check particula/gpu --fix
ruff format particula/gpu
ruff check particula/gpu
```

## Acceptance threshold

All `WarpEnvironmentData` schema and CPU-to-Warp conversion tests pass on the
Warp CPU backend with exact shape, dtype, field-access, deterministic value,
and helper-behavior assertions.
