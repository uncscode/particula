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

The shipped coverage lives entirely in
`particula/gpu/tests/warp_types_test.py`; conversion-helper and CUDA coverage
remain future-phase work because no transfer API was added in issue `#1192`.

## Suggested test locations

- `particula/gpu/tests/warp_types_test.py`

## Commands

```bash
pytest particula/gpu/tests/warp_types_test.py -v -Werror
ruff check particula/gpu --fix
ruff format particula/gpu
ruff check particula/gpu
```

## Acceptance threshold

All `WarpEnvironmentData` schema tests in `warp_types_test.py` pass on the Warp
CPU backend with exact shape, dtype, field-access, and deterministic value
assertions.
