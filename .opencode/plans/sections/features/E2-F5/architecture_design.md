# Architecture Design

## Design Principles

- Preserve source compatibility for scalar GPU API callers.
- Treat environment state as a first-class per-box input owned outside
  `GasData`.
- Normalize and validate before launching Warp kernels so device code can assume
  consistent `(n_boxes,)` inputs.
- Keep conversion explicit; helpers may construct Warp arrays from scalars, but
  CPU-to-GPU environment container transfers should live in conversion utilities.

## Proposed Compatibility Layer

Implement a small normalization layer that accepts one of these forms:

1. Legacy scalar `temperature: float` and `pressure: float`.
2. Per-box temperature/pressure Warp arrays shaped `(n_boxes,)`.
3. E2-F3 `WarpEnvironmentData` containing per-box temperature and pressure.

The helper should return canonical per-box arrays for kernel launch. Scalar
values are broadcast with `wp.full(n_boxes, value, dtype=wp.float64, device=...)`.
Array/environment inputs are validated for exact shape and device match.

## API Options

- Preferred: keep existing scalar parameters and add an optional keyword-only
  `environment=None` or internal wrapper that does not break positional callers.
- Alternative: introduce `*_step_gpu_environment(...)` wrappers while retaining
  the existing scalar functions as compatibility shims.
- Avoid: changing positional ordering or making environment mandatory in the
  first migration step.

## Kernel Feed Points

- Condensation: replace globally scalar temperature-derived inputs with per-box
  values indexed by `box_idx`. If dynamic viscosity and mean free path remain
  precomputed on CPU, precompute them as `(n_boxes,)`; otherwise compute them in
  device code from per-box temperature/pressure.
- Coagulation: change Brownian kernel inputs from scalar temperature/pressure to
  arrays and use `temperature[box_idx]` and `pressure[box_idx]`.

## Error Handling

- Environment shape must be `(n_boxes,)`; include expected shape in error text.
- Environment device must match particle arrays before launch.
- `n_boxes` comes from `particles.masses.shape[0]` and must align with gas and
  environment state.
- If scalar and environment values are both provided, the implementation must
  either reject ambiguous inputs or document that explicit environment wins.
