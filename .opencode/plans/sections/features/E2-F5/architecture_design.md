# Architecture Design

## Design Principles

- Preserve source compatibility for scalar GPU API callers.
- Treat environment state as a first-class per-box input owned outside
  `GasData`.
- Validate the environment contract before launching Warp kernels so ambiguity,
  missing-input, shape, and device rules stay consistent across entry points.
- Keep conversion explicit; helpers may construct Warp arrays from scalars, but
  CPU-to-GPU environment container transfers should live in conversion
  utilities.

## Shipped Shared Normalization Layer

Issue #1204 added `particula/gpu/kernels/environment.py` as the shared private
normalization boundary for condensation and coagulation. Both GPU entry points
now accept these contract forms:

1. Legacy scalar `temperature: float` and `pressure: float`.
2. Direct Warp arrays with shape `(n_boxes,)` on the caller device.
3. Hybrid direct inputs where one side is scalar and the other is a valid Warp
   array.
4. `temperature=None`, `pressure=None`, and `environment=...` with valid
   `WarpEnvironmentData` arrays.

Mixed direct inputs plus `environment=...` still raise an early `ValueError`.
Valid arrays are returned unchanged so the launch path reuses existing
device-local buffers without copies.

## API Options

- Chosen: keep existing scalar parameters and add an optional keyword-only
  `environment=None` that does not break positional callers.
- Chosen: normalize once through a shared helper instead of duplicating
  per-entry-point validation rules.
- Avoided: changing positional ordering, making environment mandatory, or
  introducing hidden device transfers.

## Kernel Feed Points

- Condensation: normalize temperature and pressure once, prepare dynamic
  viscosity and mean free path as per-box arrays through one dedicated
  precompute launch, then reuse those arrays during the per-particle kernel.
- Coagulation: pass normalized temperature and pressure arrays into
  `brownian_coagulation_kernel(...)` and index them by `box_idx` inside the
  kernel.

## Error Handling

- If scalar and environment values are both provided, the implementation should
  raise a clear error instead of applying precedence rules.
- If `environment` is omitted, both direct inputs must still be present.
- Shape, `n_boxes`, and device mismatches now fail with stable pre-launch
  `ValueError` messages that mention expected `(n_boxes,)` arrays or device
  mismatch.
