# Open Questions

## Resolved Answers

- `from_warp_gas_data()` should prefer explicit `name` input. Placeholder names
  may remain as documented fallback behavior, but they must not be treated as
  preserved metadata.
- Non-binary `WarpGasData.partitioning` values should raise before casting to CPU
  booleans.
- Missing `vapor_pressure` in generic `to_warp_gas_data()` may remain a
  zero-filled default, but condensation paths that need vapor pressure should
  pass it explicitly and document that requirement.
- GPU vapor pressure should not be returned to CPU as part of `GasData`. If a
  caller needs it, they should read it from `WarpGasData` before conversion or
  manage an explicit sidecar outside `GasData`.
- E2-F2 and E2-F3 constrain vapor-pressure ownership by keeping environment state
  to `temperature`, `pressure`, and `saturation_ratio`; vapor pressure remains a
  process/kernel input derived from gas and thermodynamic state.
