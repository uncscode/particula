# Open Questions

## Resolved Answers

- `from_warp_gas_data()` should prefer explicit `name` input. Placeholder names
  remain the current documented fallback behavior in the shipped `#1197`
  regression tests, but they must not be treated as preserved metadata.
- Missing `vapor_pressure` in generic `to_warp_gas_data()` may remain a
  zero-filled default, but condensation paths that need vapor pressure should
  pass it explicitly and document that requirement.
- GPU vapor pressure should not be returned to CPU as part of `GasData`. If a
  caller needs it, they should read it from `WarpGasData` before conversion or
  manage an explicit sidecar outside `GasData`.
- E2-F2 and E2-F3 constrain vapor-pressure ownership by keeping environment state
  to `temperature`, `pressure`, and `saturation_ratio`; vapor pressure remains a
  process/kernel input derived from gas and thermodynamic state.

## Still Open for Later Phases

- Whether non-binary `WarpGasData.partitioning` values should raise before
  casting to CPU booleans was not changed in `E2-F4-P1`; the phase only locked
  the current bool↔`int32` happy-path contract.
- Whether placeholder-name fallback should remain indefinitely, gain warnings,
  or become a stricter error contract is deferred beyond the shipped test-only
  audit phase.
