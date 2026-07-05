# Open Questions

## Resolved Answers

- `from_warp_gas_data()` should prefer explicit `name` input. Placeholder names
  are now the shipped fallback behavior for omitted or `None` names after
  `#1198`, but they must not be treated as preserved metadata.
- Wrong-length and empty provided name lists now fail with explicit
  actual/expected count messaging rather than falling back silently.
- Non-binary `WarpGasData.partitioning` values now raise `ValueError` before
  CPU bool restoration; only binary `0/1` values are accepted at the GPU→CPU
  boundary.
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

- Whether placeholder-name fallback should remain the long-term API contract,
  gain warnings, or become stricter in a future migration-focused change is
  still deferred beyond the shipped `#1198` implementation.
