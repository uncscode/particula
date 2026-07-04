# Open Questions

## Resolved Answers

- E2-F2 provides `EnvironmentData`; E2-F3 mirrors it as `WarpEnvironmentData`.
  Required fields are `temperature`, `pressure`, and `saturation_ratio`.
- Reject calls that provide both explicit environment state and scalar
  temperature/pressure. Explicit rejection is clearer than precedence rules for
  early migration.
- Precompute per-box dynamic viscosity and mean free path on the host for the
  first migration path. Move those calculations into Warp kernels only when a
  later feature needs fully device-resident thermodynamic updates.
- Include `saturation_ratio` in the first migration path even if a specific T5
  path only consumes temperature and pressure, so environment shape semantics are
  exercised consistently.
- Keep scalar functions as the primary public entry points initially. Add
  environment support internally or through narrowly scoped wrappers before
  expanding public exports.
