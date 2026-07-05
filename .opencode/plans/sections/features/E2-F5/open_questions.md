# Open Questions

## Resolved Answers

- E2-F2 provides `EnvironmentData`; E2-F3 mirrors it as `WarpEnvironmentData`.
  Required fields are `temperature`, `pressure`, and `saturation_ratio`.
- Reject calls that provide both explicit environment state and scalar
  temperature/pressure. Explicit rejection is clearer than precedence rules for
  early migration and remains the shipped contract.
- Reserve `environment` as a keyword-only parameter on
  `condensation_step_gpu(...)` and `coagulation_step_gpu(...)` so the
  environment handoff point does not break positional scalar callers.
- Shared `_ensure_environment_arrays(...)` validation is the canonical way to
  normalize scalar, direct-array, and `WarpEnvironmentData` inputs without
  hidden device transfers.
- Explicit environment execution now works when `environment.temperature` and
  `environment.pressure` are valid `(n_boxes,)` Warp arrays on the caller
  device.
- Condensation now prepares per-box dynamic viscosity and mean free path once
  per call in device-local precompute work and reuses them during the main
  kernel launch.
- The public `condensation_step_gpu(...)` docstring already matches the shipped
  scalar/direct-array/hybrid/explicit-environment contract, so P3 only needed
  regression coverage rather than another documentation edit pass.
- `saturation_ratio` remains part of the environment schema, but these entry
  points still only consume temperature and pressure in this phase.
- Keep scalar functions as the primary public entry points initially. Add
  environment support internally or through narrowly scoped wrappers before
  expanding public exports.
