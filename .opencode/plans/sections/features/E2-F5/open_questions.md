# Open Questions

## Resolved Answers

- E2-F2 provides `EnvironmentData`; E2-F3 mirrors it as `WarpEnvironmentData`.
  Required fields are `temperature`, `pressure`, and `saturation_ratio`.
- Reject calls that provide both explicit environment state and scalar
  temperature/pressure. Explicit rejection is clearer than precedence rules for
  early migration.
- Reserve `environment` as a keyword-only parameter on
  `condensation_step_gpu(...)` and `coagulation_step_gpu(...)` so P1 publishes
  the future handoff point without changing positional scalar callers.
- In P1, explicit environment execution remains intentionally unsupported:
  `temperature=None`, `pressure=None`, and `environment=...` raises a
  phase-scoped `ValueError` until later phases wire per-box state into
  host-side condensation helpers and Brownian coagulation launch inputs.
- Precompute per-box dynamic viscosity and mean free path on the host for the
  first migration path. Move those calculations into Warp kernels only when a
  later feature needs fully device-resident thermodynamic updates.
- Include `saturation_ratio` in the later migration path even if a specific
  E2-F5 step only consumes temperature and pressure, so environment shape
  semantics are exercised consistently.
- Keep scalar functions as the primary public entry points initially. Add
  environment support internally or through narrowly scoped wrappers before
  expanding public exports.
