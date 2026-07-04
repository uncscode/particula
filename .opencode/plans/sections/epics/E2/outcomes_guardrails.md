## Outcomes and Guardrails

### Target Outcomes

- A reviewed schema decision for particle, gas, and environment state
  ownership, including shape conventions for per-box and per-species fields.
- `EnvironmentData` for CPU state with validation, copy behavior, and tests.
- `WarpEnvironmentData` plus CPU/GPU conversion helpers and round-trip tests.
- Explicit `GasData` / `WarpGasData` semantics for names, partitioning, vapor
  pressure ownership, and round-trip behavior.
- Backward-compatible migration from scalar temperature/pressure kernel APIs
  toward per-box environment arrays.
- Written studies for mass representation precision and condensation timestep
  stiffness, with recommendations for downstream implementation epics.
- Published user/developer docs and examples for containers, transfer helpers,
  limitations, and roadmap handoff.

### Guardrails

- Do not break existing scalar GPU condensation or coagulation call sites.
- Do not hide CPU/GPU transfer loops behind implicit side effects; transfers
  must remain explicit helper calls.
- Do not lower test coverage thresholds or remove existing validation.
- Keep `float64` as the reference numerical precision until studies justify
  alternatives.
- Keep data containers focused on arrays and metadata; physics behavior remains
  in strategies, properties, or kernels.
- Unit tests ship with each feature implementation; no standalone testing-only
  workstream should be used as a substitute for co-located tests.
