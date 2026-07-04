# E2-F8 Success Criteria

## Functional Criteria

- CPU condensation public data-container paths reject multi-box inputs with
  clear, tested `ValueError` messages.
- CPU coagulation multi-box `ParticleData` behavior is no longer ambiguous:
  either unsupported calls raise clear errors, or transitional box-0 behavior is
  explicitly tested and documented.
- Single-box `ParticleData`/`GasData` dynamics behavior remains supported.

## Documentation Criteria

- `docs/Features/particle-data-migration.md` distinguishes container shape
  support from CPU strategy execution support.
- Any roadmap text that could imply all current strategies execute every box is
  clarified.
- User guidance names the workaround for multi-box workloads: caller-managed
  per-box execution or waiting for future first-class multi-box strategies.

## Test Criteria

- Focused condensation tests pass.
- Focused coagulation strategy tests pass.
- Error-message assertions cover unsupported multi-box strategy calls.
- No standalone testing phase is required because tests are co-located with the
  phases that change behavior.

## Done Signal

Docs and tests clearly distinguish data-container multi-box shape support from
strategy-level multi-box execution support for CPU dynamics.
