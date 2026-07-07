# E2-F8 Success Criteria

## Functional Criteria

- CPU condensation public data-container paths reject multi-box inputs with
  tested current `ValueError` messages.
- CPU coagulation multi-box `ParticleData` behavior is no longer implicit in the
  plan: helper-backed reads and particle-resolved `step()` mutation are
  explicitly captured as box-0-only baseline behavior until a later phase
  changes it.
- Single-box `ParticleData`/`GasData` dynamics behavior remains supported.

## Documentation Criteria

- Future doc targets are named precisely enough that later phases can
  distinguish container shape support from CPU strategy execution support
  without reopening the baseline audit.
- Any roadmap text that could imply all current strategies execute every box is
  identified for follow-up clarification.
- If user guidance is added later, it should name the workaround for multi-box
  workloads: caller-managed per-box execution or waiting for future first-class
  multi-box strategies.

## Test Criteria

- Focused condensation tests pass.
- Focused coagulation strategy tests pass.
- Assertions cover the current condensation public error boundary plus current
  coagulation box-0-only read and mutation behavior.
- No standalone testing phase is required because tests are co-located with the
  phases that change behavior.

## Done Signal

Plan sections and co-located tests clearly distinguish data-container multi-box
shape support from current CPU strategy execution support for dynamics.
