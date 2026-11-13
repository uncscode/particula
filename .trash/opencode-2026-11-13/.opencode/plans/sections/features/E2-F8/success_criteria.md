# E2-F8 Success Criteria

## Functional Criteria

- CPU condensation public data-container paths reject multi-box inputs with
  tested current `ValueError` messages.
- CPU coagulation multi-box `ParticleData` behavior is no longer implicit in the
  plan: helper-backed reads and particle-resolved `step()` paths now reject
  unsupported multi-box execution with tested `ValueError` messages.
- Single-box `ParticleData`/`GasData` dynamics behavior remains supported.

## Documentation Criteria

- The migration guide now explicitly distinguishes container shape support from
  CPU strategy execution support without reopening the baseline audit.
- Roadmap text is now qualified so container compatibility is not read as proof
  of current CPU multi-box execution.
- User guidance now names the supported workaround for multi-box workloads:
  caller-managed per-box execution until first-class CPU multi-box strategies
  exist.

## Test Criteria

- P1/P2 focused condensation and coagulation tests define the executable CPU
  support boundary used by the final docs.
- P3 validation confirms the documentation reflects the existing tested
  condensation public error boundary plus explicit coagulation multi-box
  rejection behavior.
- No standalone testing phase is required because tests are co-located with the
  phases that change behavior.

## Done Signal

Plan sections, shipped docs, and co-located tests clearly distinguish
data-container multi-box shape support from current CPU strategy execution
support for dynamics.
