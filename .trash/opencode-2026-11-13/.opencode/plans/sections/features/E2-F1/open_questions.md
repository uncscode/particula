# Open Questions

## Resolved Answers

1. `ParticleData.density` remains `(n_species,)` and shared across boxes for the
   E2 epic. Future per-box material properties may be reserved in documentation,
   but they are not part of the E2 container contract.
2. `ParticleData.volume` remains authoritative container-carried simulation
   metadata with shape `(n_boxes,)`. `EnvironmentData` should not own or mutate
   volume in E2.
3. `vapor_pressure` is explicit process/kernel state, not CPU `GasData` or
   `EnvironmentData` ownership. GPU paths may carry it as helper state with
   documented loss on CPU restoration.
4. Species names remain CPU metadata. GPU workflows should rely on caller-supplied
   names or an external index map when converting numeric `WarpGasData` back to
   CPU state.
5. The first `EnvironmentData` implementation should include `temperature`,
   `pressure`, and `saturation_ratio`. `saturation_ratio` should use
   `(n_boxes, n_species)`; temperature and pressure should use `(n_boxes,)`.

## Resolution Policy

- These answers are authoritative for sibling tracks unless E2-F2 discovers a
  direct implementation blocker.
