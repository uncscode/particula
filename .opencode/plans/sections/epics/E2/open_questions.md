## Open Questions

## Resolved Answers

1. Simulation `volume` remains authoritative on `ParticleData` for E2. Treat it
   as per-box simulation-domain metadata carried by the particle container, not
   as mutable `EnvironmentData` state.
2. Store `saturation_ratio` as the canonical primitive saturation field on
   `EnvironmentData`. Use shape `(n_boxes, n_species)` so supersaturation and
   species-specific gas state are representable without an upper bound of `1.0`.
3. Keep vapor pressure out of CPU `GasData` and `EnvironmentData` ownership. It
   remains explicit process/kernel input or GPU scratch/cache state, with loss
   semantics documented when converting back to CPU.
4. `WarpGasData` remains numeric only. CPU restoration should require or strongly
   prefer caller-supplied species names; placeholder names are only a documented
   fallback, not semantic preservation.
5. The minimum per-box environment interface for E2 is `temperature`, `pressure`,
   and `saturation_ratio`, all indexed by `n_boxes`, with `saturation_ratio`
   additionally indexed by `n_species`.
6. CPU dynamics may accept data containers only for supported single-box paths.
   Unsupported `n_boxes != 1` calls should raise explicit errors rather than use
   transitional box-0 behavior.
7. Numerical tolerance decisions are delegated to E2-F6 and E2-F7. The baseline
   should be `fp64`; lower-precision evidence is exploratory until E2-F6 closes
   the precision envelope.

## Remaining Follow-Up

- E2-F7 should add or expand gas-coupled condensation integration scope so the
  production path updates both particle and gas state with conservation tests.
