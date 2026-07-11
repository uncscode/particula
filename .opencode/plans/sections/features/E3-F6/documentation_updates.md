## Documentation Updates

### New Documentation

- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` with
  runnable CPU-only code and printed latent-heat bookkeeping diagnostics.

### Updated Documentation

- No docs index wiring or feature-doc cross-link shipped with issue #1263.
- User-facing validation coverage was added in
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
  to keep the published example executable.

### Required Wording

- State that the example is CPU-only.
- State that `last_latent_heat_energy` is per-step energy bookkeeping in joules.
- State the sign convention: positive for condensation, negative for evaporation.
- State that the example demonstrates a runnable condensation workflow and is
  not merely configuring vapor-property latent heat.
- Avoid claiming temperature feedback or GPU latent-heat parity.

### Documentation Validation

- Lint and execute the `.py` source.
- Run the focused example test module.
- Notebook sync/execution and markdown-link validation were not applicable to
  the shipped change because no notebook or index wiring was added.
