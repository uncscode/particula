## Documentation Updates

### New Documentation

- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` with
  runnable CPU-only code and printed latent-heat bookkeeping diagnostics.
- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` as the
  published paired notebook artifact synced from the `.py` source.

### Updated Documentation

- `docs/Examples/Dynamics/index.md` now links readers to
  `Condensation/Condensation_Latent_Heat.ipynb` and retains the `.py` file only
  as the editable source of truth.
- `docs/Features/condensation_strategy_system.md` now contains one direct
  latent-heat example cross-link for discoverability.
- User-facing validation coverage was added in
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
  to keep the published example executable and aligned with the notebook/index
  docs surface.

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
- Confirm the paired notebook exists at the published path.
- Confirm the Dynamics index contains the notebook path and no longer advertises
  the raw `python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`
  command entry.
- If the feature page is touched, keep exactly one direct latent-heat example
  cross-link.
