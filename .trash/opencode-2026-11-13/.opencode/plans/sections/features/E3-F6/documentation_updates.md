## Documentation Updates

### Shipped Documentation Artifacts

- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` with
  runnable CPU-only code and printed latent-heat bookkeeping diagnostics.
- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` as the
  published paired notebook artifact synced from the `.py` source.

### Validated Published Surfaces

- `docs/Examples/Dynamics/index.md` links readers to
  `Condensation/Condensation_Latent_Heat.ipynb` and retains the `.py` file as
  the editable source of truth.
- `docs/Features/condensation_strategy_system.md` contains one direct
  latent-heat example cross-link.
- User-facing validation coverage in
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
  keeps the published example executable and aligned with the notebook/index
  docs surface.

### Required Wording

- State that the example is CPU-only.
- State that `last_latent_heat_energy` is per-step energy bookkeeping in joules.
- State the sign convention: positive for condensation, negative for evaporation.
- State that the example demonstrates a runnable condensation workflow rather
  than merely configuring vapor-property latent heat.
- Avoid claiming temperature feedback or GPU latent-heat parity.

### Documentation Validation

- Execute the `.py` source and paired notebook validation workflow.
- Run the focused example test module.
- Confirm the paired notebook exists at the published path.
- Confirm the Dynamics index contains the notebook path and no longer advertises
  the raw `python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`
  command entry.
- Confirm the feature page keeps exactly one direct latent-heat example
  cross-link.
- Keep any P3 edit bounded to wording, notebook sync, or docs-link alignment.
