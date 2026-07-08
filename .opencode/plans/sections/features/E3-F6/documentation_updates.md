## Documentation Updates

### New Documentation

- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` with
  markdown cells and runnable code.
- Optional paired notebook:
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb`.

### Updated Documentation

- `docs/Examples/Dynamics/index.md` to link the new condensation latent-heat
  example.
- Optional targeted link in `docs/Features/condensation_strategy_system.md` from
  the latent-heat section to the full runnable example.

### Required Wording

- State that the example is CPU-only.
- State that `last_latent_heat_energy` is per-step energy bookkeeping in joules.
- State the sign convention: positive for condensation, negative for evaporation.
- State that the example demonstrates a runnable condensation workflow and is
  not merely configuring vapor-property latent heat.
- Avoid claiming temperature feedback or GPU latent-heat parity.

### Documentation Validation

- Lint and execute the `.py` source.
- Sync and execute the notebook if paired.
- Validate changed markdown links where practical.
