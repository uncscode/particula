## Scope

### In Scope

- Create a runnable CPU `CondensationLatentHeat` example in
  `docs/Examples/Dynamics/Condensation/`.
- Use documented public APIs, preferably `par.gas.LatentHeatFactory()` plus
  `par.dynamics.CondensationFactory().get_strategy("latent_heat", ...)` or the
  latent-heat builder path.
- Wrap the strategy in `par.dynamics.MassCondensation` and call
  `execute(aerosol, time_step=..., sub_steps=...)` on a real aerosol.
- Print or plot diagnostics for gas concentration, particle mass change,
  `condensation_strategy.last_latent_heat_energy`, and cumulative latent heat
  energy.
- Add a paired notebook if following the existing condensation example pattern,
  with Jupytext sync from the `.py` source.
- Update the Dynamics examples index and any concise feature-doc links needed
  to make the example discoverable.
- Validate that the example runs from the docs tree.

### Out of Scope

- No GPU or Warp latent-heat example.
- No claim that GPU condensation has latent-heat parity.
- No production changes to condensation numerics unless a tiny bug blocks the
  example; any such change must include co-located tests.
- No standalone testing-only phase. Validation is attached to each phase and
  consolidated in the final documentation/validation phase.
- No broad rewrite of the condensation strategy documentation beyond targeted
  cross-links.

### Acceptance Boundary

The feature is complete when a user can run the documented CPU example and see
latent-heat energy produced by the condensation strategy after executing a real
mass-condensation step.
