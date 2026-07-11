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
- Publish the paired notebook artifact at
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` from the
  `.py` source of truth.
- Make the example discoverable from `docs/Examples/Dynamics/index.md`.
- Add at most one targeted feature-doc cross-link when it improves latent-heat
  example discoverability.
- Add focused example coverage in
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`.
- Validate that the example runs from the docs tree and that the published docs
  surface points at the paired notebook path.

### Out of Scope

- No GPU or Warp latent-heat example.
- No claim that GPU condensation has latent-heat parity.
- No production changes to condensation numerics unless a tiny bug blocks the
  example; any such change must include co-located tests.
- No standalone testing-only phase. Validation is attached to each phase and
  consolidated in the final documentation/validation phase.
- No broad rewrite of the condensation strategy documentation beyond one
  targeted latent-heat cross-link.

### Acceptance Boundary

The feature is complete when a user can run the documented CPU example, find it
from the published Dynamics docs surface, and see latent-heat energy produced
by the condensation strategy after executing a real mass-condensation step.
