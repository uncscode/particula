# Open Questions

- What exact particle count and time-step count produce stable energy and mass
  deltas while keeping the test fast? Start from the existing particle-resolved
  condensation integration test and reduce only if stability remains strong.
- Should the test construct `CondensationLatentHeat` directly through
  `par.dynamics` or through `CondensationFactory(strategy_type="latent_heat")`?
  Direct public construction is likely clearer for the baseline.
- What tolerance is appropriate for total water inventory and energy comparison
  after fixture tuning? Existing integration tolerance (`delta=1e-9`) is the
  starting point, with tighter `np.testing.assert_allclose` possible for energy.
- Has E3-F6 documentation landed at implementation time, and if so what exact
  example path should be cross-linked?
