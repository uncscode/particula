# Implementation Tasks

## E3-F7-P1: Fixture Adaptation

- [x] Reviewed
  `particula/integration_tests/condensation_particle_resolved_test.py` and
  mirrored its deterministic setup style.
- [x] Created
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.
- [x] Built a supersaturated CPU water condensation scenario with partitioning
  gas and particle-resolved water mass, keeping test-local fixture builders in
  the same module as private helpers.
- [x] Instantiated `CondensationLatentHeat` through public imports with a
  constant latent-heat strategy produced by `par.gas.LatentHeatFactory()`.
- [x] Executed a short fixed loop using `MassCondensation.execute()`.
- [x] Kept the shipped P1 assertions intentionally lightweight: supersaturation
  precondition, gas decrease, particle mass increase, and finite
  `last_latent_heat_energy`.
- [x] Kept the diff scoped to the new integration test module only.

## E3-F7-P2: Conservation and Energy Assertions

- Capture initial particle water inventory and gas water concentration.
- Capture final particle water inventory and gas water concentration.
- Assert particle water increased and gas water decreased.
- Assert total water inventory is conserved within a stable deterministic
  tolerance, using the existing integration test's tolerance as the starting
  point.
- Assert `last_latent_heat_energy` is finite and positive.
- Compute expected latent heat from transferred mass and the exact constant
  latent-heat value passed into the factory or strategy setup, then compare with
  `last_latent_heat_energy` using tight but stable tolerances.
- Run the focused integration test and then the default integration-test command
  used by the repository.
- Keep the assertion edit to one integration test module with a small number of
  named assertions or helper functions, not a new shared testing utility.

## E3-F7-P3: Documentation

- Update `docs/Features/Roadmap/data-oriented-gpu.md` to state that E3-F7 adds
  the CPU reference baseline for later Epic D latent-heat parity.
- Update `docs/Features/condensation_strategy_system.md` with a concise note
  about the integration-level latent-heat conservation baseline.
- Cross-link to the E3-F6 runnable example if the example has landed.
- Preserve clear wording that GPU latent-heat parity remains future work and
  keep the docs delta to those two files unless a missing cross-link would leave
  the CPU baseline undiscoverable.
