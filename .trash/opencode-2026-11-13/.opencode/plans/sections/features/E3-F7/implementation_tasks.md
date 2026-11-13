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

- [x] Captured initial/final particle water inventory and initial/final gas water
  concentration on the shipped single-species CPU fixture.
- [x] Added explicit `LATENT_HEAT_WATER`, `CONSERVATION_RTOL`, and
  `CONSERVATION_ATOL` module constants for deterministic bookkeeping checks.
- [x] Added a private `_particle_water_inventory(...)` helper local to
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.
- [x] Split the execution flow so the test records the penultimate state before
  the fifth/final `MassCondensation.execute()` call.
- [x] Asserted whole-run directional transfer: particle water increases and gas
  water decreases.
- [x] Asserted whole-run total water inventory conservation with explicit tight
  deterministic tolerances.
- [x] Asserted `last_latent_heat_energy` is finite and positive.
- [x] Asserted final-step particle gain matches final-step gas loss and that the
  recorded latent-heat energy equals final-step transferred mass times the exact
  latent-heat constant used by the fixture.
- [x] Kept the implementation scoped to the existing integration test module
  only, with no production-code or documentation changes in issue #1268.

## E3-F7-P3: Documentation

- Update `docs/Features/Roadmap/data-oriented-gpu.md` to state that E3-F7 adds
  the CPU reference baseline for later Epic D latent-heat parity.
- Update `docs/Features/condensation_strategy_system.md` with a concise note
  about the integration-level latent-heat conservation baseline.
- Cross-link to the E3-F6 runnable example if the example has landed.
- Preserve clear wording that GPU latent-heat parity remains future work and
  keep the docs delta to those two files unless a missing cross-link would leave
  the CPU baseline undiscoverable.
