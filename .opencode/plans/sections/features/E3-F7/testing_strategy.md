# Testing Strategy

## Test Locations

- Primary new test file:
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.
- Existing reference style:
  `particula/integration_tests/condensation_particle_resolved_test.py`.

## Per-Phase Testing Approach

- **P1 (implemented):** The new module ships a deterministic single-species
  CPU integration fixture with two focused tests: one verifies the initial
  state is supersaturated, and one runs a short fixed
  `MassCondensation.execute()` loop and checks that gas decreases, particle
  mass increases, and latent-heat bookkeeping stays finite. No multi-species
  extension was added in this slice.
- **P2 (implemented):** The same integration test now captures initial,
  pre-final, and final particle/gas water bookkeeping; asserts whole-run water
  conservation; and checks that final-step particle gain, gas loss, and
  `last_latent_heat_energy` all close against the explicit constant latent heat.
- **P3:** Keep documentation validation lightweight; verify the documented test
  path and CPU-only constraints match the implemented baseline.

## Current Assertions

- Particle water inventory increases after condensation.
- Partitioning gas water concentration decreases.
- Initial supersaturation is asserted so the baseline cannot silently become a
  no-op fixture.
- Total water inventory across particles and gas is conserved within
  `CONSERVATION_RTOL = 1e-12` and `CONSERVATION_ATOL = 1e-18`.
- `CondensationLatentHeat.last_latent_heat_energy` is finite and positive.
- Final-step particle water gain matches final-step gas water loss.
- Expected final-step energy from transferred mass times
  `LATENT_HEAT_WATER = 2.26e6` matches `last_latent_heat_energy`.

## Coverage Impact

The feature adds integration coverage for a path that already has unit coverage.
Coverage thresholds must not be lowered. The test should remain fast and should
not require `slow`, `performance`, GPU, CUDA, or Warp markers.

## Suggested Validation Commands

```bash
pytest particula/integration_tests/condensation_latent_heat_conservation_test.py -q -Werror
pytest particula/integration_tests -q
```
