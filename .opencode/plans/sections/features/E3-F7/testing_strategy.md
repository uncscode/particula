# Testing Strategy

## Test Locations

- Primary new test file:
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.
- Existing reference style:
  `particula/integration_tests/condensation_particle_resolved_test.py`.

## Per-Phase Testing Approach

- **P1:** Add the integration fixture and a smoke-level assertion that the CPU
  latent-heat runnable executes deterministically through `MassCondensation`.
- **P2:** Add full conservation and energy bookkeeping assertions in the same
  integration test file. These assertions are the core acceptance gate.
- **P3:** Keep documentation validation lightweight; verify the documented test
  path and CPU-only constraints match the implemented baseline.

## Required Assertions

- Particle water/speciated mass increases after condensation.
- Partitioning gas water concentration decreases.
- Total water inventory across particles and gas is conserved within a stable
  deterministic tolerance.
- `CondensationLatentHeat.last_latent_heat_energy` is finite and positive.
- Expected energy from transferred mass times constant latent heat matches
  `last_latent_heat_energy`.

## Coverage Impact

The feature adds integration coverage for a path that already has unit coverage.
Coverage thresholds must not be lowered. The test should remain fast and should
not require `slow`, `performance`, GPU, CUDA, or Warp markers.

## Suggested Validation Commands

```bash
pytest particula/integration_tests/condensation_latent_heat_conservation_test.py -q
pytest particula/integration_tests -q
```
