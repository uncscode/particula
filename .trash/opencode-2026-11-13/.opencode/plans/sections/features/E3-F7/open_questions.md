# Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Start from `particula/integration_tests/condensation_particle_resolved_test.py`
  and reduce fixture size only if mass and energy deltas remain stable. Keep the
  final test fast enough for normal integration-test execution.
- Construct `CondensationLatentHeat` directly through `par.dynamics` for the
  baseline. Direct public construction is clearer for a conservation reference;
  factory coverage can remain separate.
- Use the existing integration tolerance of `1e-9` as the initial water
  inventory tolerance, then tighten with `np.testing.assert_allclose` only after
  fixture tuning proves stable energy behavior.
- Cross-link to the E3-F6 example when it exists. This is now resolved in the
  shipped P3 docs update: `docs/Features/condensation_strategy_system.md`
  keeps the existing latent-heat example link and also points readers to the
  executable CPU integration baseline.
