# Documentation Updates

## Current Slice Status

- No documentation files changed in issue #1267 / E3-F7-P1.
- No documentation files changed in issue #1268 / E3-F7-P2.
- Both shipped slices intentionally stayed limited to
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.

## Deferred Updates

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Mark or describe E3-F7 as the CPU integration-level latent-heat baseline
    for future Epic D GPU parity work.
  - State explicitly that the feature does not claim GPU latent-heat parity.

- `docs/Features/condensation_strategy_system.md`
  - Add a short note that latent-heat condensation now has an integration-level
    CPU baseline under `particula/integration_tests/`.
  - Summarize the later-phase invariants: particle/gas mass conservation and
    latent-heat energy bookkeeping.

## Optional Cross-References

- Link to the E3-F6 runnable CPU latent-heat example if it exists at
  implementation time.
- Mention the focused test command for maintainers who need to reproduce the
  baseline quickly.

## Wording Guardrails

- Use "CPU baseline", "CPU reference", or "future Epic D parity target".
- Avoid phrases that imply GPU latent-heat support is complete.
- Do not add CUDA or Warp validation instructions for this baseline.
