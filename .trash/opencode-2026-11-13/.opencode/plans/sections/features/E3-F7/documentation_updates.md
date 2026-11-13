# Documentation Updates

## Current Slice Status

- No documentation files changed in issue #1267 / E3-F7-P1.
- No documentation files changed in issue #1268 / E3-F7-P2.
- Issue #1269 / E3-F7-P3 shipped the planned docs-only follow-up in:
  - `docs/Features/Roadmap/data-oriented-gpu.md`
  - `docs/Features/condensation_strategy_system.md`

## Implemented Updates

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Documents E3-F7 as the current CPU integration-level latent-heat baseline
    for future Epic D GPU parity work.
  - Points readers to
    `particula/integration_tests/condensation_latent_heat_conservation_test.py`
    as the executable reference test.
  - Summarizes the baseline invariants conservatively: finite nonzero
    condensation transfer, particle water gain, gas water loss, total water
    conservation, and final-step `last_latent_heat_energy` agreement with the
    constant-latent-heat bookkeeping path.
  - Keeps wording future-facing and explicitly avoids any claim of shipped GPU
    latent-heat parity or temperature-feedback runtime support.

- `docs/Features/condensation_strategy_system.md`
  - Adds the same executable CPU baseline reference in the
    `CondensationLatentHeat` section.
  - Keeps the public support boundary explicit: CPU-only and
    diagnostic/reference only.
  - Includes the existing E3-F6 latent-heat example cross-link in related
    documentation alongside the integration baseline path.

## Validation Notes

- The shipped docs changes stay limited to the two planned documentation files.
- The executable baseline remains
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`.
- Feature-page wording continues to avoid implying completed GPU latent-heat
  parity.

## Wording Guardrails

- Use "CPU baseline", "CPU reference", or "future Epic D parity target".
- Avoid phrases that imply GPU latent-heat support is complete.
- Do not add CUDA or Warp validation instructions for this baseline.
