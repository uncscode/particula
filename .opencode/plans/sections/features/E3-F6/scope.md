## Scope

### In Scope

- Re-run the shipped CPU latent-heat example and paired notebook as validation
  artifacts.
- Confirm the published example source and paired notebook remain the docs
  source-of-truth pair under
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.{py,ipynb}`.
- Confirm `docs/Examples/Dynamics/index.md` still links the latent-heat
  notebook.
- Confirm `docs/Features/condensation_strategy_system.md` keeps the single
  targeted latent-heat example cross-link.
- Keep focused smoke coverage in
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
  aligned with example execution and docs-surface assertions.
- Apply only minimal wording, sync, or link-alignment edits if validation finds
  drift.

### Out of Scope

- No GPU or Warp latent-heat example.
- No claim that GPU condensation has latent-heat parity.
- No new example authoring, docs-surface expansion, or additional discoverability
  work beyond the already shipped notebook/index/feature-page surfaces.
- No production changes to condensation numerics unless a tiny bug blocks
  validation; any such change must include co-located tests.
- No broad rewrite of condensation documentation beyond small alignment edits.

### Acceptance Boundary

The feature is complete when the shipped CPU example and paired notebook remain
executable, the Dynamics index and condensation feature page still point readers
to the published notebook surface, and validation confirms only minimal
alignment edits were needed.
