## Success Criteria

### Pass / Fail Criteria

- [x] A runnable CPU latent-heat condensation example landed under
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` and uses a
  real `par.dynamics.MassCondensation.execute()` flow.
- [x] The example uses documented public API paths for condensation and latent
  heat construction, rather than private helpers or test-only shortcuts.
- [x] At least one execution step reports non-zero, finite
  `CondensationLatentHeat.last_latent_heat_energy` so the example demonstrates
  actual bookkeeping rather than configuration only.
- [x] The example is discoverable from the Dynamics examples index and the
  paired notebook now exists at
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb`.
- [x] Validation commands for linting, script execution, and focused pytest are
  recorded for the implementation flow, and the focused test module now guards
  notebook/link alignment for the published docs surface.
- [x] No touched documentation claims GPU latent-heat production support or
  parity, and any required production-code fix is covered by focused tests.

### Validation Evidence

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Runnable example | Example executes end-to-end on CPU with a real aerosol setup | `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` |
| Energy bookkeeping | Output shows finite, non-zero latent-heat energy after condensation | Example output plus `condensation_latent_heat_example_test.py` |
| Discoverability | Dynamics examples index links the published latent-heat notebook | `docs/Examples/Dynamics/index.md` plus `condensation_latent_heat_example_test.py` |
| Notebook hygiene | Paired notebook exists at the published path and remains sourced from the `.py` workflow | `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` plus focused docs-surface assertions |
| Scope control | Docs remain CPU-only and avoid GPU parity claims | Reviewed docs diff and focused grep |

### Definition of Done

Issues #1263 and #1264 together satisfy the feature: reviewers can confirm that
the shipped `.py` example demonstrates real CPU latent-heat mass-transfer
bookkeeping, that the paired notebook is published, and that the docs surface
points readers to the notebook without introducing GPU parity claims.
