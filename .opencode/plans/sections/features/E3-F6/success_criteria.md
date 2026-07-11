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
- [ ] The example is discoverable from the Dynamics examples index and any new
  notebook remains generated from the `.py` source, synced, and executable.
- [x] Validation commands for linting, script execution, and focused pytest are
  recorded for the implementation flow; notebook sync/execution remains
  deferred because no notebook shipped.
- [x] No touched documentation claims GPU latent-heat production support or
  parity, and any required production-code fix is covered by focused tests.

### Validation Evidence

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Runnable example | Example executes end-to-end on CPU with a real aerosol setup | `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` |
| Energy bookkeeping | Output shows finite, non-zero latent-heat energy after condensation | Example output plus `condensation_latent_heat_example_test.py` |
| Discoverability | Deferred; no Dynamics examples index link shipped in issue #1263 | Follow-up documentation phase |
| Notebook hygiene | Deferred; no paired notebook shipped in issue #1263 | Follow-up documentation phase |
| Scope control | Docs remain CPU-only and avoid GPU parity claims | Reviewed docs diff and focused grep |

### Definition of Done

Issue #1263 satisfies the runnable-example slice of the feature: reviewers can
confirm that the shipped `.py` example demonstrates real CPU latent-heat
mass-transfer bookkeeping rather than only configuring latent heat properties.
Notebook pairing and docs-index discoverability remain deferred follow-up work.
