## Success Criteria

### Pass / Fail Criteria

- [ ] A runnable CPU latent-heat condensation example lands under
  `docs/Examples/Dynamics/Condensation/` and executes a real
  `par.dynamics.MassCondensation.execute()` flow.
- [ ] The example uses documented public API paths for condensation and latent
  heat construction, rather than private helpers or test-only shortcuts.
- [ ] At least one execution step reports non-zero, finite
  `CondensationLatentHeat.last_latent_heat_energy` so the example demonstrates
  actual bookkeeping rather than configuration only.
- [ ] The example is discoverable from the Dynamics examples index and any new
  notebook remains generated from the `.py` source, synced, and executable.
- [ ] Validation commands for linting, script execution, notebook sync, and
  notebook execution are recorded or run as part of the implementation flow.
- [ ] No touched documentation claims GPU latent-heat production support or
  parity, and any required production-code fix is covered by focused tests.

### Validation Evidence

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Runnable example | Example executes end-to-end on CPU with a real aerosol setup | Example script run from P1/P3 |
| Energy bookkeeping | Output shows finite, non-zero latent-heat energy after condensation | Example logs or notebook output |
| Discoverability | Dynamics examples index links to the final artifact path | `docs/Examples/Dynamics/index.md` |
| Notebook hygiene | If paired, `.py` → `.ipynb` sync and notebook execution both succeed | `validate_notebook.py` and notebook run results |
| Scope control | Docs remain CPU-only and avoid GPU parity claims | Reviewed docs diff and focused grep |

### Definition of Done

All phases are implemented, validated, and documented, with a reviewer able to
confirm that the example demonstrates real CPU latent-heat mass-transfer
bookkeeping rather than only configuring latent heat properties.
