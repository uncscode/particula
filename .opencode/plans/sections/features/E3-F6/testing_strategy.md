## Testing Strategy

### Phase-Level Validation

- **P1:** Add a runnable-example smoke test in
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
  or extend an adjacent condensation example test module if one already exists.
  That test should execute the published example entrypoint, assert it exits
  cleanly, and verify the reported latent-heat energy diagnostic is finite and
  non-zero for the chosen condensation setup.
- **P2:** If a notebook is paired, run Jupytext sync validation and execute the
  notebook. Check `docs/Examples/Dynamics/index.md` resolves to the paired
  example path, and keep any example-specific assertions in the same smoke-test
  module rather than a later follow-up phase.
- **P3:** Repeat final script/notebook validation and inspect docs for CPU-only
  language, absence of GPU parity claims, and stable public builder/factory
  imports. If the example uncovers a production API gap, ship the matching
  regression in the same PR under the existing condensation test modules.

### Commands

```bash
python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py
ruff check docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py --fix
ruff format docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py
python3 .opencode/tools/validate_notebook.py docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb --sync
python3 .opencode/tools/run_notebook.py docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb
```

Run notebook commands only when the paired `.ipynb` exists.

### Regression Tests if Production Code Changes

If the feature uncovers a production API issue, add co-located tests with that
fix and run focused checks:

```bash
pytest particula/dynamics/condensation/tests/condensation_factories_test.py -q
pytest particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py -q
pytest particula/dynamics/condensation/tests/condensation_strategies_test.py -q
pytest particula/dynamics/tests/condensation_exports_test.py -q
```

Prefer the following split when choosing test locations:

- Example execution and output assertions:
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
- Builder/factory or latent-heat API regressions:
  `particula/dynamics/condensation/tests/condensation_factories_test.py`
  and
  `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py`
- Strategy or runnable behavior fixes discovered during example work:
  `particula/dynamics/condensation/tests/condensation_strategies_test.py`
  and `particula/dynamics/tests/condensation_exports_test.py`

### Evidence Requirements

- The example must demonstrate energy from actual mass transfer, not only show
  construction of a latent heat vapor property.
- A zero-energy result is acceptable only if explicitly explained by a no-transfer
  setup; the preferred example should choose conditions that produce a visible
  non-zero diagnostic.
