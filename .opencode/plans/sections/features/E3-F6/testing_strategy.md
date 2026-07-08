## Testing Strategy

### Phase-Level Validation

- **P1:** Run the example as a plain Python script and verify it exits with code
  0 and prints non-trivial latent-heat energy diagnostics after a real
  `MassCondensation.execute()` call.
- **P2:** If a notebook is paired, run Jupytext sync validation and execute the
  notebook. Check the examples index link resolves.
- **P3:** Repeat final script/notebook validation and inspect docs for CPU-only
  language and absence of GPU parity claims.

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

### Evidence Requirements

- The example must demonstrate energy from actual mass transfer, not only show
  construction of a latent heat vapor property.
- A zero-energy result is acceptable only if explicitly explained by a no-transfer
  setup; the preferred example should choose conditions that produce a visible
  non-zero diagnostic.
