## Testing Strategy

### Phase-Level Validation

- **P1 shipped:**
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
  now covers the published example path with smoke and invariant assertions.
  Validation is centered on running the example as `__main__`, checking the
  helper payload, and confirming finite bookkeeping diagnostics for the chosen
  CPU condensation setup.
- **P2 shipped in issue #1264:** the docs surface now includes the paired
  notebook, the Dynamics index notebook link, and one targeted feature-doc
  cross-link.
- **P3 final validation pass for issue #1265:** the focused example test module
  remains the primary regression harness for entrypoint/helper behavior and the
  published docs surface, including notebook presence, Dynamics index link
  correctness, raw-command removal, and singular feature-page cross-linking.

### Commands

```bash
python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py
ruff check docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py --fix
ruff format docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py
python3 .opencode/tools/validate_notebook.py docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb --sync
python3 .opencode/tools/run_notebook.py docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb
pytest particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py -q -Werror
```

Docs publication now expects the paired notebook path to remain stable, linked
from the published markdown surface, and executable after any `.py` alignment
edit.

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
- The shipped test module should cover both entrypoint execution and helper
  invariants so the docs example stays runnable and reviewable.
- The same test module should guard the published docs surface by checking the
  paired notebook path, Dynamics index link, and targeted feature-doc link.
- P3 should stay validation-first: prefer reusing the shipped smoke/module
  coverage instead of broadening test scope.
- A zero-energy result is acceptable only if explicitly explained by a no-transfer
  setup; the preferred example should choose conditions that produce a visible
  non-zero diagnostic.
