## Testing Strategy

### Phase-Level Validation

- **P1 shipped:**
  `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
  now covers the published example path with smoke and invariant assertions.
  Validation is centered on running the example as `__main__`, checking the
  helper payload, and confirming finite bookkeeping diagnostics for the chosen
  CPU condensation setup.
- **P2 deferred:** No notebook was paired and no Dynamics index entry was added
  in issue #1263, so notebook sync/execution and discoverability-link
  validation were intentionally not part of the shipped work.
- **P3 shipped for the `.py` artifact only:** final validation scope was the
  runnable example source and its focused tests, plus review for CPU-only
  language and absence of GPU parity claims.

### Commands

```bash
python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py
ruff check docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py --fix
ruff format docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py
pytest particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py -q -Werror
```

Notebook commands remain deferred because no paired `.ipynb` shipped in this
issue.

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
- A zero-energy result is acceptable only if explicitly explained by a no-transfer
  setup; the preferred example should choose conditions that produce a visible
  non-zero diagnostic.
