## Infrastructure Reuse

### Existing APIs to Reuse

- `particula.dynamics.CondensationLatentHeat` for latent-heat-aware CPU
  condensation strategy behavior.
- `particula.dynamics.CondensationFactory().get_strategy("latent_heat", ...)`
  as the preferred public factory entry point for docs.
- `particula.dynamics.CondensationLatentHeatBuilder` as an acceptable builder
  alternative if factory parameters become too verbose.
- `particula.gas.LatentHeatFactory().get_strategy("constant", ...)` to create a
  constant latent heat strategy with units.
- `particula.dynamics.MassCondensation` to run the strategy against an aerosol.
- `CondensationLatentHeat.last_latent_heat_energy` for per-step energy
  bookkeeping after `execute()`.

### Existing Example Patterns

- Follow Jupytext percent-style examples in
  `docs/Examples/Dynamics/Condensation/Condensation_1_Bin.py` and
  `docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.py`.
- Edit the `.py` source first, then sync `.ipynb` with
  `.opencode/tools/validate_notebook.py` if a notebook pair is created.
- Update `docs/Examples/Dynamics/index.md` with a discoverable condensation
  entry.

### Existing Tests and Validation Helpers

- Use direct script execution for the example.
- Use ruff on the example source.
- Use notebook sync/execution tools for paired notebooks.
- If any API behavior is touched, run focused condensation tests:
  `condensation_factories_test.py`, `condensation_latent_heat_builder_test.py`,
  `condensation_strategies_test.py`, and `condensation_exports_test.py`.

### Reuse Guardrails

- Prefer top-level public imports via `import particula as par` to match docs.
- Do not import `particula.gpu`.
- Do not duplicate condensation internals in the example; demonstrate the public
  runnable workflow instead.
