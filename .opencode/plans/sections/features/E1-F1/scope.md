`E1-F1` is a phased feature for latent-heat condensation construction support.
The shipped `E1-F1-P1` and `E1-F1-P2` slices now cover the builder and factory
layers: `CondensationLatentHeatBuilder` exists, it is exported from the
`particula.dynamics.condensation.condensation_builder` package, and
`CondensationFactory` registers the final `"latent_heat"` key without widening
top-level imports yet.

**In scope:**
- `particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py`
- Builder support for required shared transport inputs plus optional
  `latent_heat_strategy`, optional validated scalar `latent_heat`, and optional
  `update_gases`
- Builder-package export in
  `particula/dynamics/condensation/condensation_builder/__init__.py`
- Factory registration in
  `particula/dynamics/condensation/condensation_factories.py` using the generic
  `StrategyFactoryABC` path and builder-driven parameter handling
- Focused tests in
  `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py`
- Focused factory tests in
  `particula/dynamics/condensation/tests/condensation_factories_test.py`

**Out of scope:**
- Re-exports from `particula.dynamics.condensation` or `particula.dynamics`
- User-facing docs/examples beyond these plan-section updates
