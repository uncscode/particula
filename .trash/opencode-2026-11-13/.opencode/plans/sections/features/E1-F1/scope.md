`E1-F1` is a phased feature for latent-heat condensation construction support.
The shipped `E1-F1-P1` through `E1-F1-P3` slices now cover the builder,
factory, and public-export layers: `CondensationLatentHeatBuilder` exists, it
is exported from the builder package plus both public dynamics namespaces, and
`CondensationFactory` registers the final `"latent_heat"` key.

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
- Public re-exports in `particula/dynamics/condensation/__init__.py` and
  `particula/dynamics/__init__.py`
- Focused tests in
  `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py`
- Focused factory tests in
  `particula/dynamics/condensation/tests/condensation_factories_test.py`
- Public export smoke tests in
  `particula/dynamics/tests/condensation_exports_test.py`

**Out of scope:**
- User-facing docs/examples beyond these plan-section updates
