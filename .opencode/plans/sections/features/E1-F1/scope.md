`E1-F1` is a phased feature for latent-heat condensation construction support.
The shipped `E1-F1-P1` slice is intentionally narrow: it adds the standalone
`CondensationLatentHeatBuilder`, exports it from the
`particula.dynamics.condensation.condensation_builder` package, and adds
focused builder tests without widening the public factory or top-level import
surface yet.

**In scope:**
- `particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py`
- Builder support for required shared transport inputs plus optional
  `latent_heat_strategy`, optional validated scalar `latent_heat`, and optional
  `update_gases`
- Builder-package export in
  `particula/dynamics/condensation/condensation_builder/__init__.py`
- Focused tests in
  `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py`

**Out of scope:**
- `CondensationFactory` registration and builder-union updates
- Re-exports from `particula.dynamics.condensation` or `particula.dynamics`
- User-facing docs/examples beyond these plan-section updates
