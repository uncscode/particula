Break the feature into concrete edits that stay close to one reviewable PR per
phase and name the exact files or methods being changed.

### Backend
- [x] Add `CondensationLatentHeatBuilder` in `particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py` with `__init__()`, `set_latent_heat_strategy()`, `set_latent_heat()`, optional-key-aware `set_parameters()`, and `build()` returning `CondensationLatentHeat`
- [x] Reuse the existing molar-mass, diffusion-coefficient, accommodation-coefficient, and `update_gases` mixins instead of duplicating shared validation logic; keep latent-heat-specific validation isolated to the new builder file
- [x] Export `CondensationLatentHeatBuilder` from `particula/dynamics/condensation/condensation_builder/__init__.py`
- [ ] Register the factory key `"latent_heat"` in `particula/dynamics/condensation/condensation_factories.py` and ensure config dictionaries can pass either `latent_heat_strategy` objects or scalar `latent_heat` fallback values without changing `CondensationLatentHeat.__init__()` (~10-20 LOC)
- [ ] Re-export the builder from `particula/dynamics/condensation/__init__.py` and `particula/dynamics/__init__.py` so package-level imports stay symmetric with `CondensationIsothermalBuilder` (~10-20 LOC)

### Tooling / Tests
- [x] Add `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py` to verify successful builds, missing required-parameter failures, strategy passthrough, scalar validation, `update_gases` behavior, optional parameter loading, constructor precedence, and builder-package import/export behavior
- [ ] Extend `particula/dynamics/condensation/tests/condensation_factories_test.py` with a `"latent_heat"` factory case and one scalar-fallback case that asserts the result is `CondensationLatentHeat` and preserves `update_gases` (~25-40 LOC)
- [ ] Extend `particula/dynamics/tests/condensation_exports_test.py` to import `CondensationLatentHeatBuilder` from both namespaces and assert it appears in each `__all__` export list (~20-30 LOC)
- [ ] Run the touched condensation tests with `pytest -Werror particula/dynamics/condensation/tests/condensation_factories_test.py particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py particula/dynamics/tests/condensation_exports_test.py` after the API surface is wired

### Documentation
- [ ] Update `docs/Features/condensation_strategy_system.md` to list `CondensationLatentHeatBuilder` beside the other public builders and add a short builder/factory example that uses `CondensationFactory.get_strategy("latent_heat", ...)` (~20-40 LOC)
