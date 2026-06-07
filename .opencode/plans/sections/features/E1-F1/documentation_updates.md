`E1-F1` now includes the completed user-facing documentation sync for the
latent-heat builder and factory path:

- Added docstrings in
  `particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py`
  for the new public builder and its fluent setters/build path
- Updated the `CondensationFactory` support list docstring in
  `particula/dynamics/condensation/condensation_factories.py` to include the
  final `"latent_heat"` key
- Re-exported `CondensationLatentHeatBuilder` through
  `particula.dynamics.condensation` and `particula.dynamics`
- Added export smoke-test coverage in
  `particula/dynamics/tests/condensation_exports_test.py` for both public
  namespaces, `__all__` membership, and object identity
- Updated `docs/Features/condensation_strategy_system.md` to list
  `CondensationLatentHeatBuilder` alongside the other public builders and to
  show the shipped `CondensationFactory.get_strategy("latent_heat", ...)` path
- Updated `.opencode/plans/sections/features/E1-F1/` to reflect that the docs
  phase is complete and the plan now matches the shipped documentation state
