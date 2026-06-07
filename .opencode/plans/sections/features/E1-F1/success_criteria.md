- [x] `CondensationLatentHeatBuilder` constructs `CondensationLatentHeat` with
  validated required parameters and explicit handling for latent-heat inputs.
- [x] `CondensationFactory.get_strategy("latent_heat", params)` produces the
  same runtime strategy shape that the public builder creates.
- [x] `CondensationLatentHeatBuilder` is importable from both
  `particula.dynamics.condensation` and `particula.dynamics`.
- [x] Export regression tests cover both public namespaces, `__all__`
  membership, and shared object identity with `pytest -Werror`.
- [x] Documentation updates describe the shipped builder/factory path.
- [x] Post-review fix pass verifies reused `CondensationLatentHeatBuilder`
  instances do not leak optional `update_gases` state between
  `set_parameters()` calls.

**Metrics:**

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Builder test coverage | No dedicated latent-heat builder test file | P1 shipped dedicated builder tests covering required setters, missing-parameter failure, latent-heat input paths, `update_gases`, and optional parameter handling | `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py` |
| Factory support | No public latent-heat factory registration | P2 shipped one stable supported key with regression coverage for passthrough, scalar fallback, precedence, and builder-error propagation | `particula/dynamics/condensation/tests/condensation_factories_test.py` |
| Export stability | No public builder export | Builder appears in package import smoke tests, both `__all__` surfaces, and cross-namespace identity checks | `particula/dynamics/tests/condensation_exports_test.py` |
| Docs/test alignment | Docs can drift from final API while feature is in flight | 0 known mismatches between docs examples and shipped import paths | Docs diff plus rerun of scoped `pytest -Werror` command from `phase_details.md` |
| Post-review builder reuse | Optional flags could persist when a builder is reused | Reused `set_parameters()` calls restore documented defaults for omitted optional keys | `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py` |
