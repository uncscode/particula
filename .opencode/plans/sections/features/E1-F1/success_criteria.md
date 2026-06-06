- [ ] `CondensationLatentHeatBuilder` constructs `CondensationLatentHeat` with
  validated required parameters and explicit handling for latent-heat inputs.
- [ ] `CondensationFactory.get_strategy("latent_heat", params)` produces the
  same runtime strategy shape that the public builder creates.
- [ ] `CondensationLatentHeatBuilder` is importable from both
  `particula.dynamics.condensation` and `particula.dynamics`.
- [ ] Documentation updates describe the shipped builder/factory path and the
  required regression tests run warning-free with `pytest -Werror`.

**Metrics:**

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Builder test coverage | No dedicated latent-heat builder test file | New builder tests cover required setters, missing-parameter failure, and latent-heat input paths | `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py` |
| Factory support | No public latent-heat factory registration | One stable supported key with passing regression coverage | `particula/dynamics/condensation/tests/condensation_factories_test.py` |
| Export stability | No public builder export | Builder appears in package import smoke tests and `__all__` checks | `particula/dynamics/tests/condensation_exports_test.py` |
| Docs/test alignment | Docs can drift from final API while feature is in flight | 0 known mismatches between docs examples and shipped import paths | Docs diff plus rerun of scoped `pytest -Werror` command from `phase_details.md` |
