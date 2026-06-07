## Tests-with-feature policy

`E1-F1` changes public condensation construction APIs, so every implementation
phase must ship with same-PR test updates in the nearest module `tests/`
directory and keep the `*_test.py` naming pattern. Do not lower coverage to
land the builder/factory/export work.

## Per-phase coverage

- **E1-F1-P1 builder implementation:** shipped with
  `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py`
  covering successful builds, missing required-parameter failures,
  `latent_heat_strategy` passthrough, scalar `latent_heat` validation,
  `update_gases` propagation, optional `set_parameters()` keys, constructor
  precedence when both strategy and scalar inputs are set, and regression
  coverage that unset scalar latent heat is not forwarded as `None`.
- **E1-F1-P2 factory registration:** shipped with
  `particula/dynamics/condensation/tests/condensation_factories_test.py`
  coverage for `"latent_heat"` builder registration, strategy-object
  passthrough, optional `update_gases`, scalar latent-heat fallback,
  explicit-strategy precedence when both latent-heat inputs are supplied,
  builder error propagation, and the unchanged unknown-strategy failure path.
- **E1-F1-P3 package exports:** shipped with
  `particula/dynamics/tests/condensation_exports_test.py` coverage for import
  smoke tests from both `particula.dynamics.condensation` and
  `particula.dynamics`, `__all__` membership in both namespaces, and object
  identity assertions proving both public imports resolve the same
  `CondensationLatentHeatBuilder` class.
- **E1-F1-P4 documentation update:** shipped as a docs-only exception to adding
  a brand-new test file; the relevant consistency checks remain the existing
  condensation export/factory coverage plus doc-to-API alignment for
  `docs/Features/condensation_strategy_system.md`.

## Test types and assertions

- Prefer focused unit tests for builder setters and build-time validation.
- Use focused factory tests to exercise `latent_heat_strategy` objects, scalar
  `latent_heat` values, explicit-strategy precedence, and invalid-parameter
  failure propagation through the builder path.
- Keep at least one regression-style assertion that the new builder produces a
  `CondensationLatentHeat` object without changing existing isothermal factory
  behavior.
- Run tests with `pytest -Werror` so warning-producing paths fail before review;
  this was used for the shipped P1 builder test file, the shipped P2 factory
  coverage, and the shipped P3 export smoke tests.

## Validation commands

- `pytest -Werror particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py`
- `pytest -Werror particula/dynamics/condensation/tests/condensation_factories_test.py`
- `pytest -Werror particula/dynamics/tests/condensation_exports_test.py`

If phase scope expands beyond these files, update the nearest module tests in
the same phase rather than deferring coverage to a later PR.
