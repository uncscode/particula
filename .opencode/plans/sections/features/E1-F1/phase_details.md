## Phase Checklist

- [x] **E1-F1-P1:** Add a fluent latent-heat condensation builder
  - Issue: #1157 | Size: S | Status: Shipped
  - Depends on: Stable `CondensationLatentHeat` behavior in `condensation_strategies.py` and existing latent-heat strategy inputs from `particula/gas/latent_heat_*.py`
  - Goal: Create `CondensationLatentHeatBuilder` so callers can construct
    `CondensationLatentHeat` with the same validated builder flow used by the
    isothermal strategies.
  - Sequencing: First implementation phase. Do not start P2 until builder
    setter names, validation behavior, and `build()` semantics are fixed by
    tests.
  - Files: Shipped `particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py`, `particula/dynamics/condensation/condensation_builder/__init__.py`, and `particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py`
  - Tests: Added coverage for required setters, `latent_heat_strategy` passthrough, scalar latent-heat acceptance/rejection, `update_gases` behavior, `set_parameters()` optional keys, constructor precedence, and unset-scalar default behavior

- [x] **E1-F1-P2:** Register the builder in `CondensationFactory`
  - Issue: #1158 | Size: S | Status: Shipped
  - Depends on: E1-F1-P1 shipped or at minimum functionally complete with final
    parameter names and passing builder tests
  - Goal: Make `CondensationFactory.get_strategy("latent_heat", params)`
    construct `CondensationLatentHeat` with the same parameter names the new
    builder validates.
  - Sequencing: Second phase. This is a forward-reference guardrail: do not wire
    the factory first, because the factory would otherwise freeze a config shape
    before the builder contract is stable.
  - Files: Shipped `particula/dynamics/condensation/condensation_factories.py` and `particula/dynamics/condensation/tests/condensation_factories_test.py`
  - Tests: Added factory coverage for `"latent_heat"` registration, latent-heat strategy-object passthrough, scalar `latent_heat` fallback, explicit-strategy precedence when both inputs are present, builder error propagation, and the unchanged unknown-strategy error path

- [ ] **E1-F1-P3:** Re-export the new builder through condensation namespaces
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E1-F1-P2 so exported namespaces point at the same supported
    builder/factory flow users can already construct through configuration
  - Goal: Expose the builder from both `particula.dynamics.condensation` and
    `particula.dynamics` so import paths match existing condensation builders.
  - Sequencing: Third phase. Export work stays downstream of factory
    registration so smoke tests validate the finished public surface instead of
    a partially wired API.
  - Files: `particula/dynamics/condensation/__init__.py` (~5-10 LOC), `particula/dynamics/__init__.py` (~5-10 LOC), `particula/dynamics/tests/condensation_exports_test.py` (~20-35 LOC)
  - Tests: Extend import smoke tests and `__all__` membership checks for `CondensationLatentHeatBuilder`

- [ ] **E1-F1-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E1-F1-P3 and final confirmation of the public factory key/name
    so docs do not publish stale import paths or configuration examples
  - Goal: Document the builder/factory entry point and show how latent-heat
    configuration differs from direct `CondensationLatentHeat(...)` usage.
  - Sequencing: Final phase. Documentation is intentionally last because it
    should snapshot the shipped API, not speculate on export paths or factory
    naming still under review. This phase is the required update dev-docs step
    before the feature can be considered complete.
  - Files: `docs/Features/condensation_strategy_system.md` (~20-40 LOC)
  - Tests: Docs-only exception to adding a new test file; keep example import
    paths aligned with `particula/dynamics/tests/condensation_exports_test.py`
    and rerun `pytest -Werror particula/dynamics/condensation/tests/condensation_factories_test.py particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py particula/dynamics/tests/condensation_exports_test.py` after doc/API edits

### Dependency Review Notes

- Declared phase graph: `P1 -> P2 -> P3 -> P4`.
- No intra-feature cycle is currently implied by the checklist.
- `E1-F1-P1` and `E1-F1-P2` are complete; P3 is now the next required step
  before the feature can expose broader public imports and documentation.
- Any attempt to start P3 or P4 before P2 resolves would create a forward
  reference to an incomplete public API surface.
