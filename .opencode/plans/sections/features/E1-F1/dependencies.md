**Upstream:**
- `particula/dynamics/condensation/condensation_strategies.py` must already
  expose a stable `CondensationLatentHeat` implementation for the builder to
  wrap instead of duplicating strategy logic.
- `particula/gas/latent_heat_builders.py`,
  `particula/gas/latent_heat_factories.py`, and `particula/gas/__init__.py`
  provide the latent-heat strategy objects and public exports this feature is
  expected to accept and reference.
- Existing condensation builder mixins, `BuilderABC`, and
  `CondensationIsothermalBuilder` provide the pattern for parameter validation,
  method chaining, and `build()` behavior.

**Downstream:**
- `CondensationFactory` users depend on this feature to create non-isothermal
  strategies from configuration dictionaries instead of manual class wiring.
- Public imports in `particula.dynamics` and documentation/examples for
  non-isothermal condensation depend on this feature to expose the final builder
  and factory surface.
- Factory and export smoke tests depend on this feature to keep the public API
  stable as more validation, documentation, and acceleration-oriented work lands.

**Phase ordering notes:**
- Implement the builder first so constructor semantics are explicit and tested
  before they are exposed through the factory.
- Register the builder under the final factory key next, because factory tests
  should exercise the same parameters the builder already validates.
- Update namespace exports and smoke tests last in the feature so the public API
  only expands after the builder and factory behavior are stable.

**Explicit dependency edges:**
- `E1-F1-P1 -> E1-F1-P2`: `CondensationFactory.get_strategy(...)` cannot be
  wired until `CondensationLatentHeatBuilder.build()` and its parameter names
  are fixed, otherwise the factory would codify unstable constructor semantics.
- `E1-F1-P2 -> E1-F1-P3`: namespace exports should only expose the builder after
  factory registration lands, so public imports and smoke tests cover the same
  supported construction path users will rely on.
- `E1-F1-P3 -> E1-F1-P4`: documentation should be updated against the shipped
  import paths and final factory key, not an intermediate export layout.

**Integration points to keep aligned:**
- P1 and P2 both depend on the latent-heat strategy objects from
  `particula/gas/latent_heat_*.py`; if those inputs change, the builder setter
  names and factory parameter mapping must change together.
- P2 and P3 both touch the public API boundary in `particula.dynamics`; export
  smoke tests should verify that the registered factory path and namespace
  imports describe the same builder class.
- P4 must reference the final public factory key chosen for P2. If the naming
  decision changes, docs and examples must be updated in the same release as the
  factory registration to avoid cross-document drift.

**Review notes:**
- The scoped feature declares a linear dependency chain with no cycle. Any work
  that attempts to document or export the builder before P2 would be a forward
  reference to an unsettled public API.
