**Inbound:**
- Existing condensation strategy infrastructure in
  `particula/dynamics/condensation/condensation_strategies.py` provides the
  stable extension point for non-isothermal behavior.
- Existing builder and factory infrastructure in
  `particula.abc_builder`, `particula.abc_factory`, and
  `particula/dynamics/condensation/condensation_builder/` defines how new
  public constructors should be wired.
- Existing latent heat strategy, builder, and factory support in
  `particula/gas/latent_heat_*.py` and `particula/gas/__init__.py` supplies the
  parameterization objects the condensation layer should consume instead of
  duplicating latent-heat logic.
- Current test suites under `particula/dynamics/condensation/tests/` and
  `particula/dynamics/tests/condensation_exports_test.py` provide the regression
  harness for API compatibility, factory registration, and import stability.

**Outbound:**
- Public user code that constructs condensation strategies through
  `particula.dynamics` depends on this epic to expose a stable non-isothermal
  API that matches existing builder/factory conventions.
- Validation, documentation, and example work for non-isothermal condensation
  depends on this epic to define the canonical public entry points and expected
  diagnostics before notebooks and guides are updated.
- Future acceleration-readiness work depends on this epic to establish a Python
  reference implementation whose mass-transfer and latent-heat bookkeeping can
  be mirrored exactly in later GPU-oriented code.

**Sequencing:**
- Latent-heat parameter strategies in `particula.gas` must be stable before the
  condensation public API can rely on them.
- Core non-isothermal rate and step behavior must land before builder/factory
  exposure so public constructors do not point at incomplete strategy behavior.
- Public builder/factory registration should ship before documentation and
  example updates so docs reference the final import paths and strategy key.
- Validation, parity testing, and acceleration-readiness follow after the
  Python API is fixed, because they need a settled reference surface.

**Feature/phase dependency edges:**
- `E1-F1-P1 -> E1-F1-P2 -> E1-F1-P3 -> E1-F1-P4` is the only valid in-feature
  path for the currently scoped child plan. The builder must exist before the
  factory can instantiate it, and the factory surface must be stable before
  namespace exports and docs point to that surface.
- Epic milestone 1 maps to `E1-F1-P1`, milestone 2 maps to `E1-F1-P2`, and
  milestone 3 maps to `E1-F1-P3` plus `E1-F1-P4`; this keeps the epic timeline
  aligned with the feature checklist and avoids forward references in the
  milestone table.
- Epic follow-on validation, documentation breadth, and acceleration-readiness
  work must remain downstream of `E1-F1-P4` or be split into additional child
  plans. The current epic should not imply those later tracks can start before
  the latent-heat builder/factory/export API is settled.

**Review notes:**
- No dependency cycle is declared in the scoped epic documents. All explicit
  edges point from lower-numbered feature phases to later phases, so the
  dependency graph remains a DAG.
- Cross-document alignment currently holds between this epic map,
  `E1/milestones_timeline.md`, `E1-F1/dependencies.md`, and
  `E1-F1/phase_details.md`: all describe builder first, factory second,
  exports third, and docs last.
