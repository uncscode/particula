This epic should keep the non-isothermal condensation work split along existing
package boundaries rather than introducing a new top-level subsystem. Pure
thermodynamic helpers belong in `particula/dynamics/condensation/mass_transfer.py`,
runtime strategy behavior belongs in
`particula/dynamics/condensation/condensation_strategies.py`, public
construction surfaces belong in
`particula/dynamics/condensation/condensation_builder/` and
`condensation_factories.py`, and species-specific latent-heat parameterization
stays in `particula/gas/` beside the existing vapor-pressure strategy family.

The architectural goal is to make latent-heat-aware condensation feel like a
natural extension of the current `CondensationStrategy` and `MassCondensation`
workflow. Child features should reuse the established strategy, builder, and
factory patterns already used in condensation, vapor pressure, wall loss, and
coagulation so the public API remains consistent through `particula.dynamics`
and `particula.gas` re-exports. Validation and diagnostics should remain close
to public constructors and pure helper functions, while stateful step execution
continues to live in strategy objects that work with existing particle/gas data
paths.

Data ownership should stay explicit: gas latent-heat strategies own latent-heat
parameter evaluation, condensation strategies own rate and step orchestration,
and tests own regression proof for parity with isothermal behavior, mass
conservation, and public import stability. Documentation and acceleration-readiness
work should build on the same Python reference implementation instead of adding
alternate physics paths first, because the architecture reference requires GPU
or future acceleration code to match the Python/NumPy behavior.

Testing requirements for every child feature in this epic are:
1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)
