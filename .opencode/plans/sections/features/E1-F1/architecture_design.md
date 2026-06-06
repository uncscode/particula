### High-Level Design
This feature should expose latent-heat-aware condensation through the same
construction path users already follow for isothermal strategies: a builder in
`particula/dynamics/condensation/condensation_builder/`, factory registration
in `condensation_factories.py`, and re-exports from
`particula.dynamics.condensation` plus `particula.dynamics`. The builder should
wrap the existing `CondensationLatentHeat` strategy rather than reimplementing
physics, while consuming optional latent-heat strategy objects from
`particula.gas` or a constant latent-heat fallback value.

```text
User config
  -> CondensationLatentHeatBuilder
    -> validates molar_mass, diffusion_coefficient, accommodation_coefficient
    -> optionally accepts latent_heat_strategy or latent_heat constant
  -> CondensationFactory("latent_heat", params)
    -> builds CondensationLatentHeat
  -> particula.dynamics export surface
    -> MassCondensation / direct strategy use with existing particle+gas flows
```

### Data / API / Workflow Changes
- **Data Model:** No new persisted data model is required. The feature only adds
  constructor-time parameters that are forwarded into `CondensationLatentHeat`,
  which already owns runtime state such as latent-heat diagnostics.
- **API Surface:** Add `CondensationLatentHeatBuilder`, register the strategy in
  `CondensationFactory`, and ensure namespace exports make the builder and
  strategy importable from `particula.dynamics` and its condensation subpackage.
  The API should mirror existing condensation builder signatures so users can
  switch from isothermal to latent-heat-aware construction with minimal code
  changes.
- **Workflow Hooks:** This integrates with the existing builder/factory testing
  workflow by extending condensation factory tests and import smoke tests rather
  than creating new execution paths or developer tooling.

### Security & Compliance
This feature does not introduce permissions, network calls, or external state,
so the main compliance concern is API safety and validation quality. Builder
setters should validate units and required parameters consistently with the rest
of particula, and factory registration should fail deterministically on unknown
strategy types so invalid user configuration does not silently select the wrong
physics path.
