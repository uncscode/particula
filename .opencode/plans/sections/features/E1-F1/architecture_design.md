### High-Level Design
`E1-F1-P1` shipped only the builder slice of the broader feature. The new
`CondensationLatentHeatBuilder` lives in
`particula/dynamics/condensation/condensation_builder/` and wraps the existing
`CondensationLatentHeat` strategy without changing condensation physics,
factory registration, or broader package exports.

```text
User code / tests
  -> CondensationLatentHeatBuilder
    -> validates molar_mass, diffusion_coefficient, accommodation_coefficient
    -> optionally accepts latent_heat_strategy or positive scalar latent_heat
    -> optionally accepts update_gases
    -> forwards only explicitly provided latent_heat keyword
  -> CondensationLatentHeat(...)
```

### Data / API / Workflow Changes
- **Data Model:** No new persisted data model is required. The feature only adds
  constructor-time parameters that are forwarded into `CondensationLatentHeat`,
  which already owns runtime state such as latent-heat diagnostics.
- **API Surface:** Shipped API changes are limited to
  `CondensationLatentHeatBuilder` plus its export from the
  `condensation_builder` package. The builder mirrors existing condensation
  builder signatures, reuses shared mixins, adds latent-heat-specific setter
  validation, and preserves `CondensationLatentHeat.__init__()` precedence by
  forwarding both strategy and scalar inputs when both are set.
- **Workflow Hooks:** P1 only extends the builder test surface with a dedicated
  latent-heat builder test file. Factory coverage and broader namespace import
  smoke tests remain deferred to later phases.

### Security & Compliance
This feature does not introduce permissions, network calls, or external state,
so the main compliance concern is API safety and validation quality. Builder
setters now validate units and required parameters consistently with the rest of
particula, and the new latent-heat scalar setter rejects `None`, array-like,
non-finite, zero, and negative inputs before strategy construction.
