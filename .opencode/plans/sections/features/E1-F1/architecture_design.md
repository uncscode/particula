### High-Level Design
`E1-F1-P1` through `E1-F1-P3` now ship the builder, factory, and public-export
slices of the feature. `CondensationLatentHeatBuilder` lives in
`particula/dynamics/condensation/condensation_builder/`,
`CondensationFactory` registers it under `"latent_heat"` through the existing
generic `StrategyFactoryABC.get_strategy()` flow, and the same builder class is
now re-exported from both `particula.dynamics.condensation` and
`particula.dynamics`.

```text
User code / tests
  -> CondensationLatentHeatBuilder
    -> validates molar_mass, diffusion_coefficient, accommodation_coefficient
    -> optionally accepts latent_heat_strategy or positive scalar latent_heat
    -> optionally accepts update_gases
    -> forwards only explicitly provided latent_heat keyword
  -> CondensationFactory.get_strategy("latent_heat", params)
    -> looks up CondensationLatentHeatBuilder in get_builders()
    -> passes the config dictionary through unchanged
    -> relies on builder validation / build() behavior
  -> CondensationLatentHeat(...)
```

### Data / API / Workflow Changes
- **Data Model:** No new persisted data model is required. The feature only adds
  constructor-time parameters that are forwarded into `CondensationLatentHeat`,
  which already owns runtime state such as latent-heat diagnostics.
- **API Surface:** Shipped API changes now include
  `CondensationLatentHeatBuilder`, its `condensation_builder` package export,
  public re-exports from `particula.dynamics.condensation` and
  `particula.dynamics`, and `CondensationFactory` support for `"latent_heat"`.
  The factory does not add special latent-heat branching; it preserves the
  builder-defined parameter contract for strategy-object passthrough, scalar
  fallback, explicit-strategy precedence, and `update_gases` propagation.
- **Workflow Hooks:** The feature now has dedicated builder tests plus factory
  regression coverage for registration, passthrough, scalar fallback,
  precedence, builder-error propagation, and the unchanged unknown-strategy
  failure path, plus export smoke tests for namespace imports, `__all__`
  membership, and cross-namespace builder identity.

### Security & Compliance
This feature does not introduce permissions, network calls, or external state,
so the main compliance concern is API safety and validation quality. Builder
setters now validate units and required parameters consistently with the rest of
particula, and the new latent-heat scalar setter rejects `None`, array-like,
non-finite, zero, and negative inputs before strategy construction.
