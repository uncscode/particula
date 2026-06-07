**Problem Statement:** `CondensationLatentHeat` already existed, but callers did
not have the same validated builder and configuration-dictionary factory entry
points that are available for isothermal condensation. That gap blocked
consistent latent-heat construction through both fluent setup and
`CondensationFactory`.

**Value Proposition:** `E1-F1` now ships the complete latent-heat construction
path end to end: a dedicated builder, factory registration under the final
`"latent_heat"` key, symmetric public imports from both
`particula.dynamics.condensation` and `particula.dynamics`, and feature
documentation that shows the public builder/factory workflow. Callers can now
build `CondensationLatentHeat` through the same generic `StrategyFactoryABC`
flow used by the existing condensation strategies and find matching docs for
the shipped API surface.

**User Stories:**
- As a developer, I want to construct `CondensationLatentHeat` through a fluent
  builder so latent-heat-specific inputs follow the same validated setup path as
  existing condensation strategies.
- As a developer, I want `CondensationFactory.get_strategy("latent_heat",
  params)` to accept either a `latent_heat_strategy` object or scalar
  `latent_heat` fallback so configuration dictionaries can build the strategy
  without custom factory logic.
- As a maintainer, I want builder-specific tests for latent-heat inputs so
  scalar validation and strategy passthrough behavior stay stable as follow-on
  factory/export phases land.
- As a user, I want `CondensationLatentHeatBuilder` importable from both public
  dynamics namespaces so latent-heat usage matches the existing condensation
  builder import surface.
