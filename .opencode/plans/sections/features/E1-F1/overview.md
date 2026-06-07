**Problem Statement:** `CondensationLatentHeat` already existed, but callers did
not have the same validated builder and configuration-dictionary factory entry
points that are available for isothermal condensation. That gap blocked
consistent latent-heat construction through both fluent setup and
`CondensationFactory`.

**Value Proposition:** `E1-F1-P1` and `E1-F1-P2` now ship the core latent-heat
construction path: a dedicated builder plus factory registration under the final
`"latent_heat"` key. Callers can now build `CondensationLatentHeat` through the
same generic `StrategyFactoryABC` flow used by the existing condensation
strategies while preserving the builder-defined parameter contract.

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
