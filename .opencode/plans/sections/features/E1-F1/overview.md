**Problem Statement:** `CondensationLatentHeat` already exists, but callers did
not have the same fluent builder entry point that is available for isothermal
condensation. That gap blocked consistent validated construction for latent-heat
condensation inputs.

**Value Proposition:** `E1-F1-P1` ships the first public construction step for
latent-heat condensation by adding a dedicated builder module, exporting it from
the `condensation_builder` package, and covering the builder contract with
focused tests.

**User Stories:**
- As a developer, I want to construct `CondensationLatentHeat` through a fluent
  builder so latent-heat-specific inputs follow the same validated setup path as
  existing condensation strategies.
- As a maintainer, I want builder-specific tests for latent-heat inputs so
  scalar validation and strategy passthrough behavior stay stable as follow-on
  factory/export phases land.
