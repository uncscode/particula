# Documentation Updates

## Shipped Updates

- Current shipped state from issues #1188 and #1189 / E2-F2-P1+P2:
  - `EnvironmentData` exists at `particula.gas.environment_data`.
  - `particula.gas.EnvironmentData` is now the supported package import path.
  - `EnvironmentData` now exposes `n_boxes` and an independent `copy()`
    implementation.
  - Code-level docs in `particula/gas/environment_data.py` should reflect the
    shipped export and copy semantics.

- Issue #1190 / E2-F2-P3 updated `docs/Features/particle-data-migration.md`
  to:
  - document `EnvironmentData` as the shipped CPU owner of per-box
    `temperature`, `pressure`, and `saturation_ratio`;
  - keep simulation volume explicitly under `ParticleData.volume` rather than
    moving it into `EnvironmentData`;
  - preserve the current compatibility boundary that existing process APIs may
    still accept scalar `temperature` and `pressure` until later migrations.

- Issue #1190 / E2-F2-P3 updated
  `docs/Features/Roadmap/data-oriented-gpu.md` to:
  - mark the CPU `EnvironmentData` baseline as shipped rather than future work;
  - align ownership and mutation guidance with the feature migration guide; and
  - keep `WarpEnvironmentData`, CPU↔GPU conversion helpers, and runtime/kernel
    integration listed as downstream work.

## Remaining Optional Updates

- Add a short example snippet showing single-box and multi-box construction if
  docs already include container usage examples.
- Update `docs/Features/index.md` only if a new standalone feature page is
  created.

## Process State Guidance

Documentation now makes the following ownership rule explicit:

- processes read environment fields from `EnvironmentData` when they are
  migrated to the new API;
- processes may mutate environment fields only when the physical model owns
  that update, such as latent-heat temperature changes;
- unrelated process steps should treat environment state as read-only and
  return mutations through documented state transitions.

## Validation Expectations For This Docs Slice

- Default validation for behavior-claim changes is `mkdocs build --strict`
  when command execution is available to the builder.
- If the docs build cannot be run, fallback validation must be documented
  explicitly with the reason the command was unavailable.
- Fallback validation remains scoped to manual changed-section reread plus the
  claim-to-code/test checklist in
  `.opencode/plans/sections/features/E2-F2/testing_strategy.md`.
- No new product tests are required for this docs-only slice unless that
  checklist reveals a documented shipped behavior without existing regression
  support.
