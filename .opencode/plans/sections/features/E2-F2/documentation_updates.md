# Documentation Updates

## Required Updates

- Current shipped state from issue #1188 / E2-F2-P1:
  - `EnvironmentData` exists at `particula.gas.environment_data`.
  - Direct-module import is the current supported path.
  - Package exports, `n_boxes`, and `copy()` are still pending later phases.

- `docs/Features/particle-data-migration.md`
  - Add `EnvironmentData` to the data-container overview.
  - Clarify that thermodynamic state belongs in `EnvironmentData`, not
    `GasData`.
  - Explain current process boundary: existing dynamics APIs may still accept
    scalar `temperature` and `pressure` until migration tracks update them.

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Mark the CPU `EnvironmentData` baseline as implemented once complete.
  - Keep future `WarpEnvironmentData`, conversion, and kernel integration work
    listed as downstream.

## Optional Updates

- Add a short example snippet showing single-box and multi-box construction if
  docs already include container usage examples.
- Update `docs/Features/index.md` only if a new standalone feature page is
  created.

## Process State Guidance

Documentation should make the following ownership rule explicit:

- processes read environment fields from `EnvironmentData` when they are
  migrated to the new API;
- processes may mutate environment fields only when the physical model owns
  that update, such as latent-heat temperature changes;
- unrelated process steps should treat environment state as read-only and
  return mutations through documented state transitions.
