# Implementation Tasks

## E6-F1-P1 — Semantics and Helpers

1. Record `alpha = Q/V` and `dc/dt = -alpha*c`, with SI units and accepted
   scalar/array broadcasting.
2. Freeze the finite-step concentration update and numerical-stability rule.
3. Apply repository validation to positive finite volume, nonnegative finite
   flow/coefficient/concentration, and compatible shapes.
4. Preserve existing helper return conventions and add regression/edge tests.

## E6-F1-P2 — Container Reference

1. Add a pure calculation path that derives both new concentrations before any
   write.
2. Update `ParticleRepresentation` concentration using its representation
   volume correctly; do not change per-particle/bin mass or distribution data.
3. Update scalar or multi-species `GasSpecies` through its supported setter;
   retain species metadata and partitioning configuration.
4. Assert nonnegative finite results and exact zero-flow/time no-ops.
5. Add before/after snapshots for all protected fields and identities.

## E6-F1-P3 — Strategy and Runnable

1. Implement the named CPU strategy in the dilution module with typed,
   Google-style documented rate and step methods.
2. Add `Dilution(RunnableABC)` in `particle_process.py` and delegate to the
   strategy for each validated substep.
3. Return the same `Aerosol`; keep particle and atmosphere container identity
   unless an existing public setter requires assignment.
4. Exercise direct strategy use, runnable use, substeps, and `RunnableSequence`.

## E6-F1-P4 — Validation and Exports

1. Centralize preflight so every error occurs before particle or gas writes.
2. Cover nonfinite/negative time, invalid `sub_steps`, malformed NumPy shapes,
   unsupported types, and invalid concentration state.
3. Export strategy and runnable through `particula.dynamics` and verify the
   normal `import particula as par` usage path.
4. Run focused tests, Ruff, and mypy on touched modules without changing
   coverage thresholds.

## E6-F1-P5 — Documentation

1. Add a concise CPU runnable example that builds an aerosol, executes
   dilution, and verifies particle and gas concentration changes.
2. Document units, invariants, no-op behavior, supported container shapes, and
   the E6-F2 downstream relationship.
3. Update documentation indexes and links; explicitly defer GPU/backend claims.
4. Execute the example and run docs validation plus focused dilution tests.
