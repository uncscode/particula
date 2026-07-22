# Implementation Tasks

## E6-F1-P1 — Semantics and Helpers

**Complete in issue #1389.**

1. [x] Record `alpha = Q/V` and `dc/dt = -alpha*c`, with SI units and accepted
   scalar/array broadcasting.
2. [x] Freeze `c_new = c * exp(-alpha * time_step)` as a module-scoped exact
   step helper with warning-clean extreme finite decay.
3. [x] Apply finite-domain validation, explicit `None` rejection, and broadcast
   preflight without mutating inputs.
4. [x] Preserve existing helper return conventions and package exports; add
   regression and edge-contract tests.

## E6-F1-P2 — Container Reference

**Complete in issue #1390.**

1. [x] Add `dilute_aerosol()` as an unexported concrete-module primitive with
   finite, nonnegative scalar coefficient and duration validation.
2. [x] Preflight physical particle and both gas-group candidates, then convert
   particle concentration through representation volume before writing storage.
3. [x] Commit particle then gas concentrations in declared order, retaining
   snapshots and rolling back already-written values on an unexpected failure.
4. [x] Preserve particle distribution state, gas metadata, atmosphere state,
   and container identities; retain exact zero-input no-ops and finite
   underflow to zero.
5. [x] Add regression coverage for normal behavior, boundaries, preflight
   atomicity, no-ops, underflow, and commit recovery.

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
