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

**Complete in issue #1391.**

1. [x] Added concrete-module-only `DilutionStrategy` in
   `particula/dynamics/dilution.py`, with typed Google-style `rate()` and
   P2-delegating `step()` methods.
2. [x] Added `Dilution(RunnableABC)` in `particle_process.py`; it validates
   total time and positive integral non-boolean `sub_steps`, then delegates
   each equal substep to the strategy.
3. [x] Preserved the supplied `Aerosol` identity even for a nonconforming custom
   strategy return, while P2 retains particle and atmosphere container behavior.
4. [x] Added `particula/dynamics/tests/dilution_runnable_test.py` covering
   direct strategy/runnable use, substeps, validation ordering, no-ops, large
   finite decay, identity, and `RunnableSequence` composition.

## E6-F1-P4 — Validation and Exports

**Complete in issue #1392.**

1. [x] Centralized concrete-path preflight before every particle or gas write,
   retaining rollback for unexpected later setter failures.
2. [x] Added direct and runnable coverage for invalid scalar/state, storage,
   volume, and candidate-shape inputs, including zero-duration/zero-coefficient
   malformed state and supported-runnable preflight before its first substep.
3. [x] Preserved generic equal-substep delegation for compatible custom
   strategies without concrete-storage probing.
4. [x] Exported `DilutionStrategy` and `Dilution` through `particula.dynamics`
   and covered direct, `__all__`, and `import particula as par` import paths in
   `particula/dynamics/tests/dilution_exports_test.py`.
5. [x] Recorded focused validation in `dilution_test.py`,
   `dilution_runnable_test.py`, and `dilution_exports_test.py`; the final
   verification commands remain the focused pytest, Ruff, and mypy commands.

## E6-F1-P5 — Documentation

**Complete in issue #1393.**

1. [x] Added `docs/Examples/cpu_dilution.py`, a deterministic public-API
   runnable example that verifies exact decay for particle, partitioning-gas,
   and gas-only concentrations.
2. [x] Added `docs/Features/dilution_strategy_system.md`, documenting units,
   invariants, supported CPU shapes, substeps, validation/recovery/no-op
   behavior, and E6-F2 as a downstream consumer only.
3. [x] Added discoverability links in `docs/Features/index.md`,
   `docs/Examples/index.md`, and `docs/index.md`, using the absolute GitHub URL
   for the example source and retaining explicit GPU/backend/transport/inlet/
   performance non-goals.
4. [x] Added hardware-free `particula/tests/dilution_docs_test.py` coverage for
   execution, isolated snapshots, console output, imports, documentation text,
   and local link resolution; retained focused CPU dilution validation commands.
