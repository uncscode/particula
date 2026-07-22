# Overview

## Problem Statement

Particula exposes `get_volume_dilution_coefficient()` and
`get_dilution_rate()` as CPU free functions, but it has no process-level CPU
reference that applies dilution consistently to an `Aerosol`. Consequently,
the downstream direct GPU dilution feature, E6-F2, has no stable strategy,
runnable, validation, or mutation contract against which to prove parity.

## Value Proposition

E6-F1 defines the authoritative CPU behavior for chamber dilution: particle
number concentration and gas mass concentration decrease according to the same
validated coefficient while particle mass, charge, density, distribution,
representation volume, gas metadata, temperature, and pressure remain
unchanged. A composable `Dilution` runnable gives users a normal Particula
process API and supplies E6-F2 with a deterministic NumPy oracle.

## Implementation Status

P1 is complete in issue #1389. `particula/dynamics/dilution.py` now provides
validated, broadcasting-capable coefficient and instantaneous-rate helpers and
the concrete-module-only exact `get_dilution_step()` helper. It establishes the
numerical contract without adding container mutation, a strategy/runnable,
package exports, user documentation, examples, or GPU support; those remain in
later phases.

P2 is complete in issue #1390. The same concrete module now provides the
unexported `dilute_aerosol(aerosol, coefficient, time_step)` reference
primitive. It validates finite nonnegative scalar inputs, applies the exact P1
decay to physical particle concentration and both atmosphere gas groups, and
uses representation volume only when storing the particle result. Candidates
and storage are preflighted before commit; an unexpected write failure rolls
back already-written concentrations. No strategy/runnable, package export,
user documentation, or GPU API was added.

P3 is complete in issue #1391. `DilutionStrategy` is concrete-module-only in
`particula/dynamics/dilution.py` and delegates each step directly to P2's
`dilute_aerosol()`. `Dilution(RunnableABC)` in
`particula/dynamics/particle_process.py` validates a positive integral
`sub_steps` value and a finite nonnegative total duration, then invokes that
strategy once per equal time slice while retaining the original `Aerosol`
identity. Regression coverage lives in
`particula/dynamics/tests/dilution_runnable_test.py`. P3 deliberately added no
package exports, GPU support, examples, or general user documentation.

P4 is complete in issue #1392. The concrete dilution path now completes its
particle, gas, volume, storage, and candidate-shape preflight before any
concentration write, including zero-duration and zero-coefficient calls.
`DilutionStrategy.step()` uses the same preflight, while
`Dilution.execute()` performs it once before its first substep only for a
`DilutionStrategy`; compatible custom strategies retain generic equal-substep
delegation and own validation/atomicity. Commit-failure rollback remains in
place. `DilutionStrategy` and `Dilution` are now supported exports from
`particula.dynamics` (and therefore `par.dynamics`); concrete helpers remain
module-only. Focused direct, runnable, and export regressions record this
boundary.

P5 is complete in issue #1393. It adds the deterministic public-API example
`docs/Examples/cpu_dilution.py`, canonical guide
`docs/Features/dilution_strategy_system.md`, and dilution discoverability
entries in `docs/Features/index.md`, `docs/Examples/index.md`, and
`docs/index.md`. `particula/tests/dilution_docs_test.py` provides hardware-free
regression coverage for example execution and isolation, public-only imports,
guide scope/contract wording, and resolving local documentation links. The
change is documentation-only: no dilution production behavior, exports, GPU or
backend work was added.

## User Stories

- As a simulation user, I want to compose dilution with existing runnables so
  that a chamber timestep does not require manual concentration bookkeeping.
- As a physics developer, I want explicit units and no-op behavior so that CPU
  dilution results are reproducible and physically interpretable.
- As a GPU developer, I want a tested CPU reference so that E6-F2 can establish
  scalar and per-box parity without inventing different semantics.

**Parent epic:** [E6](../../epics/E6/vision_problem.md) — GPU Process
Completeness. This feature is issue track T1 and is the required predecessor of
E6-F2; E6-F9 consumes it during integrated closeout.
