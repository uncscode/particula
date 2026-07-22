# Phase Details

## Sequencing

Complete P1 through P4 in order; P5 documents only the CPU contract and
evidence established by those completed phases.

- [x] **E6-F1-P1:** Freeze dilution semantics and validated rate helpers with unit tests
  - Issue: #1389 | Size: S | Status: Complete
  - Goal: Document the CPU equation, units, accepted scalar/array inputs,
    nonnegative domain, no-op rules, and numerical update contract.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/tests/dilution_test.py`
  - Delivered: validated/broadcast-capable coefficient and instantaneous-rate
    helpers plus module-scoped exact `get_dilution_step()`; no package export,
    containers, runnable, docs/examples, or GPU changes.
  - Tests: scalar/array broadcasting and return conventions, exact no-ops,
    invalid domains/types/shapes and `None`, NaN/Inf rejection, warning-clean
    extreme decay, input non-mutation, and package-surface boundary.

- [x] **E6-F1-P2:** Add particle and gas container dilution updates with unit tests
  - Issue: #1390 | Size: S | Status: Complete
  - Goal: Apply one CPU dilution step to particle number concentration and gas
    mass concentration while preserving every non-concentration field.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/tests/dilution_test.py`
  - Delivered: unexported concrete-only `dilute_aerosol()` with strict scalar
    validation, full candidate and representation-storage preflight, ordered
    commit, and rollback after an unexpected commit failure.
  - Tests: particle and scalar/multi-species gas updates, exact no-ops,
    underflow, scalar-boundary errors, preflight atomicity for particle, both
    gas groups, and converted storage, plus commit-recovery and identity/state
    retention.

- [x] **E6-F1-P3:** Add dilution strategy and substepped runnable with unit tests
  - Issue: #1391 | Size: S | Status: Complete
  - Goal: Add the process-level strategy and `Dilution` runnable, including
    `rate()`, `execute()`, substep splitting, and runnable composition.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/particle_process.py`,
    `particula/dynamics/tests/dilution_runnable_test.py`
  - Delivered: concrete-module-only `DilutionStrategy`, which validates its
    scalar coefficient and delegates to `dilute_aerosol()`, plus
    `Dilution(RunnableABC)`, which validates and evenly splits time before
    strategy calls without replacing the input aerosol.
  - Tests: direct strategy/runnable decay, rate delegation, P2 validation
    propagation, invalid substep/time rejection before calls or mutation,
    no-ops, large finite decay, identity under nonconforming strategies, and
    `|`/`RunnableSequence` ordering. No exports, GPU work, examples, or general
    user documentation were added.

- [x] **E6-F1-P4:** Harden validation and publish CPU dilution exports with tests
  - Issue: #1392 | Size: S | Status: Complete
  - Goal: Ensure invalid calls fail before mutation and expose the supported API
    through `particula.dynamics` and the `par.dynamics` path.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/particle_process.py`, `particula/dynamics/__init__.py`,
    relevant import/validation tests
  - Delivered: shared concrete preflight before writes; retained setter-failure
    rollback; concrete-only runnable preflight before its first substep; and
    unchanged generic custom-strategy delegation. Exported `DilutionStrategy`
    and `Dilution`, while retaining concrete-only helpers.
  - Tests: `dilution_test.py` and `dilution_runnable_test.py` cover malformed
    sources, storage, volume, candidate shapes, zero-duration/zero-coefficient
    validation ordering, rollback, and first-substep preflight;
    `dilution_exports_test.py` covers public imports, identity, `__all__`, and
    `par.dynamics` construction/execution.

- [ ] **E6-F1-P5:** Update development documentation and CPU dilution example
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish units, construction, execution, invariants, downstream GPU
    reference status, and a runnable example without claiming GPU support.
  - Files: `docs/Features/`, `docs/Examples/`, relevant indexes, and API docs
  - Tests: execute the example, validate documentation links, and run focused
    dilution tests.
