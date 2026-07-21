# Phase Details

- [ ] **E6-F1-P1:** Freeze dilution semantics and validated rate helpers with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Document the CPU equation, units, accepted scalar/array inputs,
    nonnegative domain, no-op rules, and numerical update contract.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/tests/dilution_test.py`
  - Tests: scalar/array broadcasting, zero flow, zero concentration, invalid
    volume/flow/coefficient, NaN/Inf rejection, and input non-mutation.

- [ ] **E6-F1-P2:** Add particle and gas container dilution updates with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Apply one CPU dilution step to particle number concentration and gas
    mass concentration while preserving every non-concentration field.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/tests/dilution_test.py`
  - Tests: particle distribution variants, scalar/multi-species gas, expected
    concentration updates, exact no-ops, nonnegative outputs, object identity,
    and snapshots of mass, charge, density, volume, and metadata.

- [ ] **E6-F1-P3:** Add dilution strategy and substepped runnable with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add the process-level strategy and `Dilution` runnable, including
    `rate()`, `execute()`, substep splitting, and runnable composition.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/particle_process.py`,
    `particula/dynamics/tests/dilution_runnable_test.py`
  - Tests: direct strategy/runnable agreement, substep behavior, returned
    aerosol identity, particle-plus-gas updates, no-op behavior, and `|`
    composition with a test runnable.

- [ ] **E6-F1-P4:** Harden validation and publish CPU dilution exports with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Ensure invalid calls fail before mutation and expose the supported API
    through `particula.dynamics` and the top-level package path.
  - Files: `particula/dynamics/dilution.py`,
    `particula/dynamics/particle_process.py`, `particula/dynamics/__init__.py`,
    relevant import/validation tests
  - Tests: public import smoke tests; zero/negative/nonfinite time; invalid
    `sub_steps`; malformed shapes/types; and complete preflight immutability.

- [ ] **E6-F1-P5:** Update development documentation and CPU dilution example
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish units, construction, execution, invariants, downstream GPU
    reference status, and a runnable example without claiming GPU support.
  - Files: `docs/Features/`, `docs/Examples/`, relevant indexes, and API docs
  - Tests: execute the example, validate documentation links, and run focused
    dilution tests.
