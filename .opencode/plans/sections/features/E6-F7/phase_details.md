# Phase Details

- [ ] **E6-F7-P1:** Freeze bounded nucleation strategy and scientific contract with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Define activation/kinetic equations, SI units, validity gates, injection composition, citations, and fail-closed behavior.
  - Files: `particula/dynamics/nucleation/nucleation_strategies.py`, `particula/dynamics/nucleation/tests/nucleation_strategies_test.py`
  - Tests: Equation fixtures, unit conversion, linear/quadratic scaling, no-op gates, boundaries, and out-of-domain rejection.

- [ ] **E6-F7-P2:** Compute inventory-limited provisional source demand with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Convert potential events to immutable gas-admitted demand records and jointly cap all species by gas availability before representation planning or mutation.
  - Files: `particula/dynamics/nucleation/particle_source.py`, `particula/dynamics/nucleation/tests/particle_source_test.py`
  - Tests: Limiting species, exact depletion, zero inventory/time/rate, represented weight, diagnostics, and input snapshots.

- [ ] **E6-F7-P3:** Integrate slot activation and exhaustion transaction with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Plan E6-F5 activation and E6-F6 exhaustion, finalize any scaled demand, then atomically commit particle source and matching gas depletion.
  - Files: `particula/dynamics/nucleation/particle_source.py`, `particula/dynamics/nucleation/tests/particle_source_test.py`
  - Tests: Free/full/sparse slots, resampling-first and scaling fallback, provisional-to-represented demand diagnostics, policies-off failure, no final-domain residual, and failure atomicity.

- [ ] **E6-F7-P4:** Add nucleation builders factory and public strategy APIs with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Construct validated strategies/source configuration through repository-standard builders, factory, and stable exports.
  - Files: `particula/dynamics/nucleation/nucleation_builders.py`, `nucleation_factories.py`, package `__init__.py` files
  - Tests: Units/defaults, missing/invalid parameters, factory selection, imports, and unsupported aliases.

- [ ] **E6-F7-P5:** Add CPU nucleation runnable and substep behavior with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add `Nucleation`; each substep recomputes rate from current gas and commits one complete source transaction.
  - Files: `particula/dynamics/particle_process.py`, `particula/dynamics/tests/nucleation_runnable_test.py`, `particula/dynamics/__init__.py`
  - Tests: Delegation, substep duration, state coupling, identity, composition, zero-time no-op, and invalid substeps.

- [ ] **E6-F7-P6:** Validate multi-box multi-species conservation and failure atomicity
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove represented particle-plus-gas conservation and prepare the independent CPU oracle for E6-F8.
  - Files: nucleation tests and `particula/integration_tests/nucleation_process_test.py`
  - Tests: Independent oracle, limiting-species matrix, repeated calls, capacity cases, diagnostics, and preflight snapshots.

- [ ] **E6-F7-P7:** Update development documentation for CPU nucleation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish equations, citations, domains, APIs, conservation contract, dependencies, and deferred physics.
  - Files: `AGENTS.md`, `docs/Features/`, `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`, `docs/Examples/Nucleation/`, E6 sections
  - Tests: Links, snippets, equation/unit review, citations, applicable example execution, and focused commands.
