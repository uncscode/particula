# Phase Details

## Sequencing

E6-F5 and E6-F6 must ship first. Complete P1 through P3 before public APIs in
P4/P5, then run P6 conservation validation before P7 documentation.

- [x] **E6-F7-P1:** Freeze bounded nucleation strategy and scientific contract with unit tests
  - Issue: #1430 | Size: S | Status: Shipped
  - Goal: Shipped activation/kinetic potential-rate equations, SI units, validity gates, immutable injection/formation metadata, and fail-closed scalar behavior.
  - Files: `particula/dynamics/nucleation/nucleation_strategies.py`, `particula/dynamics/nucleation/tests/nucleation_strategies_test.py`
  - Tests: Equation fixtures, unit conversion, linear/quadratic scaling, zero/gate ordering, boundaries, scalar/record validation, overflow rejection, immutability, and no-export regression.

- [x] **E6-F7-P2:** Compute inventory-limited provisional source demand with unit tests
  - Issue: #1431 | Size: S | Status: Shipped
  - Goal: Shipped immutable CPU records that convert survival-adjusted potential events into one shared, gas-admitted count per box and provisional per-species demand before representation planning or mutation.
  - Files: `particula/dynamics/nucleation/particle_source.py`, `particula/dynamics/nucleation/tests/particle_source_test.py`
  - Tests: Per-box limiting species and deterministic ties, exact depletion, zero inventory/time/rate and zero boxes, read-only ownership/nonaliasing, validation/overflow/rounding-correction failures, diagnostics, and input snapshots.

- [x] **E6-F7-P3:** Integrate slot activation and exhaustion transaction with unit tests
  - Issue: #1432 | Size: S | Status: Shipped
  - Goal: Shipped unexported CPU `commit_particle_source`, which consumes immutable P2 records, stages E6-F5/E6-F6 work on a private particle copy, applies representative-volume scaling to existing particle and gas state, validates conservation, and atomically writes validated arrays.
  - Files: `particula/dynamics/nucleation/particle_source.py`, `particula/dynamics/nucleation/tests/particle_source_test.py`
  - Tests: Capacity, no-op, scaling, atomic rejection, record immutability, and package-export boundaries.

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
