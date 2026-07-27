# Phase Details

These phases preserve issue #1451 T1's five suggested outcomes. The fifth
outcome is split into implementation/contract coverage and the required final
documentation phase. Unit tests are co-located with every production phase.

- [x] **E7-F1-P1:** Define the typed backend capability matrix with unit tests
  - Issue: #1462 | Size: S | Status: Completed
  - Goal: Model backend, device, process, and capability declarations and make
    support queries deterministic and side-effect free.
  - Files: `particula/execution.py`, `particula/tests/execution_test.py`
  - Delivered: frozen standard-library-only `Backend`, `Device`, `Process`,
    `Capability`, requirements/declaration values, and `CapabilityMatrix` in
    `particula/execution.py`; exact nonempty matching prevents composing
    separately declared capabilities, while empty requirements match a declared
    device/process base.
  - Tests: Constructor and stable-error validation, value hashing/frozen state,
    fixed-order matrix argument validation, pure `supports()`/`require()` lookup
    and exact unsupported errors, plus a fresh subprocess import guarded against
    `warp` and `particula.gpu`.

- [x] **E7-F1-P2:** Add the execution-context selection protocol and validation tests
  - Issue: #1463 | Size: S | Status: Completed
  - Goal: Select an adapter from explicit context configuration after complete
    backend/device/capability validation, with no implicit fallback.
  - Files: `particula/execution.py`, `particula/tests/execution_test.py`
  - Delivered: frozen `ExecutionRequest`, canonical CPU normalization, opaque
    Warp preservation, context-local private adapter registration, and exact
    capability-gated selection in `particula/execution.py`.
  - Tests: request/registration validation order, invalid combinations,
    matrix-before-single-lookup ordering, registry non-mutation and locality,
    exact identity selection, no execution/fallback, and guarded no-Warp import.

- [x] **E7-F1-P3:** Specify state ownership mutation and result contracts with unit tests
  - Issue: #1464 | Size: S | Status: Completed
  - Goal: Define typed execution state/result protocols that preserve caller
    ownership and state identity while making backend-specific payloads explicit.
  - Files: `particula/execution.py`, `particula/tests/execution_test.py`
  - Delivered: runtime structural `ExecutionState`/`ExecutionAdapter` protocols,
    closed mutation declaration values, opaque `BackendResult`, frozen
    `ExecutionResult`, and a nonexecuting identity-preserving validator in
    `particula/execution.py`.
  - Tests: protocol structure, exact immutable layouts, opaque identity,
    accepted/rejected mutation declarations, validation rejection/non-mutation,
    and P2 separation without adapter invocation.

- [ ] **E7-F1-P4:** Implement the CPU reference execution adapter with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Adapt existing `RunnableABC.execute()` without changing its process
    physics, substep behavior, `Aerosol` mutation, or return identity.
  - Files: `particula/execution.py`, `particula/tests/execution_test.py`
  - Tests: Delegation arguments, return/identity semantics, error propagation,
    no transfer calls, and representative runnable regression.

- [ ] **E7-F1-P5:** Publish deliberate API exports and contract regression tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Expose only the stable T1 selection types and lock the extension seam
    used by E7-F2, E7-F3, E7-F4, and E7-F6.
  - Files: `particula/__init__.py`, `particula/execution.py`,
    `particula/tests/execution_exports_test.py`
  - Tests: Exact public export list, import without Warp, typing surface,
    capability-extension registration, and no low-level kernel promotion.

- [ ] **E7-F1-P6:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish backend-selection, ownership, mutation, dependency, and
    unsupported-scope contracts for downstream maintainers.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/guides/`
  - Tests: `mkdocs build --strict` and documentation contract/link regressions.
