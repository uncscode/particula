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

- [x] **E7-F1-P4:** Implement the CPU reference execution adapter with unit tests
  - Issue: #1465 | Size: S | Status: Shipped
  - Goal: Adapt existing `RunnableABC.execute()` without changing its process
    physics, substep behavior, `Aerosol` mutation, or return identity.
  - Files: `particula/execution.py`, `particula/tests/execution_test.py`
  - Delivered: frozen `CPUExecutionState` and unexported
    `CPUExecutionAdapter` in `particula/execution.py`. The adapter exact-type
    validates state and finite nonnegative real time/positive integral substeps,
    makes one positional runnable call, retains state/aerosol identity in a
    `MutationScope.STATE` result, and rejects a replacement aerosol after that
    call.
  - Tests: exact delegation and identity, constructor non-inspection, concrete
    dilution regression, invalid-control zero-call preflight, NumPy scalar
    forwarding, exception/replacement propagation, zero-time dispatch, and
    fresh-process guarded absence of GPU imports.

- [x] **E7-F1-P5:** Publish deliberate API exports and contract regression tests
  - Issue: #1466 | Size: S | Status: Completed
  - Goal: Expose only the stable T1 selection types and lock the extension seam
    used by E7-F2, E7-F3, E7-F4, and E7-F6.
  - Files: `particula/__init__.py`, `particula/execution.py`,
    `particula/tests/execution_exports_test.py`
  - Delivered: exactly ten dependency-neutral names in
    `particula.execution.__all__` and top-level `particula` imports, plus typed
    context-local `ExecutionContext.register_adapter()` delegation. P3/P4 and
    GPU symbols remain excluded.
  - Tests: exact export identity/list, public registration validation,
    duplicate/non-mutation, context locality, static callable inspection and
    zero execution, guarded CPU-only subprocess import, P3/P4 exclusions, and
    runnable/direct-GPU compatibility.

- [x] **E7-F1-P6:** Update development documentation
  - Issue: #1469 | Size: XS | Status: Shipped
  - Goal: Publish backend-selection, ownership, mutation, dependency, and
    unsupported-scope contracts for downstream maintainers.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/guides/`
  - Delivered: Roadmap and architecture records document the shipped
    dependency-neutral P2--P5 surface and its deferred integration boundaries.
  - Tests: documentation contract/link regressions and `mkdocs build --strict`.
