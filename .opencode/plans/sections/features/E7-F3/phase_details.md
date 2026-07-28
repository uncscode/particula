# Phase Details

- [ ] **E7-F3-P1:** Map Brownian capabilities and validation semantics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Freeze CPU/Warp Brownian support, distribution/device constraints, and deterministic selection-level rejection order.
  - Files: `particula/execution.py`,
    `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: Capability queries, unsupported mechanisms/distributions, unavailable backend/device, no invocation on rejection

- [x] **E7-F3-P2:** Define coagulation resource carriers and persistent RNG ownership with unit tests
  - Issue: #1478 | Size: S | Status: Completed 2026-07-28
  - Delivered: Concrete-only frozen CPU and resident-Warp Brownian request/result
    carriers in `particula/execution/adapters/coagulation.py`. They retain
    caller resources by identity; apply kind, ownership-form, and
    metadata-detectable alias checks; and record persistent RNG seed/reset intent
    without mutation.
  - Boundaries retained: No dispatch/kernel import, transfer, synchronization,
    allocation, backend selection, physical/schema validation, or export change.
    Native kernel validation and actual RNG progression/reset remain P3/P4 work.
  - Files: `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: Focused CPU/Warp carrier identity, lazy-import, ownership/alias,
    no-mutation, and no-dispatch coverage

- [x] **E7-F3-P3:** Implement concrete-only backend-selected Brownian coagulation adapter with unit tests
  - Issue: #1479 | Size: S | Status: Completed 2026-07-28
  - Delivered: CPU P3 state/adapter makes one locally preflighted
    `Coagulation.execute()` call with original `time_step` and `sub_steps`.
    Resident-Warp P3 state/adapter lazily resolves and calls
    `coagulation_step_gpu()` once with retained P2 resources, diagnostics, and
    RNG intent. Both wrap typed P2 identity results in an `ExecutionResult`
    declaring `MutationScope.STATE`.
  - Boundaries retained: No capability selection, conversion, synchronization,
    fallback, retry, rollback, physics/schema change, export change, or public
    documentation change. Native kernel validation and unsupported-mode policy
    remain later work.
  - Files: `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: Spy-driven CPU control/one-call/identity/failure coverage and Warp
    direct/environment forwarding, lazy resolution, identity, RNG handoff,
    no-conversion/no-sync/no-fallback, and failure coverage

- [ ] **E7-F3-P4:** Add unsupported-mode and failure-boundary validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Fail closed for non-Brownian requests and preserve issue #1451 validation, preflight atomicity, and post-launch no-rollback semantics.
  - Files: `particula/execution.py`,
    `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: Charged/sedimentation/turbulent/combined rejection, invalid time/state/device, validation order, launch failure propagation

- [ ] **E7-F3-P5:** Prove parity stochastic behavior conservation and persistent RNG integration
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Provide bounded CPU-reference/Warp CPU evidence for Brownian rates and invariants while proving persistent RNG progression and explicit transfer boundaries.
  - Files: `particula/execution/tests/coagulation_adapter_test.py`,
    `particula/execution/tests/coagulation_integration_test.py`
  - Tests: One/multi-box cases, mass and charge conservation, inactive slots, stochastic bounds, reproducible reset, optional CUDA, no intermediate transfer

- [ ] **E7-F3-P6:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document selected Brownian usage, support matrix, outputs, persistent RNG ownership, errors, limitations, and E7-F5/E7-F8 handoffs.
  - Files: backend-selection feature guide, coagulation docs, focused example/API reference, plan status
  - Tests: Documentation regression, import snippets/example execution, `mkdocs build --strict`
