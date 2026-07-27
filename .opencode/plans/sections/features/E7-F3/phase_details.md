# Phase Details

- [ ] **E7-F3-P1:** Map Brownian capabilities and validation semantics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Freeze CPU/Warp Brownian support, distribution/device constraints, and deterministic selection-level rejection order.
  - Files: `particula/execution.py`,
    `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: Capability queries, unsupported mechanisms/distributions, unavailable backend/device, no invocation on rejection

- [ ] **E7-F3-P2:** Define coagulation outputs and persistent RNG ownership with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Model collision buffers, environment/volume inputs, seed-once/reset semantics, state identity, and asynchronous failure ownership.
  - Files: `particula/execution/adapters/coagulation.py`,
    `particula/execution/__init__.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: State schemas, output capacity/dtype/device, RNG identity/reuse/reset, alias and validation failures

- [ ] **E7-F3-P3:** Implement backend-selected Brownian coagulation adapter with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Route explicit CPU and Warp requests to existing implementations without hidden transfer, synchronization, fallback, or physics changes.
  - Files: `particula/execution.py`,
    `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: Dispatch arguments/counts, CPU substeps, particle/output identity, repeated Warp calls, no conversion or fallback

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
