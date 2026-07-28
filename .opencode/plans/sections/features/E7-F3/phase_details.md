# Phase Details

- [ ] **E7-F3-P1:** Map Brownian capabilities and validation semantics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Freeze CPU/Warp Brownian support, distribution/device constraints, and deterministic selection-level rejection order.
  - Files: `particula/execution/` package,
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
  - Boundaries retained: P2 carriers performed no dispatch or kernel import,
    transfer, synchronization, allocation, backend selection, physical/schema
    validation, or export change. Native kernel validation and actual RNG
    progression/reset remain P3/P4 work.
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

- [x] **E7-F3-P4:** Validate the Brownian adapter failure boundary with unit tests
  - Issue: #1480 | Size: S | Status: Completed 2026-07-28
  - Delivered: Exact-type, fieldless `BrownianCoagulationConfig` marker
    validation now rejects all non-Brownian request-shaped objects before CPU
    dispatch, optional Warp import, or lazy native-kernel resolution. Warp P2
    validates selected finite, nonnegative time while retaining native ownership
    of physical and resident-array schemas. CPU validates execution controls
    before one runnable call; Warp resolves/calls once and propagates native and
    post-launch failures without recovery or rollback.
  - Files: `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`
  - Tests: Marker-first CPU/Warp rejection and no-Warp subprocess coverage;
    ordered Warp P2 rejection/no-mutation snapshots; CPU control rejection;
    and one-call native/pass-through/writer-failure coverage

- [x] **E7-F3-P5:** Prove parity stochastic behavior conservation and persistent RNG integration
  - Issue: #1481 | Size: S | Status: Completed 2026-07-28
  - Delivered: Test-only evidence for the concrete-only CPU and resident-Warp
    Brownian adapters. It separates CPU reference checks from Warp stochastic
    evidence and preserves the existing runtime ownership boundary.
  - Files: `particula/execution/adapters/coagulation.py`,
    `particula/execution/tests/coagulation_adapter_test.py`,
    `particula/execution/tests/coagulation_integration_test.py`
  - Tests: CPU reference and identity; one-/multi-box Warp resource identity,
    conservation, pair validity, inactive-slot preservation, and box isolation;
    fixed 100-trial acceptance bounds; persistent RNG advance and reset replay;
    optional CUDA invariants; no adapter conversion, restore, synchronization,
    or fallback

- [x] **E7-F3-P6:** Update development documentation
  - Issue: #1482 | Size: XS | Status: Shipped 2026-07-28
  - Delivered: Concrete-only selected-Brownian CPU/Warp boundaries, exact
    import path, thermo forms, ownership, persistent RNG lifecycle, capability
    exclusions, and E7-F4/E7-F5/E7-F8 deferrals. The focused explicit-transfer
    example now dispatches through the selected Warp adapter.
  - Publication scope: This records completed P6 documentation and validation
    evidence only. It does not mark overall E7-F3 shipped or ship deferred
    E7-F4 resident sessions, E7-F5 scheduling, or E7-F8 checkpoint/restart.
  - Files: `docs/Features/coagulation_strategy_system.md`,
    `docs/Examples/gpu_coagulation_direct.py`,
    `particula/tests/backend_selected_coagulation_docs_test.py`, and E7-F3
    planning records
  - Tests: `pytest particula/tests/backend_selected_coagulation_docs_test.py -q -Werror`;
    `pytest particula/gpu/tests/gpu_coagulation_direct_example_test.py -q -Werror`;
    `pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror`;
    `mkdocs build --strict`
