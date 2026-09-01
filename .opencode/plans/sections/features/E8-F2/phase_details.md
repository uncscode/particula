# Phase Details

- [x] **E8-F2-P1:** Define validated resident enqueue plans and setup contract with unit tests
  - Issue: #1552 | Size: S | Status: Implemented
  - Delivered: Added concrete-only `particula.execution.resident_enqueue` with
    frozen `PreparedResidentTimestep` metadata and
    `prepare_resident_timestep()`. Preparation accepts only an exact READY E8-F1
    binding, retains exact resident identities and canonical schedule metadata,
    performs initial and final signature-drift checks, and does not construct an
    executor/scheduler or mutate lifecycle state.
  - Shared validation: extracted read-only diagnostics and communication
    validators and added scheduler complete-loop metadata validation for reuse by
    preparation while preserving scheduler CAPTURED admission and dispatch.
  - Files: `particula/execution/resident_enqueue.py`,
    `particula/execution/resident_scheduler.py`,
    `particula/execution/diagnostics.py`,
    `particula/execution/resident_communication.py`, and focused execution tests.
  - Evidence: `resident_enqueue_test.py` covers exact/frozen identity retention,
    READY/lifecycle/attachment/duration/schedule/signature rejection, including
    second-check drift, and traps forbidden guard-token entry. Broader
    forbidden-operation trap coverage remains with later enqueue phases. Adjacent diagnostics,
    communication, graph-capture, scheduler-loop, and export-boundary regressions
    cover the extracted validators and concrete-only export contract.

- [x] **E8-F2-P2:** Split state, thermodynamic, and diagnostic setup from device enqueue with unit tests
  - Issue: #1553 | Size: S | Status: Implemented
  - Delivered: Added concrete-only P1-bound prepared setup and enqueue seams in
    `state_updates.py`, `thermodynamic_updates.py`, and `diagnostics.py`.
    Setup validates exact request/plan, session, registry, graph, schedule,
    primary-array, and pinning identities; standalone executor/coordinator paths
    remain independent and compatible.
  - Behavior: State writes retain temperature-then-pressure ordering; valid
    empty state and diagnostic schemas are write-free. Thermodynamic common
    binding is immutable while per-consumer preparation reads the coordinator's
    current cursor/stale state, preserving vapor-then-saturation ordering and
    established freshness/failure behavior.
  - Evidence: Added adjacent validate-once/enqueue-only tests that exercise
    identity/pinning rejection, writer ordering, empty no-ops, and forbidden
    setup-time operations after successful preparation.

- [x] **E8-F2-P3:** Add capture-ready communication and volume enqueue paths with unit tests
  - Issue: #1554 | Size: S | Status: Implemented
  - Delivered: Added frozen P1-bound `PreparedResidentCommunicationBinding` and
    setup in `resident_communication.py`. Setup retains exact closed-map GAS or
    PARTICLES resources, primaries, duration, mode-specific work/status state,
    and optional final volumes, rejecting structural or identity drift before
    launch.
  - Enqueue: Added fixed native GAS, PARTICLES, and volume helper paths in
    `communication.py`. Prepared enqueue performs communication before a present
    volume barrier with no enqueue-time validation, lookup, allocation,
    transfer/readback, synchronization, or reacquisition. Equal final volumes
    are write-free; changed volumes retain resident status/scaling/update
    semantics. Legacy executor and direct helper behavior remain intact.
  - Files: `particula/execution/resident_communication.py`,
    `particula/gpu/kernels/communication.py`,
    `particula/execution/tests/resident_communication_test.py`, and
    `particula/gpu/kernels/tests/communication_test.py`
  - Evidence: Focused execution and native-helper tests cover both closed modes;
    absent, equal, and changed final volumes; empty/zero barriers; identity and
    setup rejection; communication-before-volume ordering; aggregate-overdraw
    gated commit; and forbidden enqueue-time operation traps.

- [x] **E8-F2-P4:** Add prepared condensation device enqueue path with unit tests
  - Issue: #1555 | Size: S | Status: Implemented
  - Delivered: Added private `_PreparedCondensationCall` setup/enqueue separation
    in `particula/gpu/kernels/condensation.py`. Setup preserves existing direct
    validation, fallback allocation, supplied-sidecar identity, and return
    contracts; enqueue retains the authoritative four equal gas-coupled,
    inventory-limited substeps without validation, allocation, host
    refresh/readback, synchronization, or resource lookup.
  - Adapter: Added private `_PreparedWarpCondensationBinding` in
    `particula/execution/adapters/condensation.py`, retaining the prepared kernel
    call for the exact concrete resident binding and delegating only to enqueue.
  - Compatibility: No public APIs, exports, scheduler behavior,
    checkpoint/resource schemas, or condensation physics changed.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/execution/adapters/condensation.py`, condensation and adapter
    `*_test.py` files
  - Documentation: Private Google-style docstrings record setup/enqueue
    ownership and failure boundaries; focused docstring validation covers the
    changed concrete modules without requiring a public-documentation update.

- [x] **E8-F2-P5:** Add prepared coagulation, dilution, and wall-loss enqueue paths with unit tests
  - Issue: #1556 | Size: S | Status: Implemented
  - Delivered: Added private frozen prepared calls and enqueue helpers in
    `particula/gpu/kernels/{coagulation,dilution,wall_loss}.py`. Public direct
    wrappers preserve validation, allocation, identity, no-op, RNG, selected-lane,
    and physics contracts while delegating through the private seam.
  - Resident adapters: Added `_PreparedResidentBrownianCoagulationBinding` and
    private process bindings that retain exact preflight results and invoke only
    their pinned enqueue delegates. Partial wall-loss setup freezes selected
    physical lanes, its device array, and the selected-box delegate before
    enqueue; enqueue does not redo setup or reread mutable references. Resident
    coagulation remains Brownian-only; dilution and wall loss retain exact
    primary and selected-lane behavior.
  - Files: `particula/gpu/kernels/{coagulation,dilution,wall_loss}.py`,
    `particula/execution/adapters/coagulation.py`,
    `particula/execution/process_adapters.py`, adjacent tests
  - Evidence: Focused kernel and adapter tests cover direct-wrapper/prepared
    delegation, frozen references, Brownian persistent RNG/no-reset behavior,
    dilution no-ops and identities, wall-loss all/partial/empty selections, and
    enqueue-only operation traps.
  - Documentation: No public documentation change was required; local private
    seam documentation records ownership and failure boundaries.

- [x] **E8-F2-P6:** Add prepared nucleation and exhaustion enqueue path with unit tests
  - Issue: #1557 | Size: M | Status: Implemented
  - Delivery: Added allocation-complete prepared resampling, representative-
    volume scaling, and nucleation records. Preparation retains legacy public
    validation precedence, pins all launch arrays, freezes controls, and owns
    private statuses and P1--P5/P4 workspace.
  - Enqueue behavior: Resets retained status and launches a fixed device-gated
    sequence over live pinned payload values. It performs no allocation,
    readback, synchronization, host policy resolution, mutable-carrier lookup,
    or public/legacy executor call. Rebinding cannot redirect dispatch.
  - Observation: Direct wrappers interpret bounded status after enqueue and
    retain legacy error ordering and return identities. Resident adapter
    execution remains observation-free. No-admission preserves particle/gas
    primaries while documented diagnostics may update.
  - Adapter: `ResidentNucleationAdapter` now pins exact resident identities in a
    private prepared process binding and delegates only to enqueue. It rejects
    drift before resolution or launch; its one-call public resolver fallback is
    limited to injected compatibility tests and does not alter scheduler flow.
  - Files: `particula/gpu/kernels/{nucleation,exhaustion}.py`,
    `particula/execution/process_adapters.py`, and adjacent nucleation,
    exhaustion, and process-adapter tests.
  - Evidence: Focused tests cover operation traps, pinned-array rebinding,
    dynamic writer gating, free-slot/resampling/scaling paths, error precedence,
    public-wrapper identities, independent parity, conservation, exact resident
    binding, and injected compatibility fallback.

- [ ] **E8-F2-P7:** Compose capture-ready resident sequence with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Enqueue the complete prepared twelve-node timestep in authoritative
    order under one lifecycle token and prove that no setup operation occurs in
    the capture window.
  - Files: `particula/execution/resident_scheduler.py`,
    `particula/execution/tests/{resident_enqueue,full_loop}_test.py`, optional
    focused CUDA capture test
  - Tests: Full launch trace, exact request/record identity, uncaptured output
    regression on Warp CPU, lifecycle completion/fault behavior, forbidden host
    calls, and CUDA capture pass-or-clean-skip smoke evidence.

- [ ] **E8-F2-P8:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish the implemented setup/enqueue boundary and handoffs without
    claiming E8-F4 parity or E8-F8 performance closeout.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`, `AGENTS.md`, E8 plan
    sections
  - Tests: Documentation contract assertions and `mkdocs build --strict`.
