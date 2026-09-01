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

- [ ] **E8-F2-P3:** Add capture-ready communication and volume enqueue paths with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Resolve GAS/PARTICLES mode, exact closed-map resources, duration, and
    optional final-volume behavior in setup, leaving a fixed device-only barrier
    sequence.
  - Files: `particula/execution/resident_communication.py`,
    `particula/gpu/kernels/communication.py`,
    `particula/execution/tests/resident_communication_test.py`
  - Tests: Both closed modes, absent/present volume evolution, empty/no-op maps,
    identity preservation, canonical order, and no P1 scan or host work during
    prepared enqueue.

- [ ] **E8-F2-P4:** Add prepared condensation device enqueue path with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Refactor the authoritative fixed-four-substep condensation launch
    sequence behind a validated prepared record without changing public direct
    API behavior or inventory-finalization physics.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/execution/adapters/condensation.py`, condensation and adapter
    `*_test.py` files
  - Tests: Public validation order and atomic preflight, prepared sidecar
    identity, exact four-cycle launch trace, uptake/evaporation/no-op behavior,
    and no prepared-path allocation, host refresh, or readback.

- [ ] **E8-F2-P5:** Add prepared coagulation, dilution, and wall-loss enqueue paths with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Bind Brownian controls, dilution inputs, wall-loss geometry and selected
    lanes, and persistent RNG resources during setup, then expose fixed private
    enqueue sequences.
  - Files: `particula/gpu/kernels/{coagulation,dilution,wall_loss}.py`,
    `particula/execution/adapters/coagulation.py`,
    `particula/execution/process_adapters.py`, adjacent tests
  - Tests: Numerical/direct-wrapper regression, selected/all/empty wall-loss
    lanes, RNG advances without reseeding, zero-work behavior, stable output
    identities, and forbidden-host-operation spies.

- [ ] **E8-F2-P6:** Add prepared nucleation and exhaustion enqueue path with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Resolve fixed-capacity source, admission, exhaustion policy, resampling
    or scaling controls, and all sidecars before enqueueing the existing P1-P5
    device sequence.
  - Files: `particula/gpu/kernels/{nucleation,exhaustion}.py`,
    `particula/execution/process_adapters.py`, nucleation/exhaustion tests
  - Tests: No-admission no-op, free-slot activation, resampling-first and scaling
    fallback paths, conservation, failure gating, fixed shapes, and no enqueue
    allocation/readback/automatic policy change.

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
