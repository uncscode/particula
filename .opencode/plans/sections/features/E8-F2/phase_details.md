# Phase Details

- [ ] **E8-F2-P1:** Define validated resident enqueue plans and setup contract with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Introduce exact immutable prepared-plan records and one setup boundary
    that validates the resident request, E8-F1 lifecycle/signature, duration,
    node order, and retained resource identities before any capture begins.
  - Files: `particula/execution/resident_scheduler.py` or
    `particula/execution/resident_enqueue.py`,
    `particula/execution/tests/resident_enqueue_test.py`
  - Tests: Exact carrier types, complete canonical schedule, signature drift,
    lifecycle gates, duration agreement, read-only rejection, and no public
    exports.

- [ ] **E8-F2-P2:** Split state, thermodynamic, and diagnostic setup from device enqueue with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prepare update arrays, fixed thermodynamic refresh windows, and closed
    diagnostic registrations once, then enqueue only copies and kernels.
  - Files: `particula/execution/state_updates.py`,
    `particula/execution/thermodynamic_updates.py`,
    `particula/execution/diagnostics.py`, adjacent `tests/*_test.py`
  - Tests: Setup performs all schema/value/alias checks; enqueue preserves empty
    no-ops and operation order and performs no allocation, readback,
    synchronization, binding validation, or host freshness decision.

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
