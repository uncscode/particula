# Phase Details

- [x] **E7-F7-P1:** Define fixed-shape communication maps and validation with unit tests
  - Issue: #1507 | Size: S | Status: Shipped
  - Delivered: Concrete-only, unexported `particula.execution.communication`
    declarations and a sole read-only Warp validation boundary. It retains
    caller arrays by identity and validates resource metadata, schemas,
    dimensions/device, aliases, enabled/rate/volume domains, topology, and
    duplicate directed edges without writing caller payloads.
  - Files: `particula/execution/communication.py`,
    `particula/execution/tests/communication_test.py`
  - Tests: Valid 1D/arbitrary-pair maps and all transport modes; malformed
    metadata/schemas/devices; range aliases; domains; topology/duplicates;
    zero-edge and all-disabled forms; identity and write-free rejection/success.
  - Boundary: P1 has no resident-primary or `time_step` input and explicitly
    defers population-dependent outbound-overdraw validation to P3.

- [x] **E7-F7-P2:** Implement per-box volume evolution and expansion with unit tests
  - Issue: #1508 | Size: S | Status: Shipped
  - Delivered: Concrete-only, device-resident
    `volume_evolution_step_gpu(particles, gas, final_volumes)` validates
    caller-owned final `(B,)` `wp.float64` m³ volumes plus all container
    schemas, aliases, and domains. It sets particle volume and rescales particle
    and gas concentrations by `old_volume / final_volume`, preserving extensive
    inventory, container/array identity, and protected fields. Equal-volume
    calls are write-free; rejected preflight leaves caller state unchanged, and
    rollback is not promised after the apply writer launches.
  - Files: `particula/gpu/kernels/communication.py`, `particula/gpu/kernels/tests/communication_test.py`
  - Tests: Co-located Warp contract coverage for expansion/compression, mixed
    box factors, unchanged-volume no-op, invalid schema/alias/domain and unsafe
    scale rejection, protected-state and identity preservation, and particle/gas
    extensive-inventory invariants.
  - Boundary: P2 is not communication transport or a session/scheduler node;
    it adds no package export, hidden transfer/synchronization, fallback,
    resizing, or protected-field mutation.

- [x] **E7-F7-P3:** Implement conservative gas advection and mixing with unit tests
  - Issue: #1509 | Size: S | Status: Shipped
  - Delivered: Concrete-only direct-Warp `gas_communication_step_gpu` and
    `GasCommunicationBuffers` stage immutable `concentration * volume` amounts,
    aggregate synchronous explicit-Euler in-domain and declared `-1` boundary
    transfers, reject aggregate overdraw, and commit gas concentration once.
    Caller-owned `(B, S)` work and source/sink accounting ledgers are validated
    and overwritten only for active work. The operation validates optional final
    volume metadata but changes neither volume nor particle fields.
  - Files: `particula/gpu/kernels/communication.py`, `particula/gpu/kernels/tests/communication_test.py`
  - Tests: Co-located immutable-ledger oracle, fan-in/permutation order
    independence, closed/open accounting, zero-time/all-disabled no-ops,
    aggregate-overdraw and metadata gating, resource-schema, and invalid-time
    coverage.
  - Boundary: No package export, scheduler/session integration, hidden transfer
    or synchronization, fallback, resize, volume update, particle mutation, or
    post-launch rollback was added.

- [ ] **E7-F7-P4:** Implement fixed-capacity particle transport with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Move prescribed particle population between boxes while preserving slot composition/charge and rejecting unrepresentable destination plans before commit.
  - Files: `particula/gpu/kernels/communication.py`, `particula/gpu/kernels/tests/communication_test.py`
  - Tests: Whole/partial population transport, free-slot selection, inactive slots, multiple edges, capacity exhaustion, no partial commit, number/species-mass/charge conservation, and identity stability.

- [ ] **E7-F7-P5:** Integrate communication nodes with resident sessions and scheduler tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Register reusable resources and execute communication/volume nodes at a canonical scheduler barrier with no normal-step transfer or synchronization.
  - Files: `particula/execution/session.py`, `particula/execution/scheduler.py`, `particula/execution/tests/session_test.py`, `particula/execution/tests/scheduler_test.py`
  - Tests: Capability/resource validation, canonical order, disabled-map isolation, stable identities, checkpoint/restart state, transfer spies, and post-launch session faulting.

- [ ] **E7-F7-P6:** Validate multi-box parity and conservation across prescribed cases
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Establish issue #1451 evidence for independent boxes, 1D advection/mixing, expansion, and combined communication on Warp CPU with optional CUDA rows.
  - Files: `particula/gpu/tests/communication_parity_test.py`, `particula/execution/tests/multi_box_communication_test.py`
  - Tests: Independent NumPy/CPU oracles, equivalent one-box metamorphic cases, closed/open ledgers, sparse state, repeated steps, tolerance declarations, and clean CUDA skips.

- [ ] **E7-F7-P7:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish supported maps, ownership, ordering, conservation, capacity, failure, and scope boundaries and update the E7 plan state.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/guides/`, `.opencode/plans/sections/features/E7-F7/`
  - Tests: `mkdocs build --strict`, documentation link/contract regressions, and example/reference validation where introduced.
