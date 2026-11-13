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

- [x] **E7-F7-P4:** Implement fixed-capacity particle transport with unit tests
  - Issue: #1510 | Size: S | Status: Shipped
  - Delivered: Concrete-only direct-Warp `particle_communication_step_gpu` and
    frozen `ParticleCommunicationBuffers` plan all movement from immutable
    pre-step particle state. They aggregate caller-owned source-debit and
    destination-credit ledgers, preserve full species-mass vectors and signed
    charge, use exact population matches or ascending pre-step free-slot
    reservations, and perform one gated commit only for a valid closed-map plan.
  - Files: `particula/gpu/kernels/communication.py`, `particula/gpu/kernels/tests/communication_test.py`
  - Tests: Co-located contract coverage for immutable planning, deterministic
    matching/free-slot selection, caller-owned ledgers and assignments,
    closed-map number/species-mass/charge conservation, and gated commits.
  - Boundary: No package/top-level export, resident scheduler integration,
    gas or volume mutation, `-1` endpoints, transfer/synchronization, fallback,
    resizing, compaction, or implicit activation was added.

- [x] **E7-F7-P5:** Integrate communication nodes with resident sessions and scheduler tests
   - Issue: #1511 | Size: S | Status: Shipped
   - Delivered: `CommunicationResources` pins one exact closed GAS or PARTICLES
     map, matching native work record, and optional final volumes. The concrete
     executor dispatches the native primitive by identity. The resident scheduler
     runs communication then volume evolution before the original ten nodes,
     invalidating saturation ratio only. Schema-v2 checkpoints restore optional
     communication resources into fresh identities; schema-v1 noncommunication
     restart remains supported.
   - Files: `particula/execution/gpu_resources.py`, `checkpoint.py`,
     `process_graph.py`, `thermodynamic_updates.py`,
     `resident_communication.py`, `resident_scheduler.py`, and adjacent tests.
   - Tests: Resource/nonaliasing and identity checks; canonical barrier order;
     saturation-only invalidation; GAS/PARTICLES dispatch and no-op isolation;
     schema-v1/v2 checkpoint validation; no-transfer/no-sync spies; and
     writer-path guard close/session faulting.

- [x] **E7-F7-P6:** Validate multi-box parity and conservation across prescribed cases
  - Issue: #1512 | Size: S | Status: Shipped
  - Delivered: Test-only independent NumPy `float64` parity/conservation evidence
    for direct communication primitives and the concrete resident executor. No
    production behavior or public API changed.
  - Goal: Establish issue #1451 evidence for independent boxes, 1D advection/mixing, expansion, and combined communication on Warp CPU with optional CUDA rows.
  - Files: `particula/gpu/tests/communication_parity_test.py`, `particula/execution/tests/multi_box_communication_test.py`
  - Tests: Independent immutable-prestate NumPy/CPU `float64` oracles,
    equivalent one-box and isolated-box metamorphic cases, padded 1D maps,
    edge-order permutations, complete direct closed/open work ledgers, sparse
    state, repeated steps, explicit `rtol=1e-12` and documented `atol`
    declarations, and clean CUDA skips.

- [x] **E7-F7-P7:** Update development documentation
  - Issue: #1513 | Size: XS | Status: Shipped | Completed: 2026-08-09
  - Delivered: Published direct gas, particle, and standalone volume-evolution
    ownership, accounting, conservation, capacity, failure, resident-ordering,
    restart, and deferred-scope boundaries, including the architecture outline
    and its ADR-018 communication reference. No runnable example was added;
    E7-F9 owns complete-loop publication.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`,
    `.opencode/guides/architecture/architecture_guide.md`,
     `.opencode/guides/architecture_reference.md`,
     `.opencode/guides/architecture/architecture_outline.md`, and
    `particula/execution/tests/gpu_resident_session_docs_test.py`.
  - Tests: Passed `pytest particula/execution/tests/gpu_resident_session_docs_test.py -q -Werror`,
    `pytest particula/tests/execution_selection_docs_test.py -q -Werror`, and
    `mkdocs build --strict`.
