# Phase Details

- [ ] **E7-F7-P1:** Define fixed-shape communication maps and validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add immutable edge/map, volume-update, mode, and resource-shape contracts with deterministic read-only validation.
  - Files: `particula/execution/communication.py`, `particula/execution/tests/communication_test.py`
  - Tests: Valid 1D and arbitrary-pair maps; malformed indices/shapes/dtypes/devices; duplicate/conflicting edges; transfer bounds; zero-edge no-op; unchanged state on rejection.

- [ ] **E7-F7-P2:** Implement per-box volume evolution and expansion with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Apply prescribed positive volumes and renormalize particle/gas concentrations from extensive inventories without changing particle masses or charge.
  - Files: `particula/gpu/kernels/communication.py`, `particula/gpu/kernels/tests/communication_test.py`
  - Tests: Expansion, compression, unchanged volume, invalid/overflow/underflow cases, identity preservation, and particle-plus-gas amount invariants.

- [ ] **E7-F7-P3:** Implement conservative gas advection and mixing with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Stage and synchronously apply prescribed gas transfers from pre-step volume and concentration ledgers.
  - Files: `particula/gpu/kernels/communication.py`, `particula/gpu/kernels/tests/communication_test.py`
  - Tests: One-way advection, bidirectional mixing, chains, closed-map conservation, explicit boundary loss/source accounting, order independence, and NumPy parity.

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
