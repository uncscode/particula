# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F5-P1:** Partitioning gates and gas-coupling validation with unit tests
  - Issue: #1302 | Size: S | Status: Shipped (2026-07-14)
  - Delivered: atomically validate active-device binary `(n_boxes, n_species)`
    `wp.int32` masks and supplied P2 sidecar metadata before mutable work;
    strictly zero disabled-species and inactive-slot raw proposals before
    application; preserve the particle-only no-gas-mutation contract.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`, and
    `particula/gpu/tests/conversion_test.py`
  - Tests: mask shape/device/dtype/binary-value and P2-sidecar atomicity,
    disabled/inactive gate behavior, and CPU↔Warp partitioning conversion

- [ ] **E4-F5-P2:** Deterministic gas and particle inventory limits with unit tests
  - Issue: TBD | Size: S | Status: Not Started
   - Goal: Finalize transfer through per-particle bounds and fp64 per-box/species
     reduction/scale buffers; limit this new limiting pipeline to roughly 100
     production LOC before tests.
   - Files: `particula/gpu/kernels/condensation.py` (private reduction, scale,
     and apply launches) and `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: insufficient gas, evaporation, mixed signs, zero concentration, multi-box isolation

- [ ] **E4-F5-P3:** Four-substep coupled gas mutation and conserved transfer with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Mutate gas and particles from one transfer and refresh current gas each substep.
   - Files: `particula/gpu/kernels/condensation.py` (`condensation_step_gpu()`
     four-substep orchestration) and `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: exactly four steps, scratch reuse, finalized return, latent-energy coupling

- [ ] **E4-F5-P4:** Production-hook conservation regressions and backend parity
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Land the production hook with issue #1272's same-change regression gate.
   - Files: `particula/integration_tests/condensation_particle_resolved_test.py`,
     `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: per-box/species conservation, CPU parity, Warp CPU, optional CUDA

- [ ] **E4-F5-P5:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
   - Goal: Document gas ownership, limits, partitioning, and production-gate status.
   - Files: GPU roadmap and feature documentation
   - Tests: Documentation link and command/reference checks only; this is a
     docs-only exception to production-code test coverage.
