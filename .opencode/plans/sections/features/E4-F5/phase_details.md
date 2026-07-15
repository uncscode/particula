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

- [x] **E4-F5-P2:** Deterministic gas and particle inventory limits with unit tests
  - Issue: #1303 | Size: S | Status: Shipped (2026-07-14)
   - Delivered: private direct-test-only fp64 finalization of already P1-gated
     proposals: owned-mass evaporation bounds, ordered per-box/species demand
     and release accounting, and gas-plus-release uptake scaling. It mutates
     direct-helper particle masses only; public gas coupling is deferred.
   - Files: `particula/gpu/kernels/condensation.py` and
     `particula/gpu/kernels/tests/_condensation_test_support.py`
   - Tests: explicit fp64 NumPy oracle cases for ample/insufficient/zero gas,
     evaporation, mixed signs, inactive and pre-gated entries, and multi-box/
     species isolation; atomic preflight and P1 public-isolation regressions.

- [x] **E4-F5-P3:** Four-substep coupled gas mutation and conserved transfer with unit tests
   - Issue: #1304 | Size: S | Status: Shipped (2026-07-14)
   - Delivered: four fixed P1→P2 cycles apply finalized transfer to particles,
     deterministically couple its weighted opposite to gas, and use finalized
     total/energy accounting. Aggregate preflight and scratch ownership are
     atomic; later fresh-proposal failures retain earlier completed cycles.
   - Files: `particula/gpu/kernels/condensation.py`,
     `particula/gpu/kernels/tests/_condensation_test_support.py`, and
     `particula/gpu/kernels/tests/condensation_test.py`
   - Tests: four-cycle NumPy oracle/order, zero/empty/single-particle limits,
     atomic preflight/aliasing, fresh-proposal boundary, and scratch reuse

- [x] **E4-F5-P4:** Production-hook conservation regressions and backend parity
   - Issue: #1305 | Size: S | Status: Shipped (2026-07-14)
   - Delivered: regression-only concentration-weighted particle-plus-gas
     inventory evidence without changing the public hook, APIs, or four-substep
     behavior.
   - Files: `particula/integration_tests/condensation_particle_resolved_test.py`,
     `particula/gpu/kernels/tests/_condensation_test_support.py`, and
     `particula/gpu/kernels/tests/condensation_test.py`
   - Tests: CPU integration maps H2O and NH4HSO4 inventories independently and
     requires exact N2 invariance. A deterministic fp64 two-box public-hook case
     covers uptake, evaporation, disabled partitioning, zero gas, and
     zero-concentration slots; it checks per-box/species inventory at
     `rtol=1e-12, atol=1e-30` separately from CPU-oracle parity at
     `rtol=2e-10, atol=1e-30` on Warp CPU and guarded CUDA.

- [ ] **E4-F5-P5:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
   - Goal: Document gas ownership, limits, partitioning, and production-gate status.
   - Files: GPU roadmap and feature documentation
   - Tests: Documentation link and command/reference checks only; this is a
     docs-only exception to production-code test coverage.
