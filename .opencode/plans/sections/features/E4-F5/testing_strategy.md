# Testing Strategy

Issue #1302 added focused Warp regression coverage in
`particula/gpu/kernels/tests/_condensation_test_support.py` for atomic rejection
of nonbinary, wrong-shape, wrong-dtype, wrong-device, and non-Warp partitioning
masks and supplied P2 sidecars. Snapshots cover particle mass, gas
concentration, vapor pressure, transfer/energy outputs, and caller sidecars.
Multi-box/multi-species tests verify exact-zero gated work and accumulated
transfer for disabled species and zero-concentration slots while enabled active
entries match an all-enabled control. `particula/gpu/tests/conversion_test.py`
covers binary partitioning restoration at the CPU↔Warp boundary.

Unit tests ship with every remaining phase. Use explicit fp64 fixtures and
compare against CPU inventory-limit semantics. Issue #1303 adds an independent
NumPy oracle in `particula/gpu/kernels/tests/_condensation_test_support.py` for
the direct helper: ample, insufficient, and zero gas; evaporation; mixed
positive/negative transfer; disabled/pre-gated species; inactive
zero-concentration slots; and independent multi-box/multi-species limits. It
also verifies supplied P2 sidecar identity, unchanged gas, repeatability,
finite/nonnegative particle mass, and atomic rejection of malformed proposals
or P2 sidecars.

Launch-spy coverage proves the successful public four-substep P1 path launches
none of the P2 kernels and leaves sentinel P2 sidecars unchanged. P2 tests are
Warp-CPU direct-helper tests; CUDA parity and public coupled conservation remain
P3--P4 work.

Conservation is asserted independently per box/species: concentration-weighted
particle gain equals gas loss to tight bookkeeping tolerance; both inventories
remain finite and nonnegative. Global sums are supplemental only. Exactly four
substeps must use updated gas, and caller-owned buffers must retain identity and
validate before mutation. Returned transfer and E4-F4 energy must use the exact
applied transfer.

The issue #1272 production hook and regression in
`particula/integration_tests/condensation_particle_resolved_test.py` land in the
same phase. Warp CPU is required when Warp is installed; CUDA runs the same
checks when available and otherwise skips cleanly. Physics parity tolerances are
reported separately from stricter conservation tolerances. Tests must detect
hidden host conversion/synchronization and preserve the public API contract.
