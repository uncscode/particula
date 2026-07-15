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

Issue #1304 adds Warp-CPU wrapper-exported regressions for the public path:
small independent NumPy four-substep oracles, inventory-limited uptake and
mixed-sign transfer, exactly four P1/P2/total/gas sequences (including zero
time or proposal), coupled-next-proposal tracing, and empty/single-particle
boundaries. They separately assert finalized total, particle delta, weighted
gas delta, finite/nonnegative gas, optional energy, and supplied scratch
identity/reuse.

Atomic-preflight cases cover invalid primary state, sidecar metadata, and
ownership aliases before launch or mutation. Stale nonfinite work is accepted
and overwritten; a nonfinite fresh proposal fails before P2, gas, total,
particle, or energy mutation for that cycle. A later-cycle proposal failure is
explicitly partial: earlier completed cycles are retained, not rolled back.

Warp CPU is the required focused evidence; optional CUDA coverage skips cleanly
when unavailable. P4 retains the integration production-hook/CPU-parity gate.
