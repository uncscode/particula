# Testing Strategy

Every production phase includes co-located fast tests in the same change. Test
files use `*_test.py`; the configured threshold remains at least 80% and must
never be lowered. Scientific expectations use hand calculations or an
independent E6-F7 float64 oracle, never the production GPU helper itself.

## Per-Phase Approach

- **P1:** `particula/gpu/kernels/tests/nucleation_test.py` covers config and
   sidecar shape/dtype/device/overlap rules, scientific validation order,
   invalid counts, no-op gates, and byte-for-byte preflight snapshots. This
   suite shipped with #1438; it is Warp-guarded, uses Warp CPU fixtures, and
   verifies that valid preflight, gates, and rejections do not mutate caller
   state or stale sidecars. P1 has no rate or output-write assertions because
   those operations are deferred.
- **P2:** Shipped co-located fixtures compare survival-included `J=A*C` and
   `J=K*C^2`, potential/accepted demand, each limiting species and lowest-index
   ties, planned removal, and diagnostics with an independent float64 oracle.
   They cover one/many boxes and species, gate precedence, zero capacity and
   empty boxes, ULP-safe admission correction, and snapshots proving that P2
   leaves particle/gas state and P3-owned sidecars unchanged.
- **P3:** Slot interaction cases cover empty, sparse, exact-capacity, and mixed
  boxes; ascending indices, `-1` tails, exact integer counts, selected writes,
  and all preserved fields are asserted.
- **P4:** Exhaustion cases cover policy defaults and all combinations,
  resampling precedence, scaling fallback, insufficient scratch, impossible
  demand, residual rejection, and all-box failure snapshots.
- **P5:** Entry-point tests cover supplied-buffer and container identity,
  repeated calls, current-gas coupling, complete transaction ordering, lazy
  imports, explicit device inputs, and absence of CPU fallback/transfer.
- **P6:** `nucleation_parity_test.py` runs Warp CPU parity over activation and
  kinetic modes, multiple species/boxes, inventory limits, sparse/full slots,
  resampling/scaling cases, no-ops, and repeated calls. CUDA is optional and
  skips cleanly when unavailable.
- **P7:** Validate documentation links, citations, imports, equations, focused
  commands, and any explicit-transfer example.

## Required Invariants

- Without scaling, per-box/species represented particle plus gas mass is
  conserved. With scale `s`, final represented totals match `s * pre_total` and
  intensive concentration plus source transfer balance remain conserved at
  target `rtol=1e-12`, `atol=1e-30`. Aggregate-only checks are insufficient.
- Potential/admitted events and deterministic outputs match the CPU oracle at
  recorded float64 tolerances; gas remains finite and nonnegative.
- Zero time, coefficient, precursor, survival, and unsatisfied configured gates
  are exact no-ops with exact zero diagnostics where specified by E6-F7.
- Rejected calls preserve particles, gas, volume, diagnostics, requests,
  scratch/work buffers, shapes, dtypes, devices, and identities.
- Fixed capacity never causes silent truncation. Supplied sidecars are returned
  or retained by identity, and no test permits a hidden host fallback.
