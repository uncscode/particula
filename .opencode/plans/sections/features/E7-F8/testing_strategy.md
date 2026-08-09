# Testing Strategy

Every production phase ships with co-located `*_test.py` tests. Coverage
thresholds are not lowered; changed modules must retain at least 80% coverage.

## Per-Phase Tests

- **P1 (complete, Issue #1520):** `particula/execution/tests/rng_test.py` covers
  frozen carriers, strict UTF-8 IDs, integral metadata, canonical manifests,
  lane permutations, collisions, immutable lookup, FNV vectors, and stable-ID
  reorder/unrelated-ID invariance. A subprocess import hook proves host-only
  registration does not load Warp or `particula.gpu`; Warp-guarded rows cover
  caller-buffer identity, initialization, and schema preflight preservation.
  Execution export tests deny the direct-only module from package and top-level
  export surfaces.
- **P2:** Brownian adapter tests assert one setup initialization, identical
  caller/session-owned array identity, repeated advancement without reseeding,
  no-op handling, and validation-state preservation.
- **P3:** Wall-loss adapter/scheduler tests assert a distinct process stream,
  enabled-box consumption, disabled-box preservation, and existing kernel
  validation/failure behavior.
- **P4:** Registry/session tests cover complete and targeted reset, repeated-seed
  non-reset, lifecycle restrictions, immutable inspection metadata, faulted and
  finalized sessions, and no normal-step host readback.
- **P5:** `particula/execution/tests/rng_invariance_test.py` compares each enabled
  logical box against an isolated one-box reference after adding, disabling,
  removing, and permuting unrelated boxes for both stochastic processes.
- **P6:** `particula/execution/tests/checkpoint_test.py` compares an uninterrupted
  run with a checkpoint/restart split at several steps. Require exact RNG-state
  and same-backend stochastic-output equality, one explicit checkpoint sync,
  version/device/dimension/ID rejection, and no intermediate conversion.
- **P7:** Run `mkdocs build --strict`, documentation regression tests, export
  checks, and copy-pastable focused commands.

## Validation Matrix

- Warp CPU is the required stochastic baseline when Warp is installed; CUDA is
  optional and skips cleanly when unavailable.
- Re-run direct kernel tests in
  `particula/gpu/kernels/tests/coagulation_test.py` and
  `particula/gpu/kernels/tests/wall_loss_test.py` to prevent API or RNG regressions.
- Use exact comparisons only within the same backend/device class and frozen
  configuration. Cross-backend behavior uses existing statistical,
  conservation, and tolerance-based evidence; exact CPU/CUDA replay is excluded.
- Snapshot particle, gas, outputs, RNG arrays, counters, and identities around
  every expected pre-launch failure. Post-launch failures assert session faulting
  rather than rollback.
- Transfer spies reject `to_warp_*`, `from_warp_*`, `.numpy()`, or
  `wp.synchronize()` during normal scheduler steps.
