# Success Criteria

- [x] E4-F1 thermodynamic refresh executes before transfer calculation in each
  of exactly four substeps (P2, issue #1293).
- [x] Every substep uses `time_step / 4` and reads current particle state (P2,
  issue #1293).
- [x] Complete fp64 `CondensationScratchBuffers` sidecars eliminate required
  allocation of the two transfer and two property stable shapes and preserve
  caller identity and shape (P1, issue #1292).
- [x] All supplied scratch fields validate before allocation, normalization,
  refresh, launch, clear, particle mutation, or caller-buffer mutation (P1,
  issue #1292).
- [x] Returned mass transfer is the accumulated applied transfer for the full
  call; particle mass remains finite and nonnegative (P2, issue #1293).
- [x] Repeated identical runs are deterministic and gas concentration is
  unchanged (P2, issue #1293).
- [x] Scalar, Warp-array, hybrid, and explicit environment input contracts pass
  fixed-four integration coverage (P2, issue #1293).
- [ ] Recorded nanometer, accumulation-mode, and droplet-like stiffness grids
  preserve issue #1272 validation signals on Warp CPU; optional CUDA skips or
  passes cleanly.
- [ ] No container-schema, fp64, direct-import, or hidden-transfer regression.
- [ ] Documentation states limits: fixed count, no adaptive integration, no gas
  coupling, and `5e-2` as recorded evidence rather than universal tolerance.

## Metrics

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Explicit substeps per call | 1 | Exactly 4 | Launch-count tests |
| Required stable-shape allocations with complete scratch | Environment buffers allocated | 0 | P1 allocation instrumentation |
| Recorded-grid max relative error | Candidate <= `5e-2` | Production <= `5e-2` | Stiffness suite |
| Negative or non-finite particle outputs | Candidate 0 | 0 | Production regressions |
| Mutation on invalid input | Existing mass-transfer contract: 0 | 0 for all scratch | Rejection tests |
| Hidden host transfers | 0 required | 0 | API/instrumentation tests |
