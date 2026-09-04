# Success Criteria

- [x] A qualified CUDA device captures exactly the E8-F2 complete prepared
  twelve-node enqueue sequence using the exact E8-F3 resource set.
- [x] Capture and replay perform no allocation, resource acquisition, validation
  scan, host readback, synchronization, transfer, process scheduling, RNG reset,
  fallback, retry, or automatic recapture inside the native sequence.
- [x] Every accepted replay opens one resident token, launches the exact graph
  once, completes the token, and retains all primary and sidecar identities.
- [x] Structural drift is rejected before launch with an E8-F1 canonical reason;
  mutable fixed-shape payload and persistent RNG-word changes remain replayable.
- [x] Capture/enqueue/end failures publish no graph and clean up exactly once;
  post-launch replay failures fault graph and session without rollback.
- [x] Finalize, close, fault, teardown, restart, device change, or stale handle
  cannot replay; records unregister before one release and recapture is explicit
  with a fresh record/handle.
- [x] CPU, uncaptured Warp, and captured CUDA fixtures use the same supported
  process sequence and agree within documented per-field criteria over multiple
  timesteps, with conservation asserted separately.
- [x] Warp CPU capture rejects or skips explicitly; CUDA unavailability cleanly
  skips native rows and never substitutes CPU execution.
- [x] Concrete graph names remain absent from package/top-level exports and
  graph handles remain absent from checkpoints.
- [ ] Focused tests, untargeted repository coverage validation, linting, and
  strict documentation build pass without lowering thresholds.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Prepared enqueue sequences per capture | No resident capture | Exactly 1 | Fake/native capture trace |
| Native graph launches per accepted replay | 0 | Exactly 1 | Runtime spy / CUDA test |
| Native graph launches after rejected preflight | N/A | 0 | Lifecycle unit tests |
| Allocations/readbacks/synchronizations during capture or replay | Public wrappers perform setup/readback | 0 | Forbidden-operation spies |
| Structural drift categories rejected before launch | No graph owner | 100% of signature categories | Parametrized tests |
| Duplicate cleanup attempts per failed capture | Test helper only | 0 | Failure-injection tests |
| Deterministic captured-vs-uncaptured field checks | Condensation-only harness | All selected full-loop fields within explicit tolerances | Full-loop tests |
| Particle-plus-gas conservation | Uncaptured resident baseline | Existing tight process-specific bounds | Independent inventory assertions |
| Aggregate full-package coverage | Existing repository threshold | Threshold unchanged and passing | `.opencode/tools/run_pytest.py` |

P5 now supplies bounded genuine native-CUDA three-way numerical, conservation,
RNG-continuation, and stochastic evidence. Full repository validation remains
tracked by the final unchecked workflow-level criterion above.
