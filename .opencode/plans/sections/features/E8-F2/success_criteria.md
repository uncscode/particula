# Success Criteria

- [ ] One explicit setup call validates and prepares the complete resident
  twelve-node sequence before capture.
- [ ] Every prepared node has a private/concrete-only device enqueue path that
  performs no allocation, host-to-device or device-to-host transfer, payload
  readback, synchronization, resource acquisition, device selection, fallback,
  retry, RNG initialization, or dynamic schedule resolution.
- [ ] Existing public direct-kernel APIs retain signatures, validation ordering,
  atomic prelaunch rejection, numerical behavior, and export boundaries.
- [ ] Prepared enqueue preserves canonical process/refresh order, exact primary
  and sidecar identities, fixed shapes, selected-box behavior, and persistent
  RNG advancement without reseeding.
- [ ] Structural or identity drift rejects before enqueue and follows E8-F1's
  deterministic invalidation/recapture contract.
- [ ] Setup rejection leaves resident primary state and lifecycle unchanged;
  enqueue failures retain shipped writer-may-have-launched fault semantics.
- [ ] Warp CPU focused and full resident tests pass for uncaptured setup/enqueue;
  CUDA capture smoke evidence passes or cleanly skips without CPU fallback.
- [ ] Untargeted repository coverage and strict documentation validation pass
  without lowering thresholds or changing default test collection.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Host payload readbacks in prepared enqueue | Direct paths may validate per call | 0 | Forbidden-call spies / launch trace tests |
| Device allocations in prepared enqueue | Some direct paths allocate validation or normalized state | 0 | `wp.zeros`/`wp.array` spies |
| Resource acquisitions in prepared enqueue | Scheduler promises none, kernels may allocate privately | 0 | Registry and allocator spies |
| Dynamic schedule decisions in enqueue | Python loops and per-node branching each timestep | 0 | Prepared operation trace |
| Canonical resident nodes represented | 12 | 12 | Prepared-plan unit test |
| Public direct-kernel regression failures | 0 | 0 | Existing kernel suites |
| Full-package coverage threshold | Repository configured | Met or exceeded | `.opencode/tools/run_pytest.py` |

Performance speedup is deliberately not a success metric for E8-F2; E8-F5 and
E8-F8 own benchmark and profiling evidence.
