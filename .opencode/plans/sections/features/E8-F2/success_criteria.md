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

## P1 Completion Evidence (#1552)

- [x] `prepare_resident_timestep()` constructs a frozen, exact-identity READY
  metadata carrier for the canonical twelve-node request only after shared
  complete-loop validation and a second signature-drift check.
- [x] Preparation is proven setup-only by a focused guard-token-entry trap; no
  enqueue or dispatch path was introduced. Broader forbidden-operation trap
  coverage remains owned by later enqueue phases.
- [x] Extracted diagnostics and communication functional validators retain
  executor-wrapper regression coverage, while export tests keep
  `resident_enqueue` and its names absent from package and top-level APIs.

## P2 Completion Evidence (#1553)

- [x] P1-bound state, thermodynamic, and diagnostics setup validates and freezes
  exact prepared-timestep ownership, identity, and pinning chains before a
  writer is enqueued.
- [x] State and diagnostics prepared dispatch retain fixed writer ordering and
  write-free valid empty schemas; prepared thermodynamics retains the
  coordinator's established vapor/saturation cursor and freshness semantics.
- [x] Validate-once/enqueue-only tests prove prepared dispatch does not repeat
  setup work while legacy standalone executor/coordinator paths remain valid.

## P3 Completion Evidence (#1554)

- [x] P1-bound setup freezes one exact closed-map GAS or PARTICLES communication
  family, associated primary/work/status identities, duration, and optional
  final-volume sidecar; invalid mode, endpoint, schema, or identity drift rejects
  before prepared launch.
- [x] Prepared dispatch uses bound native communication helpers followed only by
  a present volume helper, with no enqueue-time validation, lookup, acquisition,
  allocation, transfer/readback, or synchronization.
- [x] Regression coverage proves absent/equal/changed volume behavior,
  communication-before-volume ordering, no-op barriers, GAS overdraw gated
  commit, and legacy/direct-path compatibility. Equal final volumes leave
  primaries, work ledgers, and resident volume status lanes unchanged.

## P4 Completion Evidence (#1555)

- [x] Private `_PreparedCondensationCall` setup retains validated direct-kernel
  inputs and sidecars while preserving the public wrapper's validation,
  fallback-allocation, identity, and return contracts.
- [x] Prepared enqueue retains the four equal gas-coupled, inventory-limited P2
  substeps and performs no validation, allocation, host refresh/readback,
  synchronization, or resource lookup.
- [x] Private `_PreparedWarpCondensationBinding` retains the concrete resident
  adapter's prepared call without changing public APIs, scheduler dispatch,
  checkpoint/resource schemas, or condensation physics. Changed local
  Google-style docstrings are covered by focused docstring validation.
