# Implementation Tasks

## Execution and Diagnostics

- [ ] Inventory E7-F1 through E7-F8 public contracts and resolve any handoff
  mismatch with the owning track before adding closeout fixtures.
- [x] **P1 / issue #1528 (2026-08-10):** Implemented the concrete-only closed
  six-operation protocol in `particula/execution/diagnostics.py`: two preserved
  snapshots plus device-side total species mass, particle-number concentration,
  latent-energy copy, and ledger-aware conservation residual reductions.
- [x] **P1 / issue #1528 (2026-08-10):** Generalized diagnostic-registration
  preflight in `particula/execution/gpu_resources.py` for operation-specific
  schemas, capacity, device, and non-aliasing validation while retaining
  caller-owned binding identity.
- [x] **P1 / issue #1528 (2026-08-10):** Added co-located contract coverage in
  `particula/execution/tests/diagnostics_test.py` and
  `particula/execution/tests/gpu_resources_test.py`; no public descriptors,
  exports, checkpoint changes, scheduler behavior, or public documentation were
  added.
- [x] **P2 / issue #1529 (2026-08-10):** Froze checkpoint schema-v3
  continuation metadata, including the valid zero-current-word form, in
  `particula/execution/checkpoint.py`. Checkpoint docstrings now identify
  canonical primary bytes and registry-owned sidecars, ledgers, diagnostics,
  and optional closed-map state as recovery authority; arbitrary caller outputs
  remain excluded.
- [x] **P2 / issue #1529 (2026-08-10):** Added fail-closed bidirectional
  coagulation and wall-loss resource/continuation pairing validation and focused
  checkpoint coverage in `particula/execution/tests/checkpoint_test.py` for
  canonical-primary immutability and schema-v2 noncommunication restart.

## Integration Regressions

- [x] **P3 / issue #1530 (2026-08-10):** Added
  `particula/execution/tests/full_loop_test.py` with two-dispatch closed GAS and
  PARTICLES resident-loop regressions. The rows assert canonical ordinary-node
  traces, virtual refresh placement, NumPy float64 derived-state observations,
  one setup upload per CPU container, stable resident identity/schema, and
  closed GAS inventory conservation at `rtol=1e-12`, `atol=1e-30`.
- [x] **P3 / issue #1530 (2026-08-10):** Corrected the private `nucleation`
  scheduler branch in `particula/execution/resident_scheduler.py` to execute the
  ordinary nucleation adapter followed by `thermal.record_completed(node)`.
  Added a late wall-loss writer-failure regression that confirms guard closure,
  `FAULTED` state, and lifecycle-preflight rejection on later dispatch; it makes
  no rollback claim. No public API or canonical ordering changed.
- [x] **P4 / issue #1531 (2026-08-10):** Added only
  `particula/execution/tests/multi_box_loop_test.py`. Its internal resident-loop
  regressions cover 4-box, 16-slot, 2-species fixed-capacity decomposition;
  logical-ID permutation, unrelated-box addition, and selected/no-work wall-loss
  isolation; per-logical-box resident RNG continuity; and bounded neutral
  wall-loss aggregate evidence. Production code, exports, checkpoints, and
  public documentation remain unchanged.
- [x] **P5 / issue #1532 (2026-08-11):** Added
  `particula/execution/tests/transport_loop_test.py` with test-only closed-map
  transport regressions. Independent NumPy float64 extensive-amount oracles
  cover two-step directed expansion and reciprocal mixing with dilution, sparse
  arbitrary-pair conservation, and write-free empty/disabled map barriers.
- [x] **P5 / issue #1532 (2026-08-11):** Added
  `particula/execution/tests/restart_loop_test.py` with exact-device closed
  transport restart coverage, fresh restored resource identities, preserved
  published coagulation/wall-loss stream words without reseeding, continued
  transport equivalence, and nonexact-device rejection without source mutation.
- [x] **P5 / issue #1532 (2026-08-11):** Extended
  `particula/gpu/kernels/tests/communication_test.py` to reconcile direct
  open-boundary gas totals against source-minus-sink ledgers at `rtol=1e-12`,
  `atol=1e-30`.
- [ ] Parametrize Warp CPU as baseline and CUDA as optional cleanly skipped rows;
  record deterministic tolerances and stochastic acceptance rules explicitly.
- [ ] Retain export, unavailable-device, unsupported-physics, no-fallback, and
  post-launch fault regressions from E7-F6.

## Documentation and Closeout

- [ ] Publish `docs/Examples/gpu_resident_multi_timestep.py` through the
  user-facing execution API; avoid direct kernel orchestration.
- [ ] Add an executable docs regression proving setup/checkpoint transfer counts.
- [ ] Update the feature guide with support matrix, ownership, diagnostics,
  checkpoint/restart, errors, reproducibility, and limitations.
- [ ] Update the Epic G roadmap with dated phase/exit-bar evidence while leaving
  Epic H graph-capture/performance and Epic I autodiff scope unclaimed.
- [ ] Publish the exact validation matrix and commands, run focused/full fast
  suites and strict docs, and link results from issue #1451 closeout evidence.
