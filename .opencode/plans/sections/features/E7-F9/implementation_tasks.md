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

- [ ] Add `particula/execution/tests/full_loop_test.py` with independent CPU
  references, transfer/sync spies, canonical order, and derived-state checks.
- [ ] Add a 4-box, 16-particle-slot, 2-species particle-resolved
  `multi_box_loop_test.py` fixture and one-box decomposition/isolation
  metamorphic assertions.
- [ ] Add prescribed advection, dilution, mixing, and expansion cases in
  `transport_loop_test.py` with extensive-amount conservation accounting.
- [ ] Add checkpoint/restart and logical-box stream equivalence cases in
  `restart_loop_test.py`.
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
