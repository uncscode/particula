# Implementation Tasks

## Execution and Diagnostics

- [ ] Inventory E7-F1 through E7-F8 public contracts and resolve any handoff
  mismatch with the owning track before adding closeout fixtures.
- [ ] Implement public typed diagnostic descriptors and explicit-boundary
  observation results in `particula/execution/diagnostics.py` only where E7-F5
  hooks are insufficient; keep reducers and restart resources private.
- [ ] Implement device-side total species mass, number concentration,
  latent-energy, and ledger-aware conservation reductions.
- [ ] Register diagnostic buffers through E7-F4 resources and preserve their
  shape, dtype, device, and identity across repeated steps.
- [ ] Freeze checkpoint schema/version and audit physical arrays, gas names,
  logical box IDs, dimensions, counters, capability/config identity, ledgers,
  diagnostics, and RNG streams.

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
