# Success Metrics

- [ ] A qualified CUDA device captures and replays at least one complete,
  fixed-order GPU-resident multi-box timestep without replay-time allocation,
  hidden host transfer, or bulk synchronization.
- [ ] Shape, device, process-order, communication-map, resource-identity, and
  unsupported-state changes reject deterministically or require documented
  explicit recapture.
- [ ] CPU, uncaptured Warp GPU, and captured Warp GPU full-loop fixtures match
  over multiple timesteps using documented per-process tolerances and tight
  concentration-weighted inventory checks.
- [ ] Persistent coagulation and wall-loss RNG streams advance across replay,
  reset only through explicit operations, remain nonaliasing, and preserve the
  supported checkpoint/restart continuation contract.
- [ ] Every E8-F1--E8-F8 implementation ships co-located tests; configured
  coverage remains at least 80% and no threshold is lowered.
- [ ] Opt-in CUDA scaling evidence covers box counts of at least 1, 10, 100,
  and 1000 where hardware memory permits, with particles per box as a secondary
  axis and explicit unavailable rows otherwise.
- [ ] Captured-versus-uncaptured launch-overhead results include small and
  medium repeated-step workloads, raw samples, summary statistics, warmup,
  synchronization method, software versions, device identity, and command.
- [ ] The memory model accounts for primary state, inactive slots, reusable
  temporary buffers, diagnostics, communication maps, checkpoint overhead, and
  projected autodiff tape storage; analytical estimates are compared with
  observed peak memory for representative fixtures.
- [ ] CUDA profiling records occupancy and memory-access evidence for the
  dominant kernels and identifies bounded follow-up work without changing
  scientific contracts merely to improve a metric.
- [ ] A runnable graph-capture example and feature documentation state setup,
  ownership, replay, teardown, limitations, recapture triggers, and clean-skip
  behavior; `mkdocs build --strict` passes.
- [ ] E8-F8 records the exact closeout command matrix and literal results, and
  all required rows pass before Epic H is promoted to Shipped.
