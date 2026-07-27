# Success Metrics

- [ ] All nine ordered feature tracks E7-F1 through E7-F9 ship with their
  declared dependency gates satisfied.
- [ ] At least one condensation and one Brownian coagulation workflow execute
  through backend selection and match independent CPU references within
  recorded tolerances.
- [ ] A multi-timestep GPU loop performs exactly one setup upload and zero bulk
  CPU/GPU transfers between explicit checkpoints.
- [ ] Checkpoint and finalization paths explicitly synchronize and restore all
  documented state and metadata needed for restart.
- [ ] Container identities, array shapes, sidecar identities, and fixed capacity
  remain stable across repeated timesteps.
- [ ] All supported processes run in a documented deterministic order with no
  stale environment, vapor-pressure, or saturation state.
- [ ] Independent multi-box results match equivalent one-box references.
- [ ] Prescribed transport, mixing, and volume evolution satisfy documented
  parity and particle-plus-gas conservation rules.
- [ ] Per-box RNG streams reproduce after restart and enabled-box results remain
  unchanged when unrelated boxes are added, disabled, or reordered.
- [ ] Unsupported physics and unavailable devices produce tested capability
  errors or an explicitly requested fallback transition; no silent movement.
- [ ] Every changed module maintains at least 80% coverage without lowering any
  repository threshold, and unit tests ship with implementation.
- [ ] Warp CPU full-loop regressions pass; optional CUDA rows skip cleanly when
  unavailable and pass when suitable hardware exists.
- [ ] The published multi-timestep example and support matrix satisfy the Epic G
  roadmap exit bar and are guarded by documentation regressions.
- [ ] Graph capture/performance work and autodiff/optimization remain deferred
  to Epics H and I.
