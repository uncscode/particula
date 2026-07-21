# Success Metrics

- [ ] All nine feature plans E6-F1 through E6-F9 ship in dependency order.
- [ ] CPU dilution and nucleation references precede and support GPU parity.
- [ ] GPU dilution and wall-loss coefficients/results meet recorded
  deterministic or statistical tolerances against CPU references.
- [ ] Zero charge and zero field reproduce neutral wall-loss behavior.
- [ ] CPU/GPU nucleation conserves per-box/species inventory and never creates
  negative gas concentration.
- [ ] Slot activation preserves every array shape and reports exact per-box
  activated, resampled, scaled, and rejected demand diagnostics.
- [ ] Resampling defaults on; representative-volume scaling defaults off; both
  are independently selectable; disabling both fails before mutation.
- [ ] Invalid calls leave particles, gas, RNG, scratch, and diagnostics
  unmodified.
- [ ] Required Warp CPU tests pass; CUDA evidence skips cleanly when absent.
- [ ] A direct fixed-shape sequence runs condensation, coagulation, dilution,
  wall loss, and nucleation without intermediate host transfers.
- [ ] Tests remain at or above the configured 80% coverage threshold and are
  co-located with each feature implementation.
- [ ] Generated epic/child plan IDs are cross-linked in both roadmap documents,
  and Epic G boundaries remain explicit.
