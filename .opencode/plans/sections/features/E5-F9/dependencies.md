# Dependencies

## Upstream

- **E5-F1 through E5-F6:** final mechanism configuration, charged,
  sedimentation, turbulent-shear, and additive execution contracts.
- **E5-F7:** passing cross-mechanism deterministic, stochastic, conservation,
  multi-box, buffer, RNG, Warp CPU, and optional CUDA evidence.
- **E5-F8:** published condensation walkthrough and complete downstream-owner
  table for Epic D carry-forward capabilities.
- Shipped E2-E4 explicit container/transfer, RNG, direct-kernel, and
  device-validation baselines.

## Downstream

- E5 roadmap completion and activation of Epic F depend on E5-F9's publication
  and closeout gate; that gate has shipped and now records E5 as shipped and
  Epic F as active.
- Epic F consumes the documented process/support boundary; Epic G owns
  high-level backend/`Runnable` integration, and later epics retain graph,
  performance, and autodiff ownership.

## Phase Ordering

P1 and P2 can run in parallel after API/evidence freeze. P3 required stable
artifact paths from P1/P2 and E5-F7/F8. P4 required P1-P3 plus shipped E5-F1
through E5-F8 and all release checks; those checks passed, so E5 is shipped and
Epic F is active. Optional CUDA unavailability is not a failure; required Warp
CPU checks were failures when Warp was installed.
