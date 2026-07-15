# Success Criteria

- [x] P1 provides independent fp64 NumPy-oracle parity for the shared one-box
  and multi-box/multi-species cases, with particle mass and gas concentration
  compared separately.
- [x] P1 executes the required parity matrix on Warp CPU and runs the identical
  matrix on CUDA when available, with a clean CUDA skip otherwise.
- [x] P2 per-box/per-species particle gain, gas loss, returned transfer, and
  latent energy agree with the same P2-finalized bounded transfer; energy is
  intentionally unweighted by particle concentration.
- [x] P2 particle-plus-gas inventory passes the separate strict tolerance;
  inactive, disabled, and zero-concentration entries remain unchanged and final
  inventories stay finite/nonnegative.
- [x] P2 deterministic fresh calls preserve immutable inputs and supplied
  transfer/energy-output identities with complete caller-owned scratch.
- [x] P2 representative invalid shape/dtype/device/configuration paths fail
  before state or caller-owned-buffer mutation.
- [ ] Supported graph capture/replay matches normal launch parity and conservation.
- [ ] Bounded autodiff experiments report supported smooth-interior behavior and
  explicit clamp/in-place limitations without claiming full differentiability.
- [ ] Focused commands and evidence/non-claims are documented.

## Metrics

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Required backend coverage | Partial | Warp CPU 100% of required matrix | Pytest device cases |
| Optional CUDA coverage | Ad hoc | Same eligible matrix when available | Pytest device cases |
| Multi-box reference independence | Vectorized reference exists | One independent CPU run per box | Parity tests |
| Conservation granularity | Global CPU precedents | Per box and species | Conservation tests |
| Conservation tolerance | Mixed | Explicit strict target, nominally `rtol=1e-12` | Invariant assertions |
| Integrator loop count | Candidate evidence | Exactly 4 | Fixed-loop tests |
| Capture readiness | No condensation regression | Deterministic capture/replay where supported | Graph tests |
| Autodiff claim | Documentation only | Bounded evidence plus explicit limitations | Autodiff tests/docs |
