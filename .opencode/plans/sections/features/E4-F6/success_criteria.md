# Success Criteria

- [ ] Independent fp64 CPU references pass for every one-box case and each box
  in multi-box production cases.
- [ ] Warp CPU executes all required parity, conservation, buffer, and fixed-loop
  cases; CUDA runs the same eligible matrix when available and skips cleanly otherwise.
- [ ] Per-box/per-species particle gain, gas loss, returned transfer, and latent
  energy agree with the same bounded transfer.
- [ ] Particle-plus-gas inventory passes a separately stated strict tolerance;
  inactive/disabled entries remain unchanged and all inventories stay finite/nonnegative.
- [ ] Exactly four substeps execute with stable fixed-shape caller-owned scratch,
  deterministic output, and no hidden host transfer.
- [ ] Invalid shape/device/configuration fails before state mutation.
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
