# Success Criteria

- [x] Warp conductivity, resistance, and corrected helper rates match CPU
  equations, including exact zero-latent isothermal helper identity.
- [x] Correction refreshes from E4-F1/F2 state in all four E4-F3 substeps.
- [x] Omitted/zero latent preserves isothermal results and existing API behavior.
- [x] Whole-call `(n_boxes, n_species)` energy remains on device and satisfies
  signed `Q = Σ(Δm_applied L)` for condensation, evaporation, zero, and clamps.
- [x] Shape, dtype, device, finiteness, and sign validation of supplied P1
  sidecars precedes environment normalization, allocation, refresh, launch,
  and caller-owned mutation.
- [x] Supplied P1 sidecars retain identity and contents; no hidden host
  transfer/schema change was introduced.
- [x] Warp-CPU four-substep oracle regressions cover composed scalar and
  explicit-environment routes, including signed energy, applied transfer,
  zero-latent energy, output identity, and unchanged gas concentration; optional
  CUDA coverage is additive and skips cleanly when unavailable.
- [x] Three GPU feature documents state the issue #1272 energy identity,
  sidecar ownership/units, focused commands, and supported limits.
- [ ] Gas conservation remains E4-F5 scope and complete evidence E4-F6 scope.

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Corrected-rate parity | No Warp correction | CPU four-substep oracle at documented tolerance | GPU tests |
| Energy identity residual | No Warp diagnostic | rtol <= 1e-12, atol <= 1e-18 where representable | Tests |
| Corrected fixed substeps | 0 | 4 of 4 | Launch/oracle tests |
| Hidden host transfers | N/A | 0 | Review/tests |
| Required allocations with all scratch | N/A | 0 | Instrumentation |
| Changed-code coverage | N/A | >= 80% | pytest-cov |
