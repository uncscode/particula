# Success Criteria

- [ ] Warp conductivity, resistance, and corrected rates match CPU equations.
- [ ] Correction refreshes from E4-F1/F2 state in all four E4-F3 substeps.
- [ ] Omitted/zero latent preserves isothermal results and existing API behavior.
- [ ] Whole-call `(n_boxes, n_species)` energy remains on device and satisfies
  signed `Q = Σ(Δm_applied L)` for condensation, evaporation, zero, and clamps.
- [ ] Shape, dtype, device, finiteness, and sign validation precedes all work.
- [ ] Supplied buffers retain identity; no hidden host transfer/schema change.
- [ ] Warp CPU passes and optional CUDA passes or skips cleanly.
- [ ] Gas conservation remains E4-F5 scope and complete evidence E4-F6 scope.

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Corrected-rate parity | No Warp correction | Recorded CPU tolerance | GPU tests |
| Energy identity residual | No Warp diagnostic | rtol <= 1e-12, atol <= 1e-18 where representable | Tests |
| Corrected fixed substeps | 0 | 4 of 4 | Integration tests |
| Hidden host transfers | N/A | 0 | Review/tests |
| Required allocations with all scratch | N/A | 0 | Instrumentation |
| Changed-code coverage | N/A | >= 80% | pytest-cov |
