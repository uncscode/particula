# Implementation Tasks

### GPU Physics
- [ ] Port CPU conductivity and thermal-resistance equations without approximation.
- [ ] Validate latent configuration and all sidecars before allocation/mutation.
- [ ] Share E4-F2 activity-adjusted Kelvin surface pressure across both paths.
- [ ] Apply correction in each E4-F3 substep; preserve zero-latent parity.
- [ ] Bound transfer before state, mass accumulation, and energy accumulation.
- [ ] Accumulate signed whole-call `(n_boxes, n_species)` energy on device.
- [ ] Preserve existing callers, returns, lazy exports, and container schemas.

### Tooling / Tests
- [ ] Add CPU-reference fixtures for conductivity, resistance, rates, and energy.
- [ ] Test nonfinite/negative latent and sidecar shape/dtype/device failures.
- [ ] Test positive, negative, zero, clamped, multi-box/species transfer on Warp CPU.
- [ ] Add optional CUDA parity with clean skips and separate tolerances.
- [ ] Verify repeated buffer identity and no required fully-supplied allocation.
- [ ] Run focused pytest, Ruff, and mypy checks on changed modules.
