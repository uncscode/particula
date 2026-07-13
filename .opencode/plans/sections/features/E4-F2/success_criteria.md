# Success Criteria

- [ ] Supported modes are explicitly selected and match issue #1272 CPU fixtures.
- [x] **P1 / issue #1287:** Ideal molar activity and kappa water activity
  include guarded zero branches, dry/zero-solute and multi-solute behavior,
  and nonzero water-index cases matching independent NumPy references.
- [ ] Composition-dependent effective surface tension affects Kelvin pressure.
- [ ] Existing per-species surface input remains compatible as static mode.
- [ ] Particle pressure is `activity * refreshed_pure_pressure * kelvin_term`.
- [ ] Unsupported modes fail early or are explicitly documented CPU-only,
  including BAT.
- [ ] Fixed-shape fp64 arrays, species order, explicit transfer boundaries, and
  direct `particula.gpu.kernels` imports are preserved.
- [ ] Invalid configuration fails before particle, gas, or environment mutation.
- [ ] Warp CPU parity passes when Warp is installed; CUDA passes when available
  and otherwise skips cleanly.
- [ ] No high-level runnable support, schema change, or hidden host computation
  is introduced; project coverage remains at least 80%.

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Supported activity modes | Unit activity only | Ideal molar + kappa water | Property parity tests |
| Effective surface modes | Static species input | Static + selected composition mode | Property/kernel tests |
| Independent parity | Not available | Pass documented `rtol`/`atol` | Warp CPU/CUDA fixtures |
| Validation mutation failures | Not comprehensive | 0 state mutations | Snapshot tests |
| Changed-code coverage | N/A | >=80%, threshold not lowered | pytest-cov |
