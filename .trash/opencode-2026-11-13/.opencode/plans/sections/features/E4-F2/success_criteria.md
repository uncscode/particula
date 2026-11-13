# Success Criteria

- [x] **P3 / issue #1289:** Supported ideal/kappa activity and static/weighted
  surface modes are explicitly selected through a frozen numeric sidecar and
  match independent coupled references.
- [x] **P1 / issue #1287:** Ideal molar activity and kappa water activity
  include guarded zero branches, dry/zero-solute and multi-solute behavior,
  and nonzero water-index cases matching independent NumPy references.
- [x] **P2 / issue #1288:** Composition-dependent effective surface tension is
  volume-weighted from `mass / density`, returns a finite arithmetic mean for
  zero total volume, and supplies Kelvin radius/term parity tests.
- [x] **P2 / issue #1288:** Existing `(n_species,)` surface input remains
  compatible through exact requested-species static selection.
- [x] Particle pressure is `activity * refreshed_pure_pressure * kelvin_term`.
- [x] Unsupported selectors and malformed supplied inputs fail before launch;
  BAT remains out of scope and CPU-only.
- [x] Fixed-shape fp64 arrays, species order, explicit transfer boundaries, and
  concrete `particula.gpu.kernels.condensation` import ownership are preserved.
- [x] Invalid configuration fails before particle, gas, environment, or supplied
  output-buffer mutation.
- [x] **P4 / issue #1290:** Independent deterministic fp64 NumPy parity covers
  all four supported mode pairs on one-box and multi-box fixtures, including
  constant/Buck vapor-pressure refresh and both Buck temperature branches.
- [x] **P4 / issue #1290:** Raw transfer, final clamp-to-zero mass, refreshed
  vapor pressure, finite/nonnegative output, unchanged gas concentration, and
  metadata ownership are asserted independently.
- [x] Warp CPU parity runs when Warp is installed; separately marked CUDA
  coverage passes when available and otherwise skips cleanly.
- [x] No high-level runnable support, schema change, or hidden host computation
  was introduced; deferred activity/surface strategies remain explicit
  CPU-only `ValueError` cases.

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Supported activity modes | Unit activity only | Ideal molar + kappa water | Property parity tests |
| Effective surface modes | Static species input | Static + selected composition mode | Property/kernel tests |
| Independent parity | Not available | Pass documented `rtol`/`atol` | Warp CPU/CUDA fixtures |
| Validation mutation failures | Not comprehensive | 0 state mutations | Snapshot tests |
| Changed-code coverage | N/A | >=80%, threshold not lowered | pytest-cov |
