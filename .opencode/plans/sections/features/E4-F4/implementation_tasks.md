# Implementation Tasks

### GPU Physics
- [x] In `particula/gpu/dynamics/condensation_funcs.py`, add private fp64 Warp
  helpers equivalent to CPU `get_thermal_resistance_factor()` and
  `get_mass_transfer_rate_latent_heat()` (roughly 60--100 production LOC), with
  the same zero-latent isothermal limit and no reduced-order approximation.
- [x] In `particula/gpu/kernels/condensation.py`, add keyword-only
  `(n_species,)` latent-heat and thermal-work preflight; reject nonfinite or
  negative values and wrong shape, fp64 dtype, device, or non-Warp inputs
  before allocation or mutation. P1 does not add or validate an energy sidecar.
- [x] Feed the E4-F2 activity-adjusted Kelvin surface pressure into the new
  latent-rate helper from the existing `condensation_step_gpu()` calculate
  launch, so the isothermal and latent paths use one surface-pressure value.
- [x] Apply the latent correction during each of E4-F3's four substeps (roughly
  30--50 orchestration LOC), retaining bitwise-compatible isothermal behavior
  when latent heat is omitted or zero.
- [ ] Route only transfer that has passed the existing particle-mass bound to
  particle mutation, whole-call transfer accumulation, and energy bookkeeping.
- [ ] Add a private on-device accumulation launch in `condensation.py` that
  writes signed `applied_transfer * latent_heat` into caller-owned fp64
  `(n_boxes, n_species)` energy storage without changing the default two-item
  `condensation_step_gpu()` return.
- [ ] Preserve positional callers, lazy kernel exports, and physical `WarpData`
  schemas; keep all new configuration and scratch as typed keyword-only
  operation sidecars.

### Tooling / Tests
- [x] In `particula/gpu/dynamics/tests/condensation_funcs_test.py`, add
  NumPy-reference fixtures for conductivity, thermal factor, corrected rate,
  and zero-latent identity.
- [x] In `particula/gpu/kernels/tests/condensation_test.py`, snapshot all mutable
  buffers before nonfinite/negative latent and sidecar shape/dtype/device
  rejections, then assert pre-mutation failure.
- [x] Add CPU four-substep oracle parity for multi-species latent heat, including
  shared activity/Kelvin pressure, exact isolated zero-latent fallback, and
  coupled mixed-latent coverage.
- [x] Verify validation atomicity, sidecar immutability, four latent-aware
  launches, deterministic fresh-state results, returned-total/scratch identity,
  scratch reuse, and no omitted-latent allocation.
- [ ] Add energy-sign, aggregation, and clamp-accounting tests with P3 energy
  bookkeeping.
