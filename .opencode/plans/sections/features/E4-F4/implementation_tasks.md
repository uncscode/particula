# Implementation Tasks

### GPU Physics
- [ ] In `particula/gpu/dynamics/condensation_funcs.py`, add private fp64 Warp
  helpers equivalent to CPU `get_thermal_resistance_factor()` and
  `get_mass_transfer_rate_latent_heat()` (roughly 60--100 production LOC), with
  the same zero-latent isothermal limit and no reduced-order approximation.
- [ ] In `particula/gpu/kernels/condensation.py`, add keyword-only latent-heat,
  thermal-work, and optional energy-sidecar preflight (roughly 20--40 LOC);
  reject nonfinite/negative values and wrong `(n_species,)` or
  `(n_boxes, n_species)` fp64/device contracts before allocation or mutation.
- [ ] Feed the E4-F2 activity-adjusted Kelvin surface pressure into the new
  latent-rate helper from the existing `condensation_step_gpu()` calculate
  launch, so the isothermal and latent paths use one surface-pressure value.
- [ ] Apply the latent correction during each of E4-F3's four substeps (roughly
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
- [ ] In `particula/gpu/dynamics/tests/condensation_funcs_test.py`, add
  NumPy-reference fixtures for conductivity, thermal factor, corrected rate,
  and zero-latent identity.
- [ ] In `particula/gpu/kernels/tests/condensation_test.py`, snapshot all mutable
  buffers before nonfinite/negative latent and sidecar shape/dtype/device
  rejections, then assert pre-mutation failure.
- [ ] Cover positive, negative, zero, clamped, and multi-box/species transfers
  on Warp CPU; assert energy sign, per-slot aggregation, and unchanged
  cross-box/species slots.
- [ ] Parameterize optional CUDA parity with clean skips and separate explicit
  physics and conservation tolerances.
- [ ] Verify repeated caller-owned thermal/energy buffer identity and that a
  fully supplied sidecar path requires no allocation or host synchronization.
- [ ] Run the two focused test modules with `-Werror`, then Ruff and mypy on
  `particula/gpu/dynamics/condensation_funcs.py` and
  `particula/gpu/kernels/condensation.py`.
