# E4-F4: Warp Latent-Heat Correction and Energy Diagnostics

**Problem Statement (pre-P2):** The Warp condensation path applied only an
isothermal rate and lacked signed latent-energy bookkeeping. This blocked
latent-rate CPU-oracle parity and verification of issue #1272's energy
identity.

**Value Proposition:** The shipped correction and opt-in P3 diagnostic make
each of the four fixed GPU condensation substeps agree with the CPU reference
while preserving the low-level, allocation-stable contract. Callers can record
signed whole-call `Q = Δm L` diagnostics without a return or schema change.

**P1 Delivered (issue #1297):** Private fp64 Warp conductivity,
thermal-resistance, and latent-corrected-rate helpers now reproduce the CPU
formulas, including exact zero-latent isothermal helper behavior.
`condensation_step_gpu()` now atomically validates optional caller-owned
`latent_heat` and `thermal_work` fp64 `(n_species,)` sidecars before any work.
`thermal_work` remains validation-only and unmodified.

**P2 Delivered (issue #1298):** `condensation_step_gpu()` applies the
per-species latent-heat correction in each of its four fixed substeps. The
activity- and Kelvin-adjusted surface vapor pressure is computed once and is
shared by the pressure-delta and thermal-rate paths. Omitted latent heat and
exactly zero per-species entries retain the preexisting isothermal arithmetic.
No return shape, gas-state, or `thermal_work` behavior changed.

**P3 Delivered (issue #1299):** `condensation_step_gpu()` accepts optional,
caller-owned, active-device `wp.float64` `energy_transfer` storage shaped
`(n_boxes, n_species)`. After atomic metadata/dependency preflight it clears
the output once, then after all four substeps each box/species has one writer
that reduces bounded accumulated particle transfer times latent heat. The
output is overwritten by identity, remains device-only, and is not a third
 return item. Omitted output adds no energy allocation or kernel work.

**P4 Delivered (issue #1300):** Fresh-state composed regressions now validate
both scalar temperature/pressure and explicit `WarpEnvironmentData` routes on
Warp CPU against the four-substep CPU oracle. They cover applied transfer,
signed caller-owned energy output, zero-latent energy, sidecar identity, and
unchanged gas concentration; equivalent CUDA coverage is additive through the
existing availability-skipped fixture. The three GPU feature documents now
state the shipped equation, ownership, commands, and bounded non-goals.

**User Stories:**
- As a simulation user, I want latent heat to reduce GPU condensation rates so
  Warp results agree with CPU reference physics.
- As a model validator, I want deterministic CPU-oracle/Warp parity for the
  corrected four-substep rate and signed bounded energy totals.
- As a GPU integrator, I want fixed-shape caller-owned buffers so repeated and
  graph-oriented execution remains allocation-stable.

Parent: E4. This feature converges E4-F1, E4-F2, and E4-F3 and preserves issue
#1272 energy-bookkeeping criteria.
