# E4-F4: Warp Latent-Heat Correction and Energy Diagnostics

**Problem Statement (pre-P2):** The Warp condensation path applied only an
isothermal rate and lacked signed latent-energy bookkeeping. This blocked
latent-rate CPU-oracle parity and verification of issue #1272's energy
identity.

**Value Proposition:** The shipped P2 fp64 latent-rate correction makes each
of the four fixed GPU condensation substeps agree with the CPU reference while
preserving the existing low-level, allocation-stable contract. Signed
whole-call `Q = Δm L` diagnostics and totals are deferred P3 scope.

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
No public API, return shape, gas-state, or `thermal_work` behavior changed.
P3 signed energy diagnostics and totals remain future scope.

**User Stories:**
- As a simulation user, I want latent heat to reduce GPU condensation rates so
  Warp results agree with CPU reference physics.
- As a model validator, I want deterministic CPU-oracle/Warp parity for the
  corrected four-substep rate without claiming P3 energy totals.
- As a GPU integrator, I want fixed-shape caller-owned buffers so repeated and
  graph-oriented execution remains allocation-stable.

Parent: E4. This feature converges E4-F1, E4-F2, and E4-F3 and preserves issue
#1272 energy-bookkeeping criteria.
