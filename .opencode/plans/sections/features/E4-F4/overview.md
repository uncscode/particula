# E4-F4: Warp Latent-Heat Correction and Energy Diagnostics

**Problem Statement:** The Warp condensation path applies an isothermal rate
and lacks signed latent-energy bookkeeping, preventing CPU physics parity and
verification of issue #1272's energy identity.

**Value Proposition:** Porting the CPU thermal-resistance equation to fp64
Warp code makes each E4-F3 substep physically consistent. A whole-call
diagnostic based on bounded applied mass gives a deterministic `Q = Δm L`
audit trail without host transfers or container schema changes.

**P1 Delivered (issue #1297):** Private fp64 Warp conductivity,
thermal-resistance, and latent-corrected-rate helpers now reproduce the CPU
formulas, including exact zero-latent isothermal helper behavior.
`condensation_step_gpu()` now atomically validates optional caller-owned
`latent_heat` and `thermal_work` fp64 `(n_species,)` sidecars before any work.
They are validation-only and remain unused and unmodified; production rate
correction and energy diagnostics remain future phases.

**User Stories:**
- As a simulation user, I want latent heat to reduce GPU condensation rates so
  Warp results agree with CPU reference physics.
- As a model validator, I want signed per-box/per-species energy totals so I can
  audit condensation, evaporation, and zero transfer.
- As a GPU integrator, I want fixed-shape caller-owned buffers so repeated and
  graph-oriented execution remains allocation-stable.

Parent: E4. This feature converges E4-F1, E4-F2, and E4-F3 and preserves issue
#1272 energy-bookkeeping criteria.
