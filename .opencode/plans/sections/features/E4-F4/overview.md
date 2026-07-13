# E4-F4: Warp Latent-Heat Correction and Energy Diagnostics

**Problem Statement:** The Warp condensation path applies an isothermal rate
and lacks signed latent-energy bookkeeping, preventing CPU physics parity and
verification of issue #1272's energy identity.

**Value Proposition:** Porting the CPU thermal-resistance equation to fp64
Warp code makes each E4-F3 substep physically consistent. A whole-call
diagnostic based on bounded applied mass gives a deterministic `Q = Δm L`
audit trail without host transfers or container schema changes.

**User Stories:**
- As a simulation user, I want latent heat to reduce GPU condensation rates so
  Warp results agree with CPU reference physics.
- As a model validator, I want signed per-box/per-species energy totals so I can
  audit condensation, evaporation, and zero transfer.
- As a GPU integrator, I want fixed-shape caller-owned buffers so repeated and
  graph-oriented execution remains allocation-stable.

Parent: E4. This feature converges E4-F1, E4-F2, and E4-F3 and preserves issue
#1272 energy-bookkeeping criteria.
