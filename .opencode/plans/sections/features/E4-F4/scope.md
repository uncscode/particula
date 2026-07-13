# Scope

Deliver an fp64 Warp port of the CPU latent-heat correction, apply it during
every fixed condensation substep, and expose deterministic signed whole-call
energy bookkeeping from applied mass transfer.

**In scope:**
- Warp thermal-conductivity and thermal-resistance calculations matching CPU.
- Fixed-shape per-species latent configuration and pre-mutation validation.
- Reuse of E4-F1 refreshed saturation pressure, E4-F2 activity/Kelvin surface
  pressure, and E4-F3 four-substep scratch and accumulation contracts.
- Correction per substep and exact zero-latent isothermal fallback.
- Caller-owned on-device energy output accumulated as signed
  `bounded_mass_transfer * latent_heat` over the full call.
- Warp CPU parity, optional CUDA checks, regressions, and documentation.

**Out of scope:**
- Gas mutation, inventory limits, and full system conservation (E4-F5).
- High-level `Aerosol`/`Runnable` integration or GPU container schema changes.
- Temperature evolution, adaptive substeps, BAT support, hidden host reduction,
  telemetry, logging, or additional diagnostics.
- Complete cross-device epic evidence (E4-F6).
