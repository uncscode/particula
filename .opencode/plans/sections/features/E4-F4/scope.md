# Scope

Deliver an fp64 Warp port of the CPU latent-heat correction, apply it during
every fixed condensation substep, and expose deterministic signed whole-call
energy bookkeeping from applied mass transfer.

**Delivered P1 boundary:** The formula port and validation-only sidecar API are
complete. Applying the correction, consuming thermal work, and energy
bookkeeping remain explicitly deferred to P2/P3.

**In scope:**
- Private Warp thermal-conductivity, thermal-resistance, and latent-rate
  calculations matching CPU, including exact zero-latent helper identity.
- Fixed-shape per-species `latent_heat` and `thermal_work` pre-mutation
  validation; P1 does not consume either sidecar.
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
