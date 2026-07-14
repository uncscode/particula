# Scope

Deliver an fp64 Warp port of the CPU latent-heat correction, apply it during
every fixed condensation substep, and expose deterministic signed whole-call
energy bookkeeping from applied mass transfer.

**Delivered P1/P2/P3 boundary:** The formula port, sidecar validation,
four-substep latent correction, and opt-in energy bookkeeping are complete.
`thermal_work` consumption remains explicitly deferred.

**In scope:**
- Private Warp thermal-conductivity, thermal-resistance, and latent-rate
  calculations matching CPU, including exact zero-latent helper identity.
- Fixed-shape per-species `latent_heat` and `thermal_work` pre-mutation
  validation; P2 consumes nonzero latent entries only and leaves thermal work
  deferred.
- Reuse of E4-F1 refreshed saturation pressure, E4-F2 activity/Kelvin surface
  pressure, and E4-F3 four-substep scratch and accumulation contracts.
- Correction per substep and exact zero-latent isothermal fallback.
- Optional caller-owned, device-only fp64 `(n_boxes, n_species)` energy output
  for signed whole-call bounded applied transfer times latent heat, with atomic
  preflight, one clear, and one final single-writer reduction.
- CPU-oracle/Warp parity, validation atomicity, deterministic scratch-reuse
  regressions, and focused production-docstring updates.

**Out of scope:**
- Gas mutation, inventory limits, and full system conservation (E4-F5).
- High-level `Aerosol`/`Runnable` integration or GPU container schema changes.
- Temperature evolution, adaptive substeps, BAT support, hidden host reduction,
  telemetry, logging, or additional diagnostics.
- Complete cross-device epic evidence (E4-F6).
- `thermal_work` consumption, temperature evolution, or a persistent energy
  accumulator/result-object API.
