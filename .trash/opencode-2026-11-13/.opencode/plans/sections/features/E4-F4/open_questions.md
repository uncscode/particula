# Open Questions

- [x] Final-substep or full-call energy?
  - Resolved 2026-07-12: all four substeps, matching E4-F3 and CPU call semantics.
- [x] Which transfer drives energy?
  - Resolved 2026-07-12: bounded mass actually applied after clamping.
- [x] Add diagnostics to GPU containers?
  - Resolved 2026-07-12: no; use fixed-shape caller-owned sidecars.
- [x] Should opt-in energy be only an output keyword or join a future result
  object while preserving the default two-item return?
  - Resolved 2026-07-13: add optional caller-owned `energy_transfer` output and
    preserve the default two-item return. Defer a result object until multiple
    mandatory diagnostics justify an API migration.
- [x] What production energy granularity is supported?
  - Resolved 2026-07-13: signed whole-call `(n_boxes, n_species)` energy only.
    Particle-resolved energy may exist as private/test scratch but is not a
    public diagnostic.
- [x] What latent-heat model does E4-F4 evaluate on device?
  - Resolved 2026-07-13: constant per-species latent heat is the minimum E4
    contract. Linear and power-law temperature models are deferred.
- [x] What full-physics tolerances are recorded separately from energy identity?
  - Resolved 2026-07-13: use E4-F6's case/device parity policy, initially
    `rtol=1e-10` with scale-derived `atol`. Check `Q = delta_mass * latent_heat`
    separately at conservation tolerance.
- [x] What E4-F3 scratch API does E4-F4 consume without duplication?
  - Resolved 2026-07-13: consume the typed sidecar's substep work and whole-call
    mass accumulator, adding only fixed-shape latent-property and
     `(n_boxes, n_species)` energy buffers.
- [x] Does P2 apply the correction at every fixed substep without changing the
  isothermal contract?
  - Resolved 2026-07-14: yes. All four launches receive the latent sidecar and
    enabled flag; omitted latent heat and exact zero entries use the original
   isothermal arithmetic. `thermal_work` remains validated and deferred.
- [x] How does P3 aggregate and initialize opt-in energy output?
  - Resolved 2026-07-14: after successful atomic preflight, clear the
    caller-owned `(n_boxes, n_species)` output once and use one writer per
    box/species after all four substeps to reduce accumulated bounded transfer
     times latent heat. Output contents are write-only and never validated/read.
- [x] Do composed scalar and explicit-environment routes preserve the P2/P3
  latent and energy contracts?
  - Resolved 2026-07-14 (issue #1300): independent fresh-state Warp-CPU oracle
    regressions cover both routes; optional CUDA uses the existing
    `cuda_device` availability skip. Both retain applied-transfer accounting,
    signed energy, unchanged gas concentration, and a zero-latent energy column.
