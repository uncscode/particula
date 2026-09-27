# Outcomes and Guardrails

**Primary outcome:** Supported v0.3.0 CPU workflows use a flat three-container
`Aerosol`, retain runnable composition and scientific behavior, and no longer
depend on `ParticleRepresentation`, `GasSpecies`, or `Atmosphere`.

## Secondary goals

- Publish explicit units, shapes, ordered species alignment, mutation and
  ownership contracts, including non-unit-volume behavior.
- Preserve practical initialization/presets, necessary getters/setters and
  identity-preserving whole-container access and replacement.
- Migrate scientific regression tests and supported examples before removing
  obsolete facades, bridges, exports and facade-only tests/messages.
- Publish executable migration guidance and v0.3.0 breaking-change notes.

## Guardrails

- Complete and validate M1 before M2, then M3, M4, M5 and M6; no parallel
  implementation and no early deletion.
- CPU process execution remains single-box. Data containers retain their
  existing schemas and names; no precision or storage redesign is included.
- Scientific configuration stays on processes. There is no permanent new
  compatibility facade or duplicate authoritative state.
- Getters return held container objects; accepted setters retain supplied
  identity. Rejection leaves current state unchanged. Coordinated replacement
  validates all three candidates before publishing any replacement.
- Direct mutation remains subject to process validation. CPU replacement never
  automatically rebinds prepared GPU/resident state.
- Retain CPU↔Warp transfer helpers and current export, device, ownership, RNG,
  checkpoint and graph-capture boundaries. No hidden transfer, synchronization,
  fallback, new GPU public API or new performance claim.
- Temporary compatibility must have an explicit owner, consumer list and
  removal gate, and must be gone before M6 closes.
