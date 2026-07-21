# Open Questions

- [x] What makes a slot available for overwrite?
  - Resolved 2026-07-21: only the canonical all-zero record (all species mass,
    concentration, and charge) is free; contradictory partial state is invalid.
- [x] How are multiple requests assigned?
  - Resolved 2026-07-21: valid request prefixes map by rank to ascending free
    indices independently in each box.
- [x] Does E6-F5 recover from insufficient capacity?
  - Resolved 2026-07-21: no; activation fails before mutation and E6-F6 owns
    resampling or representative-volume scaling.
- [x] Where do diagnostics live?
  - Resolved 2026-07-21: in return values/optional caller-owned sidecars, never
    as new `ParticleData` or `WarpParticleData` fields.
- [ ] Should zero-box and zero-slot containers be accepted as exact no-ops or
  rejected consistently by both APIs?
  - Open: decide against existing Warp launch constraints during P1/P3 and
    record the behavior in tests and API docs.
- [ ] Must GPU diagnostic output buffers reject all aliasing with particle and
  request arrays, or is dtype separation sufficient for safe non-aliasing?
  - Open: resolve during P3 preflight design and test the chosen rule.
