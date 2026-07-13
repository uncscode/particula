# Open Questions

- [x] Final-substep or full-call energy?
  - Resolved 2026-07-12: all four substeps, matching E4-F3 and CPU call semantics.
- [x] Which transfer drives energy?
  - Resolved 2026-07-12: bounded mass actually applied after clamping.
- [x] Add diagnostics to GPU containers?
  - Resolved 2026-07-12: no; use fixed-shape caller-owned sidecars.
- [ ] Should opt-in energy be only an output keyword or join a future result
  object while preserving the default two-item return?
- [ ] Is `(n_boxes, n_species)` the only production granularity, or should
  particle-resolved scratch remain available strictly for tests/debugging?
- [ ] Does E4-F1 provide constant latent heat only or a numeric temperature
  model that E4-F4 evaluates on device?
- [ ] What full-physics tolerances are recorded separately from energy identity?
- [ ] What final E4-F3 scratch API must E4-F4 consume without duplication?
