# Documentation Updates

- [x] `AGENTS.md` records the P1 direct-wall-loss contract: charged inputs are
  validation-only, rectangular field ownership/schema is explicit, and neutral
  execution remains unchanged.
- [x] Record P2's private primitive boundary in plan content: fp64
  self-potential-ratio/image-enhancement helpers and their test evidence ship,
  but no public or direct-kernel charged coefficient behavior is documented.
- [ ] Defer charged coefficient equations and direct-step CPU-parity claims
  until P3-P5; future documentation must distinguish P1/P2 primitives from
  integrated charged physics.
- Update `.opencode/guides/` testing or architecture references if E6-F3's
  direct-kernel conventions gain charged-specific validation guidance.
- Add or extend a `docs/Examples/` low-level example only if it can preserve
  explicit CPU-to-Warp transfers and does not imply a high-level runnable;
  otherwise defer the integrated example to E6-F9.
- Update E6, E6-F3, E6-F4, and E6-F9 plan cross-references to show the shipped
  dependency and preserve sibling boundaries.
- Document supported vs deferred behavior explicitly: no hidden transfer or CPU
  fallback, dynamic slots, backend selector, scheduler, graph capture,
  differentiability, exact cross-backend RNG, mandatory CUDA, or performance
  claim.
