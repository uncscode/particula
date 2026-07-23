# Documentation Updates

- [x] `AGENTS.md` records the direct-wall-loss contract: charged selection is
  configuration-selected in the existing step; nonzero charge uses image/drift
  composition, zero charge falls back to neutral execution, and rectangular
  field ownership/schema is explicit.
- [x] Record P2's private primitive boundary in plan content: fp64
  self-potential-ratio/image-enhancement helpers and their test evidence ship;
  P4 consumes them privately without adding a public charged-coefficient API.
- [x] `.opencode/guides/architecture/architecture_outline.md` records the
  configuration-selected fixed-slot boundary, private helper ownership,
  nonzero-charge charged selection, zero-charge neutral fallback, frozen
  preflight, and external RNG-sidecar ownership.
- [x] #1413 records completed direct-step charged CPU/Warp coefficient parity,
  exact zero-charge fallback/ownership, invalid/no-op non-mutation, and the
  frozen eight-stratum exact-binomial survival validation in the plan and
  regression module only. No user-facing documentation changed and no broader
  parity or RNG-replay claim is made.
- Add or extend a `docs/Examples/` low-level example only if it can preserve
  explicit CPU-to-Warp transfers and does not imply a high-level runnable;
  otherwise defer the integrated example to E6-F9.
- Update E6, E6-F3, E6-F4, and E6-F9 plan cross-references to show the shipped
  dependency and preserve sibling boundaries.
- Document supported vs deferred behavior explicitly: no hidden transfer or CPU
  fallback, dynamic slots, backend selector, scheduler, graph capture,
  differentiability, exact cross-backend RNG, mandatory CUDA, or performance
  claim.
