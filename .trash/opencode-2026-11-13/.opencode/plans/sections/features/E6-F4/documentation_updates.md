# Documentation Updates

- [x] #1414 / P6 updates `AGENTS.md`, the wall-loss feature page, architecture
  guidance, and ADR-001 for the direct-only charged boundary: concrete imports,
  SI configuration, signed spherical and magnitude-based rectangular field
  resolution, zero-potential image charge, and exact zero-charge neutral
  fallback are explicit.
- [x] P6 records caller-owned charge/RNG state, fixed-slot clearing, explicit
  transfer and synchronization, preflight non-mutation, the post-launch
  rollback limit, focused commands, Warp CPU baseline, optional CUDA skips, and
  deferred scope without adding a runnable, fallback, or performance claim.
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
- [ ] Add or extend a `docs/Examples/` low-level example only if it can preserve
  explicit CPU-to-Warp transfers and does not imply a high-level runnable;
  otherwise defer the integrated example to E6-F9.
- [x] E6-F4 plan references preserve the shipped E6-F3 → E6-F4 → E6-F9
  relationship and sibling boundaries.
- [x] Supported versus deferred behavior explicitly excludes hidden transfer or
  CPU fallback, dynamic slots, backend selector, scheduler, graph capture,
  differentiability, exact cross-backend RNG, mandatory CUDA, and performance
  claims.
