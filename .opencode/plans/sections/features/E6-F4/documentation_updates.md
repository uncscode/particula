# Documentation Updates

- Update `AGENTS.md` with the charged direct-wall-loss contract, focused test
  commands, image-charge-at-zero-potential rule, field semantics, exact neutral
  fallback, and caller-owned RNG/charge guidance.
- Update the relevant `docs/Features/` GPU/wall-loss page with supported
  spherical/rectangular charged configuration, SI units, coefficient equation,
  fixed-slot clearing, Warp CPU baseline, and optional CUDA evidence.
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
