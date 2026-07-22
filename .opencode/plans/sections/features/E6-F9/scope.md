# Scope

E6-F9 integrates and publishes the bounded low-level capabilities delivered by
E6-F1 through E6-F8. It adds validation and documentation rather than new
physics or a runtime scheduler.

## In Scope

- Multi-process fixed-shape fixtures for condensation, coagulation, dilution,
  neutral and charged wall loss, slot management, and nucleation. Neutral wall
  loss owns the integrated statistical fixture; charged mode receives a
  deterministic integrated case plus its E6-F4 statistical evidence.
- Required Warp CPU integration coverage, with optional CUDA coverage that
  skips cleanly when unavailable.
- Process-specific parity, conservation/loss-budget, diagnostics, shape,
  device, dtype, identity, RNG persistence, and failure-before-mutation checks.
- A runnable explicit CPU-to-Warp setup, direct-call sequence with no
  intermediate host transfer, and explicit final CPU checkpoint restore.
- Regression tests for the example's CPU-only guard and Warp-enabled path.
- Cross-links for E6 and E6-F1 through E6-F9 in
  `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/Roadmap/index.md`.
- An E6 exit-bar checklist that permits closeout only after all eight upstream
  features and this feature are shipped and their focused evidence passes.

## Out of Scope

- User-facing backend selection, a high-level GPU `Runnable`, automatic process
  ordering, resident-loop scheduling, or multi-box transport; these belong to
  Epic G.
- New dilution, wall-loss, slot, exhaustion, or nucleation physics beyond the
  contracts owned by E6-F1 through E6-F8.
- Hidden transfers, CPU fallback, dynamic GPU resizing, graph capture,
  differentiability, or performance claims.
- Exact CPU/Warp RNG stream equality or mandatory CUDA hardware.
