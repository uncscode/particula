# Scope

E5-F9 publishes the final user-facing GPU coagulation contract and direct
example, reconciles both GPU roadmap pages with assigned plan/child IDs and
release artifacts, and performs an explicitly dependency-gated E5 closeout.

## In Scope

- Publish the supported particle-resolved direct GPU coagulation mechanisms,
  configuration semantics, import path, mutation/sidecar ownership, validation
  behavior, device policy, distribution limits, and focused test commands.
- Add `docs/Examples/gpu_coagulation_direct.py`, defaulting to Warp CPU when
  Warp is installed, with explicit CPU↔Warp transfers, caller-owned collision
  buffers and persistent RNG state, deterministic output, and a clean no-Warp
  path.
- Test example import/execution, lazy optional-runtime loading, buffer/RNG reuse,
  output claims, support-boundary wording, roadmap IDs, and link resolution.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/Roadmap/index.md` with E5, E5-F1 through E5-F9, completed scope,
  stable artifact links, and the next epic's status.
- Link the support guide and example from `docs/Examples/index.md`,
  `docs/Features/coagulation_strategy_system.md`, and the foundations guide.
- Verify E5-F1 through E5-F8 are shipped, their required evidence passes, all
  artifacts resolve, and E5-F9's own pre-closeout phases pass before changing
  E5 to shipped and Epic F from pending to active.

## Out of Scope

- New coagulation physics, formulas, kernels, configuration, or public runtime
  behavior; those belong to E5-F1 through E5-F7.
- Reimplementing the E5-F8 condensation walkthrough or changing condensation
  physics.
- High-level `Aerosol`/`Runnable` integration, automatic backend selection,
  implicit CPU fallback/transfers, general GPU production claims, required
  CUDA support, graph capture, performance optimization, or broad autodiff.
- DNS turbulence, non-unit sedimentation efficiency, unsupported binned or
  moving-bin GPU execution, and unapproved mechanism combinations.
- Marking E5 shipped, or Epic F active, based only on documentation completion
  when any upstream child, validation command, artifact, or link is incomplete.
