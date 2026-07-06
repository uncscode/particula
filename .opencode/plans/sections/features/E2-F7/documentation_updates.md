# E2-F7 Documentation Updates

## Required Documentation

- P1 added `docs/Features/Roadmap/condensation-stiffness-study.md` with:
  - the shared deterministic stress-case catalog,
  - baseline metric vocabulary,
  - inclusive threshold semantics,
  - accepted scalar and direct `(n_boxes,)` environment-input assumptions, and
  - explicit particle-only runtime caveat language.
- P1 updated `docs/Features/Roadmap/index.md` to link to the new baseline note.
- Updates to `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/Roadmap/warp-autodiff-limitations.md` remain for later phases
  when measured bounds or recommendations exist.

## Optional Documentation

- Add a short table or appendix linking stress cases to particle size,
  environment, gas concentration, and observed stable timestep once P2 has
  measured bounds.
- Add follow-up implementation notes for future `WarpEnvironmentData` and gas
  concentration update work if those are blocked by upstream features.

## Documentation Quality Gates

- The docs must distinguish baseline definitions from measured evidence or
  recommendations.
- Any rejected option must include a concise reason, especially for random
  staggered modes and dynamically adaptive loops.
- Cross-references to E2-F2, E2-F6, and issue #1213 should be present where
  relevant.
