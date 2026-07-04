# E2-F7 Documentation Updates

## Required Documentation

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with:
  - the stress-case catalog,
  - the explicit GPU timestep stiffness map,
  - comparison of fixed sub-stepping, deterministic staggered/batch-inspired
    options, and semi-implicit/asymptotic candidates,
  - the final integration recommendation,
  - dependency notes for E2-F2 and E2-F6.
- Update `docs/Features/Roadmap/warp-autodiff-limitations.md` with:
  - graph-capture constraints discovered during the study,
  - differentiability implications of hard clamps and in-place updates,
  - recommended deterministic fixed-loop condensation foundation.

## Optional Documentation

- Add a short table or appendix linking stress cases to particle size,
  environment, gas concentration, and observed stable timestep.
- Add follow-up implementation notes for future `WarpEnvironmentData` and gas
  concentration update work if those are blocked by upstream features.

## Documentation Quality Gates

- The final docs must distinguish measured evidence from recommendations.
- Any rejected option must include a concise reason, especially for random
  staggered modes and dynamically adaptive loops.
- Cross-references to E2-F2, E2-F6, and issue #1172 should be present.
