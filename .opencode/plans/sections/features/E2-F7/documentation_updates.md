# E2-F7 Documentation Updates

## Required Documentation

- P1 added `docs/Features/Roadmap/condensation-stiffness-study.md` with:
  - the shared deterministic stress-case catalog,
  - baseline metric vocabulary,
  - inclusive threshold semantics,
  - accepted scalar and direct `(n_boxes,)` environment-input assumptions, and
  - explicit particle-only runtime caveat language.
- P1 updated `docs/Features/Roadmap/index.md` to link to the new baseline note.
- P2 updated `docs/Features/Roadmap/condensation-stiffness-study.md` with a
  measured-results section and markdown table that mirrors
  `_RECORDED_TIMESTEP_GRID_BY_CASE`, including stable/unstable labels,
  environment-input mode, caller-owned buffer reuse context, and the explicit
  unchanged-gas particle-only caveat.
- P3 updated `docs/Features/Roadmap/condensation-stiffness-study.md` with
  candidate-specific evidence for `fixed_count_substeps_4` and
  `asymptotic_relaxation`, explicit CPU-reference and baseline-comparison notes,
  graph-capture/autodiff implications, and the documented decision to defer any
  production gas-coupling hook and integration regression to later work.
- P4 updates `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/Roadmap/warp-autodiff-limitations.md` to publish the final
  recommendation, the fixed-loop/graph-capture constraints, and the still
  deferred gas-coupled production gate without duplicating the full evidence
  tables.

## Optional Documentation

- Add a follow-up appendix only if later phases materially expand the current
  fixed-count recommendation or ship a gas-coupled production hook.
- Add follow-up implementation notes for future `WarpEnvironmentData` and gas
  concentration update work if those are blocked by upstream features.

## Documentation Quality Gates

- The docs must distinguish baseline definitions from measured evidence or
  recommendations.
- Any rejected option must include a concise reason, especially for random
  staggered modes and dynamically adaptive loops.
- Cross-references to E2-F2, E2-F6, and issues #1213/#1214 should be present
  where relevant.
