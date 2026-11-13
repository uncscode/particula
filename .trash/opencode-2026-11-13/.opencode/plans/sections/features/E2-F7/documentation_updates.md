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
- P4 promotes `docs/Features/Roadmap/condensation-stiffness-study.md` into the
  canonical recommendation record, naming `fixed_count_substeps_4` as the
  preferred fixed-shape foundation while keeping the shipped production scope
  particle-only, tying any future gas-coupled production claim to same-issue
  conservation coverage in
  `particula/integration_tests/condensation_particle_resolved_test.py`, and
  keeping the E2-F2 environment-shape plus E2-F6 `float64` evidence boundaries
  explicit.
- P4 updates `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/Roadmap/warp-autodiff-limitations.md` to summarize that final
  recommendation, the fixed-loop/graph-capture constraints, and the still
  deferred gas-coupled production gate without duplicating the full evidence
  tables.
- P4 also normalizes plan-section rerun references so executable focused
  verification points at `_condensation_test_support.py` selectors and
  `particula/dynamics/condensation/tests/staggered_stability_test.py -m slow
  -v` instead of placeholder wrapper files.

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
