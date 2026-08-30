# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with E8-F7 status,
  measured machine/software identity, exact workload matrix, commands, raw
  artifact references, and the distinction between launch and kernel costs.
- Create or update `docs/Features/gpu_graph_capture_performance.md` with the
  profiling method, warmup/synchronization rules, captured-versus-uncaptured
  results, dominant-kernel metrics, bottleneck table, recommendations, and
  explicit machine/workload limits.
- Update `docs/Features/Roadmap/index.md` if the Epic H progress summary or link
  set changes.
- Update `AGENTS.md` only with stable reproduction commands and supported
  conclusions; do not paste volatile raw profiles or imply portable guarantees.
- Update `.opencode/guides/testing_guide.md` only if the repository-wide
  profiling procedure becomes a lasting policy beyond this feature.
- Reconcile `.opencode/plans/sections/epics/E8/child_plans.md`,
  `dependency_map.md`, and milestone text so T7/E8-F7 profiling does not conflict
  with stale E8-F8 labels.
- Update all E8-F7 plan sections with shipped phase status and final evidence
  links during closeout.
- Add a documentation contract test under `particula/tests/` that asserts the
  exact command, CUDA-only qualification, launch/kernel separation,
  machine-bounded language, and no-fallback limitation.
- Run `mkdocs build --strict`; broken links or missing required evidence block
  publication.
