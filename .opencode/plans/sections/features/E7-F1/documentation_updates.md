# Documentation Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with the new
  execution-context import, typed backend/device selection, CPU reference
  behavior, ownership/mutation contract, and explicit no-fallback rule.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` to record E7-F1 phase
  progress and the chosen separate-context API location without claiming that
  GPU adapters or resident loops have shipped.
- Add or update an architecture guide under `.opencode/guides/architecture/`
  describing the boundary between strategies, runnables, execution contexts,
  process adapters, and future sessions.
- Add a small CPU-only usage snippet showing explicit request construction,
  context validation, execution, and result/state identity. Keep direct GPU
  examples unchanged until E7-F2 through E7-F5 supply executable paths.
- Document the capability-extension contract for E7-F2, E7-F3, and E7-F4 and
  the policy handoff to E7-F6.
- Update these E7-F1 plan sections and phase statuses as implementation ships.
- Do not update `README.md` or `AGENTS.md` unless implementation introduces a
  generally advertised quick-start API or contributor workflow requirement.

Validation: run relevant documentation regressions, verify all relative links,
and run `mkdocs build --strict` with warnings treated as failures.
