# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` under Epic H Performance
  and Memory with the exact plain reproduction command, date, Warp/Python
  versions, qualified device, matrix, raw artifact path, summary table, memory
  categories, analytical/observed comparison, and machine-bounded caveats.
- Add or update a focused feature report under `docs/Features/` describing the
  benchmark schema, fair captured/uncaptured timing boundary, configured budget,
  structured unavailable rows, and how to interpret logical versus observed
  versus projected bytes.
- Link E8-F7's graph-capture example/limits page to the published evidence but
  leave runnable lifecycle ownership in E8-F7.
- Update `.opencode/guides/testing_guide.md` only if the concrete resident
  benchmark command or artifact convention adds a reusable repository policy;
  preserve `--benchmark` as the only collection-affecting option.
- Update `AGENTS.md` with the focused reproduction command and evidence location
  only when useful for future contributors; do not paste machine-specific
  timings into general quick-start text.
- Keep `.artifacts/benchmarks/` results machine-generated and identify the
  reviewed source-of-record artifact explicitly. Never present unavailable rows
  as zero time or zero memory.
- Reconcile these plan sections and phase states after implementation and run
  documentation contract tests plus `mkdocs build --strict`.
