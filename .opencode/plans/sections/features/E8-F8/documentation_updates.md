# Documentation Updates

- Create `docs/Examples/gpu_resident_graph_capture.py` as the canonical runnable
  fixed-shape capture/replay example.
- Create `docs/Features/gpu_graph_capture.md` as the operator runbook covering
  setup, ownership, RNG, replay, synchronization, diagnostics, invalidation,
  recapture, teardown, failures, limitations, and clean skips.
- Create `docs/Features/Roadmap/graph-capture-closeout.md` with dated
  environment metadata, exact commands, literal results, artifact links,
  changed executable targets, metric disposition, and final blockers/status.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` to replace future-tense
  Epic H statements only where supported, preserve unsupported limitations,
  publish the exit-bar disposition, and link the example/runbook/evidence.
- Update `docs/Features/Roadmap/index.md` and `docs/index.md` with discoverable
  links and the evidence-bounded Epic H status.
- Update `AGENTS.md` with stable supported imports, lifecycle rules, recapture
  triggers, and reproduction commands; omit raw machine-specific profiles.
- Update `.opencode/guides/testing_guide.md` only if closeout establishes a
  lasting repository-wide graph-capture validation policy.
- Reconcile `.opencode/plans/sections/epics/E8/child_plans.md`, dependency and
  milestone sections so E8-F7 is profiling and E8-F8 is T8
  example/runbook/closeout, then update shipped phase/status evidence.
- Add and maintain hardware-free documentation contract tests under
  `particula/tests/`; run `mkdocs build --strict` before publication.
