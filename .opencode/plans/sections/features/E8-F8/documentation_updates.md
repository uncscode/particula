# Documentation Updates

- Completed P1 (#1595): Added
  `docs/Examples/gpu_resident_graph_capture.py` as the canonical lazily
  qualified native-CUDA-only fixed-identity capture/replay walkthrough.
- Completed P1 (#1595): Added its link and native-CUDA-only invocation summary
  to `docs/Examples/index.md`.
- Completed P2 (#1596): Added `docs/Features/gpu_graph_capture.md` as the
  operator runbook covering setup, ownership, RNG, replay, synchronization,
  invalidation, recapture, teardown, failures, limitations, and clean skips.
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
- Completed P1 (#1595): Added hardware-free contract and optional native-CUDA
  smoke coverage in `particula/tests/gpu_resident_graph_capture_docs_test.py`.
- Completed P2 (#1596): Added hardware-free runbook contract coverage in
  `particula/tests/gpu_graph_capture_runbook_docs_test.py`.
- P3--P4 closeout documentation updates remain pending; no closeout report,
  roadmap/status update, or parent-plan reconciliation was completed.
