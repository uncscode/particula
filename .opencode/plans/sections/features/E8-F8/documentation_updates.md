# Documentation Updates

- Completed P1 (#1595): Added
  `docs/Examples/gpu_resident_graph_capture.py` as the canonical lazily
  qualified native-CUDA-only fixed-identity capture/replay walkthrough.
- Completed P1 (#1595): Added its link and native-CUDA-only invocation summary
  to `docs/Examples/index.md`.
- Completed P2 (#1596): Added `docs/Features/gpu_graph_capture.md` as the
  operator runbook covering setup, ownership, RNG, replay, synchronization,
  invalidation, recapture, teardown, failures, limitations, and clean skips.
- Completed P3 (#1597): Added
  `docs/Features/Roadmap/graph-capture-closeout.md` with dated environment
  metadata, exact commands, literal-result placeholders, artifact provenance,
  changed-target disposition, and fail-closed blockers/status.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` to replace future-tense
  Epic H statements only where supported, preserve unsupported limitations,
  publish the exit-bar disposition, and link the example/runbook/evidence.
- Update `docs/Features/Roadmap/index.md` and `docs/index.md` with discoverable
  links and the evidence-bounded Epic H status.
- Update `AGENTS.md` with stable supported imports, lifecycle rules, recapture
  triggers, and reproduction commands; omit raw machine-specific profiles.
- Update `.opencode/guides/testing_guide.md` only if closeout establishes a
  lasting repository-wide graph-capture validation policy.
- Reconcile `.opencode/plans/sections/epics/E8/child_plans.md`, dependency,
  milestone, and implementation-strategy sections so E8-F7/T7 owns profiling
  and machine-bounded recommendations and E8-F8/T8 owns the example, runbook,
  limitations, documentation reconciliation, and closeout.
- Completed P1 (#1595): Added hardware-free contract and optional native-CUDA
  smoke coverage in `particula/tests/gpu_resident_graph_capture_docs_test.py`.
- Completed P2 (#1596): Added hardware-free runbook contract coverage in
  `particula/tests/gpu_graph_capture_runbook_docs_test.py`.
- Completed P3 (#1597): Added
  `docs/Features/Roadmap/graph-capture-closeout.md` and its hardware-free
  contract in `particula/tests/gpu_graph_capture_closeout_docs_test.py`.
  The record is explicitly `UNSHIPPED/BLOCKED`: it captures unavailable
  E8-F2/E8-F3/F6/F7 closeout evidence without promoting Epic H.
- Completed P4 (#1598): Reconciled durable roadmap/status and E8 plan content
  for T7/T8 ownership. Focused closeout/runbook documentation contracts passed
  (16 passed), `mkdocs build --strict` passed (exit 0), and active split-plan
  validation is recorded. P4 is documentation-only and non-promoting: P3
  remains `UNSHIPPED/BLOCKED`, while E8-F2, E8-F6, E8-F7/T7, and
  designated-device final-revision H1--H11 evidence remain blockers. No CUDA
  evidence is claimed.
