# Scope

E8-F8 publishes and regression-tests the final supported graph-capture usage
and operations contract, then reconciles the Epic H roadmap and records exact
closeout evidence. It consumes implementation, validation, scaling, memory, and
profiling outputs from E8-F1 through E8-F7; it does not redesign capture.

## In Scope

- Add a canonical runnable example for setup, explicit RNG initialization,
  capture, repeated replay, invalidation handling, teardown, and recapture.
- Add hardware-free documentation contract tests plus CUDA-gated execution
  coverage that passes or cleanly skips without CPU fallback.
- Publish an operator runbook with supported preconditions, state ownership,
  recapture decision table, failure handling, diagnostics, and reproduction
  commands.
- Document recapture triggers for device, shape, process order/configuration,
  session/request, communication map, diagnostics, registry/resource identity,
  lifecycle, checkpoint/restart, and explicit teardown changes.
- State limitations: CUDA qualification, fixed shapes/order/maps, opaque
  nonserializable graph handles, explicit synchronization, no migration,
  fallback, automatic recapture, rollback, retry, dynamic resizing, or portable
  performance guarantee.
- Build a dated closeout matrix linking literal command output and artifacts for
  every Epic H exit criterion; unavailable required commands keep the epic open.
- Reconcile plan and durable documentation labels so T7 owns profiling and
  machine-bounded recommendations, while T8 owns the example, runbook,
  limitations, documentation reconciliation, and closeout. P4 remains in
  progress pending consistent focused, active split-plan, and strict MkDocs
  validation evidence for the current revision.

## Delivered in P1 (#1595)

- `docs/Examples/gpu_resident_graph_capture.py` lazily qualifies native CUDA,
  publishes resources, initializes streams once, captures, replays exactly
  twice, synchronizes before observation, invalidates structurally, retires,
  renews, freshly captures, and closes in order.
- `docs/Examples/index.md` publishes the source link and native-CUDA-only
  command/limitation summary.
- `particula/tests/gpu_resident_graph_capture_docs_test.py` adds hardware-free
  contract coverage and an optional native-CUDA smoke row.

## Delivered in P2 (#1596)

- `docs/Features/gpu_graph_capture.md` documents the concrete-only,
  native-CUDA-qualified operator lifecycle: setup, resource publication,
  stream initialization, capture, replay, structural invalidation, retirement,
  renewal, fresh recapture, teardown, failure handling, and limitations.
- `particula/tests/gpu_graph_capture_runbook_docs_test.py` adds a hardware-free
  runbook contract covering lifecycle, recapture triggers and non-triggers,
  failure boundaries, limitations, commands, links, and stdlib-only imports.

## Delivered in P3 (#1597)

- `docs/Features/Roadmap/graph-capture-closeout.md` records the dated,
  fail-closed closeout ledger as `UNSHIPPED/BLOCKED`, with unavailable evidence
  and promotion requirements for H1--H11.
- `particula/tests/gpu_graph_capture_closeout_docs_test.py` validates the
  closeout schema and evidence traceability without requiring CUDA, Warp,
  artifacts, Git history, or plan-file reads.

P4 roadmap-closeout documentation reconciliation remains in progress pending
consistent current-revision validation evidence. P3 records blocked evidence
only and neither P3 nor P4 promotes Epic
H or claims CUDA evidence;
E8-F2, E8-F3, E8-F6, E8-F7/T7, and designated-device final-revision H1--H11 evidence
remain required.

## Out of Scope

- New capture/replay runtime semantics, public package exports, kernels,
  scientific equations, tolerances, scheduler order, or RNG behavior.
- Automatic recapture, backend/device migration, CPU fallback, graph-handle
  checkpointing, dynamic process selection, resizing, or compaction.
- Re-running E8-F6/E8-F7 studies with different methodology or claiming their
  machine-specific results generalize to other systems.
- Promoting Epic H when a required result is missing, inferred, stale, or only
  represented by an optional CUDA clean skip.
