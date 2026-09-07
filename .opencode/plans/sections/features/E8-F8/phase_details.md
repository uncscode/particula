# Phase Details

- [x] **E8-F8-P1:** Runnable graph-capture example with hardware-free contract tests
  - Issue: #1595 | Size: S | Status: Implemented 2026-09-07
  - Goal: Publish one canonical setup/capture/replay/invalidate/teardown example
    that defaults to a clear unsupported path and runs on qualified CUDA.
  - Delivered files: `docs/Examples/gpu_resident_graph_capture.py`,
    `docs/Examples/index.md`, and
    `particula/tests/gpu_resident_graph_capture_docs_test.py`.
  - Delivered behavior: lazy native-CUDA qualification; one resource
    publication/stream initialization; capture; exactly two replays; explicit
    synchronization before observation; structural invalidation, retirement,
    renewal, fresh capture, and ordered teardown.
  - Tests added: hardware-free AST/text/import/unavailable/lifecycle contracts
    and optional CUDA-gated replay smoke coverage.

- [x] **E8-F8-P2:** Operator runbook with recapture triggers and limitation checks
  - Issue: #1596 | Size: S | Status: Implemented 2026-09-07
  - Delivered files: `docs/Features/gpu_graph_capture.md` and
    `particula/tests/gpu_graph_capture_runbook_docs_test.py`.
  - Delivered behavior: fail-closed native-CUDA-qualified setup, replay,
    invalidation, retirement, renewal, explicit fresh recapture, teardown,
    failure handling, and lifecycle/recapture/limitation tables.
  - Tests added: hardware-free assertions for the ordered structural triggers,
    mutable-value non-triggers, lifecycle failures, checkpoint/RNG boundaries,
    literal commands and links, and stdlib-only imports.

- [x] **E8-F8-P3:** Epic closeout evidence matrix and publication checks
  - Issue: #1597 | Size: S | Status: Implemented 2026-09-07
  - Delivered files: `docs/Features/Roadmap/graph-capture-closeout.md` and
    `particula/tests/gpu_graph_capture_closeout_docs_test.py`.
  - Delivered behavior: records a fail-closed `UNSHIPPED/BLOCKED` closeout
    disposition, unavailable final-revision/device/artifact evidence, ordered
    command and provenance ledgers, and the H1--H11 promotion requirements.
    It does not promote Epic H or treat missing E8-F2/E8-F3/F6/F7 evidence as
    passing evidence.
  - Tests added: hardware-free documentation contract coverage for the closeout
    schema, evidence traceability, command ordering, coverage-target
    derivation, optional-CUDA labeling, unsafe provenance rejection, and
    fail-closed missing-row checks.

- [ ] **E8-F8-P4:** Roadmap and development documentation closeout
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Reconcile T7/T8 labels, publish supported conclusions, update durable
    commands, and promote Epic H only when P3 proves the exit bar.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/index.md`, `docs/index.md`, `AGENTS.md`, E8 parent and
    child plan sections
  - Tests: Documentation links/contracts, plan validation, and
    `mkdocs build --strict`.
