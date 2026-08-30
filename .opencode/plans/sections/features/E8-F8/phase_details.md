# Phase Details

- [ ] **E8-F8-P1:** Runnable graph-capture example with hardware-free contract tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish one canonical setup/capture/replay/invalidate/teardown example
    that defaults to a clear unsupported path and runs on qualified CUDA.
  - Files: `docs/Examples/gpu_resident_graph_capture.py`,
    `particula/tests/gpu_resident_graph_capture_docs_test.py`
  - Tests: AST/text ownership and sequence checks, import-boundary checks,
    subprocess unsupported-path check, and CUDA-gated replay smoke coverage.

- [ ] **E8-F8-P2:** Operator runbook with recapture triggers and limitation checks
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Give operators a fail-closed lifecycle procedure and complete
    recapture/limitation decision tables.
  - Files: `docs/Features/gpu_graph_capture.md`, documentation contract test
  - Tests: Required trigger, failure, no-fallback, no-automatic-recapture,
    checkpoint, RNG, and reproduction-command assertions.

- [ ] **E8-F8-P3:** Epic closeout evidence matrix and publication checks
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Record dated environment metadata, exact executable targets, literal
    required command results, artifact links, and pass/block disposition.
  - Files: `docs/Features/Roadmap/graph-capture-closeout.md`,
    `particula/tests/gpu_graph_capture_closeout_docs_test.py`
  - Tests: Evidence schema/completeness, command ordering, coverage-target
    derivation, optional-CUDA labeling, and fail-closed missing-row checks.

- [ ] **E8-F8-P4:** Roadmap and development documentation closeout
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Reconcile T7/T8 labels, publish supported conclusions, update durable
    commands, and promote Epic H only when P3 proves the exit bar.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/index.md`, `docs/index.md`, `AGENTS.md`, E8 parent and
    child plan sections
  - Tests: Documentation links/contracts, plan validation, and
    `mkdocs build --strict`.
