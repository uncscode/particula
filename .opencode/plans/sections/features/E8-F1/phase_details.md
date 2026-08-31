# Phase Details

- [x] **E8-F1-P1:** Declare graph-capture capability and compatibility signature with unit tests
  - Issue: #1547 | Size: S | Status: Delivered
  - Goal: Define a concrete-only capture capability result and immutable
    resident compatibility signature without importing Warp at declaration
    time or claiming capture support on CPU.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Delivered: `graph_capture.py` is Warp-import-free and declares
    caller-probed capability plus exact identity-based request signatures.
    `graph_capture_test.py` covers capability ordering, import/export boundaries,
    carrier validation, real Warp-guarded request compatibility, representative
    drift reasons, and stable RNG-array identity.

- [x] **E8-F1-P2:** Implement capture lifecycle and explicit invalidation with unit tests
  - Issue: #1548 | Size: S | Status: Delivered
  - Goal: Add host-only immutable capture-state transitions and deterministic
    invalidation reasons without native capture, replay, or resident mutation.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Delivered: Exact lifecycle metadata covers `READY`, `CAPTURED`,
    `INVALIDATED`, `FAULTED`, `RETIRED`, and `CLOSED`. The direct-module API
    retains P1 carrier identities, implements first-reason-wins invalidation,
    explicit renewal after retirement, failure classification, and idempotent
    paths where specified. It does not inspect resident bindings or perform
    native graph work.
  - Tests: Hardware-free transition-table, exact-type, identity, first-reason,
    failure-classification, direct-import-only, and forbidden-import subprocess
    coverage in `graph_capture_test.py`.

- [x] **E8-F1-P3:** Define recapture gates and resident binding validation with integration tests
  - Issue: #1549 | Size: S | Status: Delivered
  - Goal: Bind lifecycle checks to the exact resident request and require
    explicit recapture after any device, dimension, schedule, map, or resource
    identity change.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/resident_scheduler.py`,
    `particula/execution/tests/graph_capture_test.py`,
    `particula/execution/tests/full_loop_test.py`
  - Delivered: `ResidentGraphCaptureBinding` retains exact resident identities
    and owns lifecycle gate transitions. A private one-time attachment binds it
    to the final frozen request. The optional scheduler binding gates before
    token entry and preserves scheduler ordering and writer-failure behavior.
    Captured signature drift invalidates metadata; explicit retirement, renewal,
    and completion are required before later admission.
  - Tests: `graph_capture_test.py` covers exact identity, attachment, gate,
    drift, payload compatibility, and lifecycle cases; `full_loop_test.py`
    covers scheduler pre-token rejection, dispatch preservation, and writer
    failure. CUDA admission is optional; Warp CPU is rejected.

- [ ] **E8-F1-P4:** Update development documentation
  - Issue: #1550 | Size: XS | Status: In Progress
  - Goal: Publish the contract boundaries and handoff requirements for E8-F2
    through E8-F8 without advertising executable graph replay prematurely.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, relevant
    E8 plan sections
  - Files changed: `AGENTS.md`, `docs/Features/Roadmap/data-oriented-gpu.md`,
    `particula/execution/tests/graph_capture_docs_test.py`,
    `particula/execution/tests/exports_test.py`, and
    `particula/tests/execution_exports_test.py`.
  - Validation on 2026-08-30: focused developer-document and export checks
    passed (17 passed). The untargeted `.opencode/tools/run_pytest.py` passed
    with 6381 passed, 9 skipped, 1 xfailed, and 93.59% coverage. `mkdocs build
    --strict` is unavailable because no supported MkDocs runner is available.
    P4 and its parent handoff remain incomplete.
  - No user example is in scope.
