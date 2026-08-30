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

- [ ] **E8-F1-P3:** Define recapture gates and resident binding validation with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Bind lifecycle checks to the exact resident request and require
    explicit recapture after any device, dimension, schedule, map, or resource
    identity change.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/resident_scheduler.py`,
    `particula/execution/tests/graph_capture_test.py`,
    `particula/execution/tests/full_loop_test.py`
  - Tests: Exact session/registry/guard/request acceptance; every documented
    structural recapture trigger; stable active-slot payload changes remain
    compatible; no CPU fallback; optional CUDA capability rows cleanly skip.

- [ ] **E8-F1-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish the contract boundaries and handoff requirements for E8-F2
    through E8-F8 without advertising executable graph replay prematurely.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, relevant
    E8 plan sections
  - Tests: Documentation contract assertions where applicable and
    `mkdocs build --strict`.
