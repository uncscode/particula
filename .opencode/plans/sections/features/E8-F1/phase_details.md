# Phase Details

- [ ] **E8-F1-P1:** Declare graph-capture capability and compatibility signature with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Define a concrete-only capture capability result and immutable
    resident compatibility signature without importing Warp at declaration
    time or claiming capture support on CPU.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Tests: Exact carrier validation; CUDA-capable versus CPU/unsupported
    outcomes; complete signature fields; identity drift and malformed metadata.

- [ ] **E8-F1-P2:** Implement capture lifecycle and explicit invalidation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add legal capture-state transitions and deterministic invalidation
    reasons while preserving active-session, closed-guard, and no-hidden-work
    constraints.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Tests: Transition table, idempotent invalidation/close, illegal replay and
    recapture transitions, no mutation on read-only preflight rejection, and
    writer-failure fault classification.

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
