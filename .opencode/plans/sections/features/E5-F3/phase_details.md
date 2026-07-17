# Phase Details

- [x] **E5-F3-P1:** Prove and implement a safe charged majorant with unit tests
  - Issue: #1342 | Size: S | Status: Completed
  - Goal: Compute a finite non-negative upper bound over every active charged
    pair and expose it through the E5-F1 term-majorant dispatcher.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Delivered: Private charged hard-sphere majorant scans every unique compact
    active pair, sanitizes candidates, and is dispatched internally without
    changing public charged execution.
  - Tests: Independent deterministic physics, invalid/zero candidate, sparse
    active-list, per-box, dispatcher-addition, and Brownian-regression coverage.

- [x] **E5-F3-P2:** Enable charged-only one-pass execution with unit tests
  - Issue: #1343 | Size: S | Status: Completed
  - Goal: Register exact charged-only particle-resolved capability and route it
    through the existing bounded sampler and charge-safe merge exactly once.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Delivered: Charged rates and a compact O(A²) majorant use prepared private
    per-particle total mass; charged mode forces finite-charge preflight and
    retains one selector/apply pass with charge-conserving merges.
  - Tests: Deterministic rate/majorant, multi-species, invalid-input,
    ownership/capacity, invariants, mass/charge conservation, and aggregate
    stochastic charged-only coverage.

- [ ] **E5-F3-P3:** Enable Brownian-plus-charged execution with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Sum Brownian and charged rates and majorants before one acceptance
    decision without launching mechanism-specific stochastic passes.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Additive rate/majorant probes, one-pass RNG and merge evidence,
    bounded repeated-seed statistics, canonical-order equivalence, legacy
    Brownian regression, caller buffers, mixed-scale and multi-box conservation.

- [ ] **E5-F3-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document charged-only and Brownian-plus-charged direct execution,
    required inputs, tested devices, ownership guarantees, and limitations.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/coagulation_strategy_system.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, docstrings, and E5
    plan sections
  - Tests: Markdown link/reference validation plus focused import/signature
    checks for any executable snippets.
