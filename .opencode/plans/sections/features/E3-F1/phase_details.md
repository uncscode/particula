# Phase Details

- [x] **E3-F1-P1:** Define RNG API compatibility and seed-once initialization contract with tests
  - Issue: #1236 | Size: S | Status: Shipped on 2026-07-08
  - Depends on: No prior feature phase. This contract-setting phase must land
    before repeated-call regressions or implementation changes rely on the final
    `rng_states` semantics.
  - Goal: Decide and codify how `rng_seed`, omitted `rng_states`, provided
    `rng_states`, and an explicit initialization flag interact without breaking
    existing callers.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Shipped details: added keyword-only `initialize_rng: bool = False` to
    `coagulation_step_gpu`; locked omitted-vs-provided `rng_states` behavior;
    preserved validation-before-mutation ordering.
  - Tests: Added compatibility coverage for omitted `rng_states`, caller-owned
    reuse without implicit reset, explicit reset via `initialize_rng=True`,
    wrong-shape/wrong-device validation, and no mutation on failure.

- [ ] **E3-F1-P2:** Add persisted rng_states regression tests for repeated coagulation steps
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F1-P1 defining the supported seed-once contract so the
    regression encodes the intended caller-visible behavior rather than a
    provisional assumption.
  - Goal: Extend beyond the shipped compatibility contract to prove full
    persisted repeated-call semantics across longer timestep sequences.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Warp CPU and CUDA-if-available tests for non-overwrite, repeated-call
    state advancement, and no manual `rng_seed` increment requirement.

- [ ] **E3-F1-P3:** Implement seed-once persisted RNG semantics in coagulation_step_gpu
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F1-P1 and E3-F1-P2. Implementation should satisfy the locked
    compatibility contract and the failing repeated-step regressions before any
    documentation is refreshed.
  - Goal: Broaden the shipped P1 contract into full feature semantics while
    keeping internal allocation-and-seed convenience for callers that omit
    `rng_states`.
  - Files: `particula/gpu/kernels/coagulation.py`, optional consistency update in
    `particula/gpu/tests/benchmark_test.py`
  - Tests: Co-located regression and compatibility tests from P1/P2 pass on Warp
    CPU and CUDA-if-available.

- [ ] **E3-F1-P4:** Update GPU RNG documentation and graph-capture usage guidance
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E3-F1-P3 shipping the final runtime behavior so the docs record
    the exact reset-helper, buffer-ownership, and repeated-step guidance that
    tests already protect.
  - Goal: Document the final feature behavior once P2/P3 runtime scope is fully
    shipped.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`, optional examples or
    benchmark comments that show repeated timestep loops.
  - Tests: Documentation link/format validation and any updated example tests if
    executable snippets change.
