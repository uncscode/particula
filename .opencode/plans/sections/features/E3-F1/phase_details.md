# Phase Details

- [ ] **E3-F1-P1:** Define RNG API compatibility and seed-once initialization contract with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Decide and codify how `rng_seed`, omitted `rng_states`, provided
    `rng_states`, and any explicit initialization flag/helper interact without
    breaking existing callers.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Compatibility tests for legacy no-`rng_states` calls, provided buffer
    validation, and validation-before-mutation failure paths.

- [ ] **E3-F1-P2:** Add persisted rng_states regression tests for repeated coagulation steps
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove preallocated RNG states advance across repeated calls and are not
    reset to the same seed-derived values each timestep.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Warp CPU and CUDA-if-available tests for non-overwrite, repeated-call
    state advancement, and no manual `rng_seed` increment requirement.

- [ ] **E3-F1-P3:** Implement seed-once persisted RNG semantics in coagulation_step_gpu
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Remove unconditional RNG reinitialization for persisted buffers while
    keeping internal allocation-and-seed convenience for callers that omit
    `rng_states`.
  - Files: `particula/gpu/kernels/coagulation.py`, optional consistency update in
    `particula/gpu/tests/benchmark_test.py`
  - Tests: Co-located regression and compatibility tests from P1/P2 pass on Warp
    CPU and CUDA-if-available.

- [ ] **E3-F1-P4:** Update GPU RNG documentation and graph-capture usage guidance
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document seed-once usage, persistent buffer ownership, and graph-capture
    setup rules.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`, optional examples or
    benchmark comments that show repeated timestep loops.
  - Tests: Documentation link/format validation and any updated example tests if
    executable snippets change.
