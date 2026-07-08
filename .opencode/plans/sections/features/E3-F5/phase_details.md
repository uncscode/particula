# Phase Details

- [ ] **E3-F5-P1:** Define Warp pytest markers and device options with hook tests
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: No earlier E3-F5 phase. This policy foundation can start once
    Epic C keeps the marker axes as `warp`, `cuda`, `gpu_parity`, and
    `stochastic`, without waiting for E3-F4 documentation wording.
  - Goal: Register the project-level marker/option contract for Warp, CUDA, GPU
    parity, and stochastic parity without changing default skip behavior.
  - Files: `particula/conftest.py`, `pyproject.toml`,
    `particula/tests/pytest_marker_policy_test.py` or
    `particula/tests/benchmark_option_test.py`
  - Tests: Add pytest hook regression tests that verify marker registration and
    any new collection behavior; existing benchmark option tests remain green.

- [ ] **E3-F5-P2:** Standardize Warp device fixtures and CUDA skip helpers with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F5-P1 defining the marker and option contract so helper names,
    fixture scope, and skip semantics align with the registered policy.
  - Goal: Provide a reusable helper/fixture contract for Warp CPU plus
    CUDA-if-available parametrization and CUDA-only skip behavior.
  - Files: `particula/gpu/tests/cuda_availability.py` or a new adjacent helper,
    `particula/gpu/tests/cuda_availability_test.py`.
  - Tests: Unit tests for CPU-only, CUDA-available, and warning-suppressed helper
    behavior using monkeypatch/fakes; no real CUDA requirement.

- [ ] **E3-F5-P3:** Document stochastic parity and floating tolerance policy
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E3-F1 and E3-F2 providing the final RNG and mixed-scale behavior
    assumptions that the tolerance classes must describe, plus E3-F5-P1 naming
    the marker taxonomy used by the policy prose.
  - Goal: Capture deterministic allclose, conservation, and stochastic aggregate
    tolerance classes as the standard for CPU, Warp CPU, and CUDA validation.
  - Files: `.opencode/guides/testing_guide.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`.
  - Tests: Documentation-link or markdown validation where available; no
    production code changes.

- [ ] **E3-F5-P4:** Apply device policy markers and helpers to GPU kernel tests
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F5-P1 and E3-F5-P2 establishing the reusable policy surface,
    with E3-F1 and E3-F2 stable enough that migrated stochastic tests are not
    immediately invalidated by pending RNG or sampler changes.
  - Goal: Migrate key GPU kernel tests to the standardized markers/helpers while
    preserving current Warp CPU and CUDA-if-available coverage.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/environment_test.py`,
    `particula/gpu/tests/conversion_test.py`
  - Tests: Focused pytest runs for GPU kernel tests on Warp CPU; CUDA tests skip
    cleanly or run when local CUDA exists.

- [ ] **E3-F5-P5:** Update release validation documentation for CUDA-optional workflows
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E3-F5-P3 and E3-F5-P4 for the shipped policy and applied helper
    examples. Do not block on E3-F4 quick-start wording; perform any final
    import-path wording sync as a follow-up pass once E3-F4 closes.
  - Goal: Record local/manual CUDA validation commands and acceptance language so
    release checks are clear without requiring CUDA in standard CI.
  - Files: `.opencode/guides/testing_guide.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, optionally release notes or
    contributor docs.
  - Tests: Run documentation validation if configured; verify example commands
    are syntactically correct.
  - Deliverable: Update developer-facing testing and roadmap guidance so the
    final phase leaves one canonical CUDA-optional validation policy.
