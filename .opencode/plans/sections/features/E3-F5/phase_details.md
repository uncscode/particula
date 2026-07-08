# Phase Details

- [ ] **E3-F5-P1:** Define Warp pytest markers and device options with hook tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Register the project-level marker/option contract for Warp, CUDA, GPU
    parity, and stochastic parity without changing default skip behavior.
  - Files: `particula/conftest.py`, `pyproject.toml`,
    `particula/tests/*pytest*_test.py` or adjacent hook tests.
  - Tests: Add pytest hook regression tests that verify marker registration and
    any new collection behavior; existing benchmark option tests remain green.

- [ ] **E3-F5-P2:** Standardize Warp device fixtures and CUDA skip helpers with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Provide a reusable helper/fixture contract for Warp CPU plus
    CUDA-if-available parametrization and CUDA-only skip behavior.
  - Files: `particula/gpu/tests/cuda_availability.py` or a new adjacent helper,
    `particula/gpu/tests/cuda_availability_test.py`.
  - Tests: Unit tests for CPU-only, CUDA-available, and warning-suppressed helper
    behavior using monkeypatch/fakes; no real CUDA requirement.

- [ ] **E3-F5-P3:** Document stochastic parity and floating tolerance policy
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Capture deterministic allclose, conservation, and stochastic aggregate
    tolerance classes as the standard for CPU, Warp CPU, and CUDA validation.
  - Files: `.opencode/guides/testing_guide.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`.
  - Tests: Documentation-link or markdown validation where available; no
    production code changes.

- [ ] **E3-F5-P4:** Apply device policy markers and helpers to GPU kernel tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Migrate key GPU kernel tests to the standardized markers/helpers while
    preserving current Warp CPU and CUDA-if-available coverage.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/environment_test.py`, and related GPU test
    modules as needed.
  - Tests: Focused pytest runs for GPU kernel tests on Warp CPU; CUDA tests skip
    cleanly or run when local CUDA exists.

- [ ] **E3-F5-P5:** Update release validation documentation for CUDA-optional workflows
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Record local/manual CUDA validation commands and acceptance language so
    release checks are clear without requiring CUDA in standard CI.
  - Files: `.opencode/guides/testing_guide.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, optionally release notes or
    contributor docs.
  - Tests: Run documentation validation if configured; verify example commands
    are syntactically correct.
