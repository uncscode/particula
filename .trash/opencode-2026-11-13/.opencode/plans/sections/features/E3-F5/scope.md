# Scope

## In Scope

- Add formal pytest markers and/or options for Warp/GPU parity tests in both
  `particula/conftest.py` and `pyproject.toml`.
- Preserve current default behavior: Warp-specific modules skip when Warp is not
  installed; Warp CPU runs when Warp is installed; CUDA is included only when
  Warp reports CUDA availability.
- Standardize fixture/helper usage around `particula/gpu/tests/cuda_availability.py`
  or a small adjacent device-policy helper module.
- Record CUDA local/manual release validation expectations without making CUDA a
  required CI dependency.
- Document deterministic floating-point, conservation, and stochastic parity
  tolerance expectations for CPU, Warp CPU, and CUDA-if-available comparisons.
- Apply the policy to GPU kernel tests, especially coagulation and condensation
  tests that currently define local device fixtures.

## Current Implementation Status

- Shipped in `E3-F5-P1`: marker registration in `particula/conftest.py`, static
  marker parity in `pyproject.toml`, and hook regression tests in
  `particula/tests/pytest_marker_policy_test.py` plus
  `particula/tests/benchmark_option_test.py`.
- Shipped in `E3-F5-P2` and `E3-F5-P3`: the shared `CUDA_SKIP_REASON` helper
  contract plus the published Warp CPU / CUDA-optional / stochastic-tolerance
  documentation policy.
- Shipped in `E3-F5-P4`: test-only marker adoption in
  `particula/gpu/kernels/tests/coagulation_test.py`,
  `particula/gpu/kernels/tests/_condensation_test_support.py`,
  `particula/gpu/kernels/tests/condensation_test.py`,
  `particula/gpu/kernels/tests/condensation_stiffness_test.py`,
  `particula/gpu/kernels/tests/environment_test.py`, and
  `particula/gpu/tests/conversion_test.py`.
- Shipped in `E3-F5-P5`: final release-validation documentation wording in
  `.opencode/guides/testing_guide.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`, with `docs/contribute/`
  `CONTRIBUTING.md` intentionally left unchanged.
- The shipped P4 migration preserved `pytest.importorskip("warp")`, kept
  CPU-first `warp_devices(wp)` parametrization, avoided module-level `cuda`
  over-marking, and limited scope to test-surface policy adoption rather than
  helper API or production-kernel changes.

## Out of Scope

- Rewriting GPU kernels or changing production simulation behavior.
- Making CUDA mandatory in CI or failing CPU-only environments.
- Changing the CPU/GPU container transfer boundary or introducing hidden
  synchronization/fallback behavior.
- Solving E3-F1 RNG semantics or E3-F2 mixed-scale sampler behavior directly;
  this feature consumes those contracts as dependencies.
- Large benchmark expansion beyond policy-related test hook coverage.

## Constraints

- CUDA must remain optional and skipped cleanly when unavailable.
- Tests should remain fast unless explicitly marked as benchmark, slow, or
  performance-intensive.
- Co-located tests must ship with each implementation phase.
