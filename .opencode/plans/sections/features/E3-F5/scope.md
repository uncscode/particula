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
