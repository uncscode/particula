# Milestones and Timeline

## Milestone 1: Coagulation Stochastic Correctness

- Tracks: E3-F1, E3-F2
- Goal: repeated-call RNG behavior is fixed and mixed-scale sampling behavior is
  tested, characterized, and either hardened or explicitly bounded.
- Exit criteria: RNG state persistence tests pass; mixed NPF/droplet scenarios
  have regression coverage and documentation notes.
- Evidence to ship: `particula/gpu/kernels/tests/coagulation_test.py` contains
  persisted-`rng_states` regressions plus mixed-scale acceptance/conservation
  coverage, and `docs/Features/Roadmap/data-oriented-gpu.md` records the chosen
  mixed-scale decision with reproduction commands.

## Milestone 2: Coagulation Architecture Decision

- Tracks: E3-F3
- Goal: current one-thread-per-box design has measured scaling evidence and a
  decision record.
- Exit criteria: roadmap or architecture note explains accepted limits and
  names follow-up criteria for a parallel-within-box variant.
- Evidence to ship: `particula/gpu/tests/benchmark_test.py` remains the single
  benchmark entrypoint, while `docs/Features/Roadmap/data-oriented-gpu.md`
  captures exact command, hardware context, and the accepted single-box versus
  multi-box usage boundary.

## Milestone 3: Low-Level API Discoverability

- Tracks: E3-F4
- Goal: supported kernel import paths are clear and users have a direct GPU
  quick-start for condensation and coagulation.
- Exit criteria: quick-start example runs on Warp CPU and skips/extends to CUDA
  cleanly when available.
- Evidence to ship: public import assertions land in
  `particula/gpu/tests/kernel_exports_test.py` or equivalent export coverage,
  and the quick-start plus troubleshooting live under `docs/Examples/` and the
  GPU roadmap docs.

## Milestone 4: Validation Policy

- Tracks: E3-F5
- Goal: pytest policy codifies Warp CPU, CUDA-if-available, parity, stochastic,
  and benchmark expectations.
- Exit criteria: docs and tests consistently apply the policy without requiring
  CUDA hardware.
- Evidence to ship: `particula/conftest.py`, `pyproject.toml`, and
  `particula/gpu/tests/cuda_availability.py` define the reusable policy, while
  `.opencode/guides/testing_guide.md` records CPU-required and CUDA-optional
  validation commands.

## Milestone 5: Latent-Heat CPU Reference Material

- Tracks: E3-F6, E3-F7
- Goal: CPU latent-heat example and conservation baseline are available for
  later GPU parity work.
- Exit criteria: runnable docs example and integration-level conservation test
  pass in normal CPU test environments.
- Evidence to ship:
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` and
  `.ipynb` provide the runnable example, while
  `particula/integration_tests/condensation_latent_heat_conservation_test.py`
  captures the CPU conservation baseline.

## Sequencing Notes

The first two milestones should land before export-policy and pytest-policy
work so that downstream documentation reflects final stochastic semantics. The
latent-heat work can proceed in parallel once documentation ownership is clear.
