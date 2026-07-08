# Dependencies

## Internal Dependencies

- **E3-F2:** Required before final interpretation. Mixed-scale sampling
  behavior affects the benchmark target and the fairness of any scaling claim.
- **E3-F1:** Provides RNG persistence context that may affect repeated-step
  benchmark behavior and reproducibility.
- **Epic E3 roadmap:** `docs/Features/Roadmap/data-oriented-gpu.md` remains the
  source of truth for Epic C status and design guardrails.

## Code and Test Dependencies

- `particula/gpu/kernels/coagulation.py` for the implementation being measured.
- `particula/gpu/tests/benchmark_test.py` for opt-in benchmark execution.
- `particula/gpu/tests/benchmark_helpers_test.py` for fast helper regression
  coverage.
- `particula/gpu/tests/cuda_availability.py` and Warp runtime behavior for CUDA
  skip handling.

## External Dependencies

- CUDA-capable hardware is needed for measured GPU benchmark updates.
- NVIDIA Warp remains optional in environments where CUDA is unavailable.
- No new runtime package dependencies are expected.

## Sibling Feature Awareness

- E3-F4 and later Epic C tracks should consume the documented decision instead
  of rediscovering the same one-thread-per-box limitation.
- If E3-F3 creates a follow-up parallel-within-box scope, later features should
  reference that follow-up rather than expanding E3-F3 beyond documentation and
  benchmark evidence.

## Phase Ordering Notes

- P1 depends on E3-F1 and E3-F2 being stable enough to reproduce benchmark runs
  with the intended RNG and mixed-scale behavior; do not refresh measurements
  against a sampler contract that is still changing.
- P2 follows P1 because the documented scaling limit must cite the exact matrix
  and environment evidence gathered in the benchmark refresh.
- P3 follows P2 so user-facing boundaries reference measured limits instead of
  assumptions.
- P4 is conditional and last: only open a parallel-within-box follow-up when P2
  and P3 show the current one-thread-per-box path is outside Epic C's accepted
  usage envelope.
