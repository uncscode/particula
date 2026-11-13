# Infrastructure Reuse

## Existing Implementation

- `particula/gpu/kernels/coagulation.py` contains the current low-level
  coagulation path. The kernel uses `wp.tid()` as the box index and assigns one
  thread to each box, keeping pair selection sequential to avoid cross-thread
  write races.
- The public `coagulation_step_gpu(...)` wrapper already validates environment,
  volume, device, and optional buffer inputs before launching Warp kernels.
- E3-F1 and E3-F2 should be treated as upstream context for RNG persistence and
  mixed-scale sampling behavior before final benchmark interpretation.

## Benchmark Infrastructure

- `particula/gpu/tests/benchmark_test.py` already defines a coagulation matrix
  covering single-box sizes from 500 to 50k particles and multi-box sizes such
  as `10x500`, `50x5k`, and `100x1k`.
- The benchmark suite is opt-in behind `pytest --benchmark` and skips CUDA-only
  paths when CUDA is unavailable.
- `particula/gpu/tests/benchmark_helpers_test.py` provides fast regression
  coverage for benchmark metadata and helper behavior.

## Documentation Infrastructure

- `docs/Features/Roadmap/data-oriented-gpu.md` is the Epic C source of truth and
  already calls out the one-thread-per-box design limit.
- `docs/Theory/nvidia-warp/examples/gpu_benchmarks.py` is the paired source for
  benchmark interpretation content and should be edited before syncing the
  notebook when notebook output changes are required.
- `docs/Features/data-containers-and-gpu-foundations.md` may host concise user
  guidance if the decision affects low-level GPU API usage boundaries.
