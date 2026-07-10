# Documentation Updates

## Required Updates

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - The shipped benchmark evidence now reads as a measured decision record,
    preserving the exact pytest command, benchmark date, CUDA/Warp/device
    context, controlled artifact path, and representative timing values.
  - The interpretation now explicitly separates the recorded large single-box
    caution band (`1x10k` to `1x50k`) from the measured many-box effective
    region (`10x500`, `10x1k`, `50x1k`, `10x5k`, `50x5k`, `100x1k`, `10x10k`).
  - The wording stays machine-bounded to the recorded 2026-07-10 capture for
    the current one-thread-per-box design rather than presenting a universal GPU
    cutoff.

- `docs/Theory/nvidia-warp/examples/gpu_benchmarks.py`
  - Aligned the notebook-backed source text to the controlled artifact path
    `.artifacts/benchmarks/gpu_benchmark_results.json`.
  - Updated the theory/example observations and recommendation wording to match
    the roadmap's measured single-box caution band, many-box effective region,
    and machine-bounded caveat.

## Optional Updates

- Store raw benchmark output under `.artifacts/benchmarks/` when produced; the
  shipped roadmap note should summarize only the decision-relevant evidence.
- Add a follow-up issue/plan reference if parallel-within-box work is scoped.

## Documentation Quality Requirements

- State CUDA optionality clearly.
- Avoid presenting machine-specific benchmark numbers as universal guarantees.
- Avoid implying hidden CPU/GPU synchronization or automatic transfers.
- Keep future optimization work outside this feature's scope.
