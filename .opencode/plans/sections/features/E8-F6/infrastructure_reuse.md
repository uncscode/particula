# Infrastructure Reuse

- `particula/gpu/tests/benchmark_test.py:46-55` provides the early explicit
  `--benchmark` gate; retain it so default collection never imports Warp for
  these heavy rows.
- `particula/gpu/tests/benchmark_test.py:133-195` provides established warmup,
  step counts, box/particle configuration tables, and the box-first convention.
- `BenchmarkMemoryBudget` and sizing helpers at
  `particula/gpu/tests/benchmark_test.py:198-209` and `312-435` provide safe
  dense-array byte arithmetic and pre-allocation skip behavior to generalize.
- `_build_benchmark_metadata()` and `_save_results()` at
  `particula/gpu/tests/benchmark_test.py:268-292` and `450-466` already record
  command, Warp/device context, timestamps, and durable partial JSON output.
- `particula/gpu/tests/benchmark_helpers_test.py` isolates benchmark helpers for
  fast default-suite tests without running CUDA timings; extend this pattern.
- `particula/conftest.py:19-62` and `particula/_pytest_support.py` own benchmark
  option propagation and marker gating; do not introduce another CLI flag.
- E8-F3's planned `CaptureResourceSet` and `CaptureResourceByteReport` are the
  authority for reusable logical bytes; consume their canonical records rather
  than duplicate process-sidecar formulas.
- E8-F4's captured replay boundary and E8-F5's three-way full-loop fixtures are
  the execution/correctness seams. Benchmark setup must reuse them rather than
  constructing a second scheduler or weakening validation.
- Existing CUDA availability helpers in
  `particula/gpu/tests/cuda_availability.py` supply canonical clean-skip logic.
- `.artifacts/benchmarks/gpu_benchmark_results.json` and the evidence format
  documented at `docs/Features/Roadmap/data-oriented-gpu.md:1790-1809` provide
  the artifact location and source-of-record precedent.
- The parent memory categories and benchmark axes are defined at
  `docs/Features/Roadmap/data-oriented-gpu.md:1776-1788`; preserve those names
  so downstream Epic I comparisons remain traceable.
