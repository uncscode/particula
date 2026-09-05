# Scope

E8-F6 delivers supplemental, opt-in CUDA evidence for captured-versus-
uncaptured resident scaling and publishes a dimension-driven memory model tied
to the exact E8 resource inventory.

## In Scope

- A resident benchmark case schema varying boxes first, then particles per box,
  species, active-slot fraction, process set, communication, and diagnostics.
- Required box-count rows of 1, 10, 100, and 1000 when hardware permits, plus
  small and medium repeated-step captured/uncaptured comparisons.
- Warmup, explicit synchronization, raw samples, summary statistics, device and
  software metadata, command, seed/configuration, and artifact schema version.
- Budget-aware preflight and structured unavailable/clean-skip rows for missing
  CUDA capture support or insufficient configured/device memory.
- Analytical categories for primary resident state, inactive slot capacity,
  E8-F3 reusable resources, diagnostics, communication, checkpoints, and
  explicitly projected autodiff tape/checkpointing scenarios.
- Representative analytical-versus-observed peak-device-memory comparisons.
- Fast helper/contract tests in default collection and opt-in benchmark rows
  behind the existing `--benchmark` gate.
- Reproducible JSON evidence and a user-facing roadmap/feature report.

## Delivered in P1 (Issue #1581)

- Concrete host-only schema, provenance, deterministic JSON round-trip, and
  path-safe atomic generic JSON writer in
  `particula/execution/tests/resident_benchmark_support.py`.
- Default-collection host-only contract tests in
  `particula/execution/tests/resident_benchmark_support_test.py`.
- No benchmark dispatch, CUDA/Warp import or probe, production execution change,
  package export, public documentation, or artifact publication.

## Delivered in P2 (Issue #1582)

- Schema-v2 capture-comparison records in
  `particula/execution/tests/resident_benchmark_support.py`: two dedicated
  device-synchronized modes, bounded paired timing collection, backward v1
  decode, and provenance-only setup/capture durations.
- A fixed, isolated atomic artifact destination:
  `.artifacts/benchmarks/resident_capture_comparison.json`. Generic benchmark
  state and its writer remain untouched.
- Lazy CUDA-only fixture/capture support in
  `particula/execution/tests/resident_benchmark_cuda_support.py`, with
  hardware-free support tests and exact binding identity validation before
  collection.
- One `--benchmark`, `slow`, `performance`, `warp`, and `cuda` resident
  comparison row in `particula/gpu/tests/benchmark_test.py`. It records equal,
  nonempty paired sample counts only after CUDA/native-capture qualification;
  unavailable CUDA/native capture skips rather than falling back.

## Out of Scope

- Changing scientific kernels, process ordering, graph semantics, or memory
  ownership merely to improve a benchmark result.
- Universal speedup promises, hard performance CI gates, or committed results
  inferred on machines where a row did not execute.
- Multi-GPU/distributed execution, dynamic capacity growth, compaction, hidden
  CPU fallback, or allocator-specific guarantees across Warp/CUDA versions.
- Implementing autodiff or allocating a real tape; tape values are labeled
  projections for Epic I until measured evidence exists.
- CUDA occupancy/kernel profiling, which belongs to E8-F7, and the runnable
  lifecycle example, runbook, and final Epic H closeout, which belong to E8-F8.
- A box-count matrix, memory-budget model/probe, published documentation, and
  any speed threshold or general performance conclusion.
