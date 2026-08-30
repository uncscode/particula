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

## Out of Scope

- Changing scientific kernels, process ordering, graph semantics, or memory
  ownership merely to improve a benchmark result.
- Universal speedup promises, hard performance CI gates, or committed results
  inferred on machines where a row did not execute.
- Multi-GPU/distributed execution, dynamic capacity growth, compaction, hidden
  CPU fallback, or allocator-specific guarantees across Warp/CUDA versions.
- Implementing autodiff or allocating a real tape; tape values are labeled
  projections for Epic I until measured evidence exists.
- CUDA occupancy/kernel profiling and final Epic H closeout, which belong to
  E8-F8, and the runnable lifecycle example, which belongs to E8-F7.
