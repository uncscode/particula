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

## Delivered in P3 (Issue #1583)

- Host-only canonical matrix rows for 1, 10, 100, and 1000 boxes in
  `particula/execution/tests/resident_benchmark_support.py`, with explicit
  case axes and exact requested/actual shapes; no row is capacity-downscaled.
- Conservative requested-case-estimate preflight before device probing,
  fixture construction, allocation, or timing. Budget equality is allowed;
  over-budget rows are structured `skipped_budget` outcomes and eligible
  unavailable CUDA/device/capture rows are structured `unavailable` outcomes.
- Exact `n_boxes`, `n_particles`, and `n_species` forwarding through the P2
  CUDA fixture seam and an opt-in aggregate resident-artifact consumer.
- Host and injected-double regression coverage for matrix metadata, budget and
  availability boundaries, exact dimensions, availability memoization, cleanup,
  one aggregate writer call, and no CPU/Warp-CPU fallback.

## Delivered in P4 (Issue #1584)

- Standard-library-only checked byte arithmetic and frozen resident-memory
  category/model records in
  `particula/execution/tests/resident_benchmark_support.py`.
- Exact analytical primary-field categories; one caller-provided E8-F3 registry
  logical-byte category; selected diagnostic outputs; and visible, non-additive
  communication selection and inactive-capacity attribution.
- Separate excluded checkpoint primary/sidecar/inspection-copy scenarios and
  checked full-retention/checkpointed tape projections. Unknown Epic I overhead
  is explicitly excluded rather than estimated.
- Comprehensive default-collection host-only formula, boundary, invariant,
  reconciliation, immutability, and import-isolation tests. No production or
  public documentation files changed.

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
- P5 allocator high-water probing and
  analytical-versus-observed memory comparisons; published documentation; and
  any speed threshold or general performance conclusion.
