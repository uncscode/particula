# Implementation Tasks

## Benchmark and Evidence Support

- [x] Create `particula/execution/tests/resident_benchmark_support.py` with
  frozen `ResidentBenchmarkCase` and `ResidentBenchmarkResult` records; validate
  positive dimensions, bounded samples, process combinations, and case IDs.
- [x] Implement `build_resident_benchmark_metadata()` in that support module to
  record Python/platform/Warp versions, exact device identity and memory, UTC
  timestamp, command, synchronization method, warmup, timestep count, seed, and
  prepared-signature digest.
- [x] Implement bounded paired capture-comparison timing in
  `collect_paired_device_timings()` and the lazy CUDA binding support. It
  alternates uncaptured enqueue and captured replay on one qualified binding,
  retains separate raw tuples/summaries, and records setup/capture time only as
  provenance.
- [x] Implement `summarize_timing_samples()` with deterministic count/min/median/
  mean/nearest-rank-p95 fields; reject nonfinite, negative, empty, or over-cap
  executed timing sample sequences.
- [x] Define `RESIDENT_BOX_COUNTS = (1, 10, 100, 1000)` and construct cases with
  explicit particle/species, active-fraction, communication, diagnostics, and
  process-set fields rather than anonymous tuple configurations.
- [x] Implement `preflight_resident_benchmark_case()` before fixture/device
  allocation; return structured `executed`, `skipped_budget`, or `unavailable`
  rows with the requested and actual capacities plus a reason.
- [x] Add the dedicated `write_json_artifact()` atomic writer in
  `resident_benchmark_support.py`; it accepts normalized generic JSON only below
  a verified `.artifacts` root and rejects containment/symlink escapes.
- [x] Add `write_resident_capture_comparison_artifact()` for the fixed isolated
  schema-v2 resident artifact without invoking generic benchmark output.

## Memory Model

- [x] Implement checked `checked_dense_array_bytes()` and category aggregation
  in `resident_benchmark_support.py`; reject invalid/overflowing Python-integer
  inputs without NumPy fixed-width arithmetic.
- [x] Implement `build_resident_memory_model()` from exact primary-array
  shapes/dtypes, with visible non-additive fixed inactive-capacity attribution.
- [x] Accept the E8-F3 logical-byte aggregate as one validated integer category
  and include it exactly once without importing `gpu_resources`.
- [x] Model selected diagnostics, communication selection, checkpoint host-copy
  scenarios, and projected tape storage as named categories with distinct
  steady-state/checkpoint/tape scenarios and provenance.
- [x] Implement symbolic full-retention `T * S` and checkpointed
  `ceil(T / K) * C + K * S` tape projections with checked arithmetic; label
  them projected and exclude unknown Epic I operation/intermediate overhead.
- [x] Implement the lazy `CudaDefaultPoolHighWater` CUDA Runtime adapter and
  private fixture monitor behind the optional CUDA benchmark path. They use only
  documented default-pool `cudaMemPoolAttrUsedMemHigh`, retain method/version/
  coverage context and synchronized before/peak/after values, and never treat
  allocator reservation as logical bytes.
- [x] Add hardware-free adapter/monitor and opt-in CUDA-row coverage. Unsupported
  Runtime/API, incomplete sentinel coverage, inaccessible counters, and failed
  snapshots produce all-null unavailable observations without NVML substitution
  or timing-row failure.
- [x] Implement P4 comparison attachment: retain analytical steady-state bytes,
  observed delta, and signed difference without equality enforcement, once per
  executed fixture after cleanup.

## Tooling / Tests

- [x] Add `particula/execution/tests/resident_benchmark_support_test.py` for
  schema, summaries, serialization, and artifact path safety; byte sizing and
  fixture routing remain future P2/P3 work. Keep existing generic coverage in
  `particula/gpu/tests/benchmark_helpers_test.py`.
- [x] Add one opt-in resident CUDA benchmark marked `slow`, `performance`,
  `benchmark`, `warp`, and `cuda`, with clean native-CUDA-capture skip and no
   CPU/Warp-CPU substitution.
- [x] Replace the resident artifact's single-case path with the P3 matrix
  consumer: preflight every exact-shape row, memoize eligible availability,
  reuse one binding per approved row, and publish one aggregate artifact only
  after all rows complete.
- [x] Forward validated exact `n_boxes`, `n_particles`, and `n_species` through
  the P2 fixture/request seam without changing its default 16-by-2 callers.
- [x] Keep default test collection unchanged and add regression coverage for
  malformed dimensions, overflow, duplicate categories, memory-category
  reconciliation, tape projections, and import isolation.
- [x] Add `particula/tests/resident_benchmark_docs_test.py` to verify the
  roadmap/report link, exact source artifact path, fixed configuration,
  unavailable/not-measured evidence state, accounting vocabulary, and
  documentation-only limitations without reading an artifact or importing Warp.
- [x] Publish `docs/Features/resident_benchmark_memory_budget.md` and its single
  roadmap link; preserve the explicit absence of reviewed resident evidence and
  do not publish fabricated timing, allocator, or provenance values.
