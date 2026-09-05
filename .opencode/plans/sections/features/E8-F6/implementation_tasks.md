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
- [ ] Define `RESIDENT_BOX_COUNTS = (1, 10, 100, 1000)` and construct cases with
  explicit particle/species, active-fraction, communication, diagnostics, and
  process-set fields rather than anonymous tuple configurations.
- [ ] Implement `preflight_resident_benchmark_case()` before fixture/device
  allocation; return structured `executed`, `skipped_budget`, or `unavailable`
  rows with the requested and actual capacities plus a reason.
- [x] Add the dedicated `write_json_artifact()` atomic writer in
  `resident_benchmark_support.py`; it accepts normalized generic JSON only below
  a verified `.artifacts` root and rejects containment/symlink escapes.
- [x] Add `write_resident_capture_comparison_artifact()` for the fixed isolated
  schema-v2 resident artifact without invoking generic benchmark output.

## Memory Model

- [ ] Implement `checked_dense_array_bytes()` and `aggregate_memory_categories()`
  in `resident_benchmark_support.py`; reject negative dimensions and arithmetic
  overflow without relying on NumPy fixed-width multiplication.
- [ ] Implement `build_resident_memory_model()` from exact primary-array
  shapes/dtypes, including fixed inactive capacity rather than active population.
- [ ] Accept one E8-F3 `ResourceByteReport` in
  `build_resident_memory_model()` and include its total exactly once.
- [ ] Model selected diagnostics, GAS/PARTICLES communication, checkpoint host
  copies, and projected tape storage as named optional categories with distinct
  logical-versus-observed labels.
- [ ] Implement symbolic tape scenarios using explicit `S`, `C`, `T`, and `K`
  inputs: full retention `T * S` and checkpointed working storage
  `ceil(T / K) * C + K * S`. Label both projected and exclude unknown Epic I
  operation/intermediate overhead.
- [ ] Implement `probe_device_memory()` behind the optional CUDA benchmark path;
  accept only a documented, version-qualified public allocator high-water API,
  record its coverage, method, versions, availability, and before/peak/after
  values, and never treat allocator reservation as logical bytes.
- [ ] Add a local opt-in allocator-probe smoke test. Unsupported Warp/CUDA
  versions, incomplete graph/non-Warp coverage, or inaccessible counters must
  produce an unavailable observed-memory record without NVML substitution.
- [ ] Implement `compare_memory_evidence()` to retain the analytical total,
  observed delta, and unexplained difference rather than forcing equality.

## Tooling / Tests

- [x] Add `particula/execution/tests/resident_benchmark_support_test.py` for
  schema, summaries, serialization, and artifact path safety; byte sizing and
  fixture routing remain future P2/P3 work. Keep existing generic coverage in
  `particula/gpu/tests/benchmark_helpers_test.py`.
- [x] Add one opt-in resident CUDA benchmark marked `slow`, `performance`,
  `benchmark`, `warp`, and `cuda`, with clean native-CUDA-capture skip and no
  CPU/Warp-CPU substitution.
- [ ] Keep default test collection unchanged and add regression coverage for
  malformed dimensions, overflow, duplicate categories, unavailable probes,
  and artifact path safety.
- [ ] Update documentation contract tests for commands, axes, categories,
  limitations, and artifact provenance.
