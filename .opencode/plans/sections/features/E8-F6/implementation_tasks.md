# Implementation Tasks

## Benchmark and Evidence Support

- [ ] Create `particula/execution/tests/resident_benchmark_support.py` with
  frozen `ResidentBenchmarkCase` and `ResidentBenchmarkResult` records; validate
  positive dimensions, bounded samples, process combinations, and case IDs.
- [ ] Implement `build_resident_benchmark_metadata()` in that support module to
  record Python/platform/Warp versions, exact device identity and memory, UTC
  timestamp, command, synchronization method, warmup, timestep count, seed, and
  prepared-signature digest.
- [ ] Implement `time_prepared_resident_modes()` to accept one E8-F2/E8-F5
  fixture, pin its E8-F3/E8-F4 identities before warmup, and return distinct raw
  uncaptured, replay, setup, and capture sample sequences.
- [ ] Implement `summarize_raw_samples()` with deterministic count/min/median/
  mean/p95 fields; reject nonfinite or empty timed sample sequences.
- [ ] Define `RESIDENT_BOX_COUNTS = (1, 10, 100, 1000)` and construct cases with
  explicit particle/species, active-fraction, communication, diagnostics, and
  process-set fields rather than anonymous tuple configurations.
- [ ] Implement `preflight_resident_benchmark_case()` before fixture/device
  allocation; return structured `executed`, `skipped_budget`, or `unavailable`
  rows with the requested and actual capacities plus a reason.
- [ ] Extend `_save_results()` in `particula/gpu/tests/benchmark_test.py` or add a
  dedicated atomic writer in `resident_benchmark_support.py` so completed rows
  survive interruption without accepting artifact paths outside `.artifacts/`.

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
- [ ] Implement `probe_device_memory()` behind the optional CUDA benchmark path;
  record probe method, availability, and before/peak/after values without
  treating allocator reservation as logical bytes.
- [ ] Implement `compare_memory_evidence()` to retain the analytical total,
  observed delta, and unexplained difference rather than forcing equality.

## Tooling / Tests

- [ ] Add `particula/execution/tests/resident_benchmark_support_test.py` for
  schema, byte sizing, summaries, fixture routing, serialization, and artifact
  path safety; keep existing generic benchmark coverage in
  `particula/gpu/tests/benchmark_helpers_test.py`.
- [ ] Add opt-in resident CUDA benchmark tests marked `slow`, `performance`,
  `benchmark`, `warp`, and `cuda`, with clean skip and no CPU substitution.
- [ ] Keep default test collection unchanged and add regression coverage for
  malformed dimensions, overflow, duplicate categories, unavailable probes,
  and artifact path safety.
- [ ] Update documentation contract tests for commands, axes, categories,
  limitations, and artifact provenance.
