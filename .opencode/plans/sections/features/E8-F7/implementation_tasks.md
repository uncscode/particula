# Implementation Tasks

## Profiling and Analysis Support

- [ ] Add versioned profiling records and strict validators in
  `particula/gpu/tests/profiling_support.py`.
- [ ] Define deterministic small and medium workloads with fixed box, particle,
  species, communication, replay-count, warmup, and sample parameters.
- [ ] Reuse the benchmark artifact root and safe filename rules; add checksums
  and relative references for raw profiler exports.
- [ ] Add separate host enqueue/launch and synchronized elapsed samplers without
  placing setup, capture construction, fixture reset, or serialization in the
  measured region.
- [ ] Route both replay modes through the same prepared state contract and emit
  explicit unavailable rows for missing CUDA/capture support.
- [ ] Add parsers for selected machine-readable Nsight Systems/Compute exports;
  fail closed on unsupported schema or units and retain raw source references.
- [ ] Map observed kernel names to canonical resident process IDs where the
  mapping is evidenced; preserve an `unattributed` category otherwise.
- [ ] Aggregate duration and invocation counts and retain occupancy, achieved
  bandwidth, memory-transaction, and stall metrics only when the profiler
  reports them with documented units.
- [ ] Rank host-launch and device-kernel contributions deterministically and
  produce recommendations containing workload, machine, metric, evidence, and
  explicit non-portability language.
- [ ] Refuse recommendations that propose scientific-contract, tolerance,
  ownership, process-order, or RNG changes without a separate correctness plan.

## Tooling / Tests

- [ ] Add `particula/gpu/tests/profiling_support_test.py` for schema, parser,
  aggregation, ranking, path safety, and recommendation guardrails.
- [ ] Extend `particula/gpu/tests/benchmark_helpers_test.py` with timer-spy tests
  proving warmup/setup exclusion and synchronization boundaries.
- [ ] Add CUDA-gated opt-in rows to `benchmark_test.py`; retain `benchmark`,
  `slow`, `performance`, `warp`, and `cuda` intent markers as applicable.
- [ ] Store compact text/JSON parser fixtures in a bounded test-data directory;
  do not require NVIDIA tooling for default unit tests.
- [ ] Record exact profiler commands, selected metric sets, exit codes, and
  literal result summaries; unavailable required rows remain unshipped.

## Documentation

- [ ] Publish the measured machine/software table, workload matrix, raw sample
  references, synchronization method, and profiler overhead caveat.
- [ ] Publish bottleneck and recommendation tables with evidence links and
  explicit machine/workload bounds.
- [ ] Reconcile the T7/E8-F7 assignment with stale parent references that label
  profiling as E8-F8 before epic closeout.
