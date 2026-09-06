# Implementation Tasks

## Profiling and Analysis Support

- [x] Add versioned profiling records and strict validators in
  `particula/gpu/tests/profiling_support.py`.
- [x] Define deterministic small and medium workloads with fixed box, particle,
  species, communication, replay-count, warmup, and sample parameters.
- [x] Freeze the configured E8-F6 matrix as small `(1, 16, 2)` and medium
  `(1000, 16, 2)` with canonical IDs. These are not asserted to be executed or
  feasible; later collection must use them unchanged or record unavailable.
- [x] Reuse the benchmark artifact-root and safe filename rules; add checksums
  and relative references for raw profiler exports.
- [x] Create `.artifacts/benchmarks/profiling/raw/` on demand as the only local
  raw-report staging directory. Enforce canonical containment, reject symlink or
  traversal escape, and keep its narrow `.gitignore` rule from hiding normalized
  evidence elsewhere in `.artifacts/benchmarks`.
- [ ] Commit only bounded normalized summaries and parser fixtures. Include raw
  filename, byte size, and SHA-256 provenance, but provide no upload, attachment,
  release-asset, Git LFS, or ordinary-Git raw-report path.
- [x] Add separate host enqueue/launch and synchronized elapsed samplers without
  placing setup, capture construction, fixture reset, or serialization in the
  measured region.
- [x] Route both replay modes through the same prepared state contract and emit
  explicit unavailable rows for missing CUDA/capture support.
- [x] Publish four P1-valid mode/method artifacts and `manifest.json` atomically;
  restore the prior complete publication when staging or replacement fails.
- [x] Add a private identity-preserving CUDA fixture reset that snapshots and
  restores mutable primary and acquired continuation arrays, including RNG
  sidecars, between warmups and independent batches.
- [ ] Add parsers for selected machine-readable Nsight Systems/Compute exports;
  fail closed on unsupported schema or units and retain raw source references.
- [ ] Qualify the Arch Linux `nsight-systems 2026.1.3.425-1` and
  `nsight-compute 2026.2.1.5-1` package pair, persist literal CLI version
  output, and keep both profilers as optional external tools rather than Python
  dependencies.
- [ ] Add bounded Python subprocess orchestration for version probes, collection,
  and export. Use explicit argument vectors with `shell=False`, capture exit
  status and bounded diagnostics, and write only beneath the controlled artifact
  root; do not launch external tools from the default test suite.
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

- [x] Add `particula/gpu/tests/profiling_support_test.py` for schema, canonical
  JSON, workload, evidence-union, and raw-provenance path-safety coverage.
- [x] Extend `particula/gpu/tests/benchmark_helpers_test.py` with timer-spy tests
  proving warmup/setup exclusion and synchronization boundaries.
- [x] Add CUDA-gated opt-in rows to `benchmark_test.py`; retain `benchmark`,
  `slow`, `performance`, `warp`, and `cuda` intent markers as applicable.
- [ ] Store compact text/JSON parser fixtures in a bounded test-data directory;
  do not require NVIDIA tooling for default unit tests.
- [ ] Unit-test subprocess argument construction, version rejection, timeout,
  nonzero exit, truncated diagnostics, path safety, and fixture parsing without
  launching `nsys`, `ncu`, or a GPU workload.
- [ ] Add `particula/gpu/tests/profiling_smoke_test.py` as an explicit
  `--benchmark` CUDA smoke test. It must invoke the installed `nsys` and `ncu`
  binaries, verify the selected version identities, profile one bounded CUDA
  workload, export both supported machine-readable formats, and parse them with
  the production profiling parser.
- [ ] Keep smoke artifacts beneath `.artifacts/benchmarks`, retain command and
  exit-status provenance, and report missing binaries, unsupported versions,
  denied counters, export failures, or missing metrics without changing host
  permissions or using CPU fallback.
- [ ] Record exact profiler commands, selected metric sets, exit codes, and
  literal result summaries; unavailable required rows remain unshipped.
- [ ] Require a qualified NVIDIA CUDA GPU for real profiler rows; missing CUDA,
  profiler access, or hardware counters must not route to Warp CPU.

## Documentation

- [ ] Publish the measured machine/software table, workload matrix, raw sample
  references, synchronization method, and profiler overhead caveat.
- [ ] Publish bottleneck and recommendation tables with evidence links and
  explicit machine/workload bounds.
- [ ] Reconcile the T7/E8-F7 assignment with stale parent references that label
  profiling as E8-F8 before epic closeout.
