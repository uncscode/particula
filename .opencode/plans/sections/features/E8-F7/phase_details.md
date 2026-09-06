# Phase Details

- [x] **E8-F7-P1:** Define profiling artifact schema and representative workload matrix with unit tests
  - Issue: #1589 | Size: S | Status: Shipped
  - Delivered: Host-only versioned workload, machine, method, raw-sample,
    metric, and executed/unavailable evidence records for the exact small
    `(1, 16, 2)` and medium `(1000, 16, 2)` matrix. Bounded canonical JSON and
    safe injected `.artifacts` raw-report provenance are included; no timing or
    profiler evidence was collected.
  - Files: `particula/gpu/tests/profiling_support.py`,
    `particula/gpu/tests/profiling_support_test.py`
  - Tests: Hardware-free exact schema validation, canonical serialization, safe
    paths and changed-report detection, deterministic workload IDs, malformed
    metrics, and unavailable evidence. No Warp/CUDA/profiler process or public
    export was added.

- [x] **E8-F7-P2:** Measure captured and uncaptured host launch costs with unit tests
  - Issue: #1590 | Size: S | Status: Shipped
  - Delivered: One qualified native-CUDA binding per frozen workload supplies
    prepared-uncaptured and captured-replay rows for host-launch and explicitly
    synchronized elapsed methods. Four P1-valid artifacts and an explicit
    manifest are atomically published below `.artifacts/benchmarks/profiling/`;
    executed rows retain raw JSON sample provenance, while unavailable
    prerequisites publish all four complete workload-ordered unavailable rows.
  - Reset boundary: Benchmark-private snapshots restore existing mutable
    resident primary arrays and acquired continuation/RNG sidecars before every
    warmup and independent batch, then validate identity. No reset rebuilds,
    checkpoints, restarts, or recaptures the qualified fixture.
  - Files: `particula/gpu/tests/benchmark_test.py`,
    `particula/gpu/tests/benchmark_helpers_test.py`,
    `particula/execution/tests/resident_benchmark_cuda_support.py`,
    `particula/execution/tests/resident_benchmark_cuda_support_test.py`
  - Tests: Hardware-free spies cover two-clock host timing with no in-interval
    synchronization, one synchronized-elapsed completion boundary, replay-major
    raw samples and positive deltas, unavailable no-call routing, and atomic
    publication rollback. CUDA-support tests cover primary/RNG restoration,
    empty registries, validation ordering, and error propagation.

- [x] **E8-F7-P3:** Collect CUDA kernel profiling evidence with tests
  - Issue: #1591 | Size: S | Status: Shipped
  - Delivered: Strict versioned Nsight evidence records/parsers, bounded
    `shell=False` orchestration, typed unavailable/failed outcomes, a closed
    fixed-argv native-CUDA worker, and opt-in CUDA smoke composition.
  - Files: `particula/gpu/tests/profiling_support.py`,
    `particula/gpu/tests/profiling_support_test.py`,
    `particula/gpu/tests/profiling_workload_runner.py`,
    `particula/gpu/tests/profiling_workload_runner_test.py`, and
    `particula/gpu/tests/profiling_smoke_test.py`.
  - Tests: Hardware-free fixtures and mock runners cover parsing, conversion,
    mapping/aggregation, argv, timeouts, diagnostics, paths, and failures;
    worker tests cover pre-import validation, call order, unavailable status, and
    cleanup. The CUDA smoke is opt-in and skips unavailable prerequisites with no
    CPU/Warp-CPU fallback. No production scheduler/API behavior, user
    documentation, or measured result was added.

- [ ] **E8-F7-P4:** Analyze bottlenecks and generate machine-bounded recommendations with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Rank measured launch and kernel contributions and generate an
    evidence-linked decision table with explicit scope and confidence limits.
  - Files: `particula/gpu/tests/profiling_support.py`,
    `particula/gpu/tests/profiling_support_test.py`
  - Tests: Deterministic ranking, ties, unavailable inputs, percentage
    reconciliation, recommendation guardrails, and no portable-claim language.

- [ ] **E8-F7-P5:** Update development documentation and publish reproducible profiling results
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish exact commands, raw-artifact references, machine metadata,
    findings, limitations, and bounded follow-up recommendations.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/gpu_graph_capture_performance.md`, `AGENTS.md`
  - Tests: Documentation contract assertions and `mkdocs build --strict`.
