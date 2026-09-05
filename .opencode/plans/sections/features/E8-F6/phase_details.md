# Phase Details

- [x] **E8-F6-P1:** Define resident benchmark matrix and reproducible artifact schema with unit tests
  - Issue: #1581 | Size: S | Status: Shipped
  - Delivered: Frozen host-only case, timing-summary, result, and artifact
    records validate canonical configurations and references before CUDA-facing
    work. Complete caller-provided provenance, deterministic schema-versioned
    serialization/deserialization, and a verified-`.artifacts` atomic generic
    JSON writer are implemented without Warp/CUDA imports or probes.
  - Files: `particula/execution/tests/resident_benchmark_support.py` and
    `particula/execution/tests/resident_benchmark_support_test.py`.
  - Tests: Default-collection host-only coverage for construction, status/sample
    rules, metadata, canonical IDs, deterministic round trips, malformed rows,
    import isolation, path/symlink rejection, and atomic-write failures.

- [x] **E8-F6-P2:** Benchmark captured versus uncaptured repeated resident timesteps with unit tests
  - Issue: #1582 | Size: S | Status: Shipped
  - Delivered: Schema-v2 paired timing support alternates uncaptured prepared
    enqueue and captured replay on continuing state. Measured calls synchronize
    once each; setup/capture elapsed time is immutable provenance, not a sample.
    A lazy CUDA-only qualified binding and one opt-in benchmark row emit exactly
    two executed comparison results to the isolated fixed artifact.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/execution/tests/resident_benchmark_cuda_support.py`,
    `particula/execution/tests/resident_benchmark_cuda_support_test.py`, and
    `particula/gpu/tests/benchmark_test.py`.
  - Tests: Host and fake-CUDA tests validate bounds, callback order,
    synchronization placement, schema compatibility, fixed artifact isolation,
    lazy import, identity rejection, and setup/capture boundaries. The real
    CUDA-only row is opt-in and passes or cleanly skips when native capture is
    unavailable; it is supplemental evidence only.

- [ ] **E8-F6-P3:** Extend box-first scaling matrix and budget-aware unavailable rows with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Define 1/10/100/1000-box cases and a <=100-LOC preflight that emits
    `executed`, `skipped_budget`, or `unavailable` rows before oversized device
    allocation; retain the requested case shape in every row.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/gpu/tests/benchmark_test.py`
  - Tests: Matrix axes, budget boundaries, requested-versus-reduced capacity,
    CUDA clean skip, and no hidden CPU fallback.

- [ ] **E8-F6-P4:** Build analytical resident memory-budget model with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add <=100-LOC checked byte-category aggregation that imports the E8-F3
    report once and separately accounts for primary, inactive capacity,
    diagnostics, communication, checkpoint host copies, and tape projections.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/execution/gpu_resources.py`
  - Tests: Closed-form fixtures, checked arithmetic, category reconciliation,
    duplicate-category rejection, and zero dimensions.

- [ ] **E8-F6-P5:** Compare analytical and observed peak device memory with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add one <=100-LOC optional CUDA probe adapter that records comparable
    before/peak/after readings and the unexplained analytical-to-observed delta
    without relabeling allocator reservation as logical bytes.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/gpu/tests/benchmark_test.py`
  - Tests: Probe availability, nonnegative readings, unavailable-evidence rows,
    and representative fixture integration.

- [ ] **E8-F6-P6:** Publish benchmark and memory-budget evidence with documentation validation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish exact commands, executed/unavailable rows, limitations, and reviewed artifacts.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, feature report, artifact metadata
  - Tests: documentation contract assertions and `mkdocs build --strict`
