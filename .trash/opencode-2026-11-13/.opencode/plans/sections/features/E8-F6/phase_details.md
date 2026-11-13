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

- [x] **E8-F6-P3:** Extend box-first scaling matrix and budget-aware unavailable rows with unit tests
  - Issue: #1583 | Size: S | Status: Shipped
  - Delivered: Host-only 1/10/100/1000-box cases with explicit axes and exact
    requested/actual shapes; conservative preflight permits budget equality and
    emits only `executed`, `skipped_budget`, or `unavailable` evidence before
    CUDA probing/allocation. Exact dimensions flow through the P2 fixture seam;
    the opt-in consumer aggregates all rows without capacity downscaling or
    CPU/Warp-CPU fallback. P4/P5 allocator and byte accounting were not added.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/execution/tests/resident_benchmark_cuda_support.py`,
    `particula/execution/tests/multi_box_loop_test.py`,
    `particula/execution/tests/captured_full_loop_test.py`,
    `particula/gpu/tests/benchmark_test.py`, and
    `particula/gpu/tests/benchmark_safety_test.py`.
  - Tests: Matrix/axis and no-downscale assertions; invalid/equal/over-budget
    boundaries; unavailable and malformed-availability outcomes; exact
    forwarding; availability memoization; binding reuse/cleanup; aggregate
    writer behavior; and no fallback.

- [x] **E8-F6-P4:** Build analytical resident memory-budget model with unit tests
  - Issue: #1584 | Size: S | Status: Shipped
  - Delivered: Checked, ceiling-bounded Python-integer arithmetic; immutable
    ordered categories/models; exact primary, registry, diagnostic,
    communication-selection, inactive-capacity, and checkpoint accounting; and
    full-retention/checkpointed projected tape formulas. The E8-F3 report enters
    once as a validated integer; unknown Epic I overhead is explicitly excluded.
  - Files: `particula/execution/tests/resident_benchmark_support.py` and
    `particula/execution/tests/resident_benchmark_support_test.py`.
  - Tests: Comprehensive host-only closed-form and zero fixtures; input/type,
    overflow, uniqueness, reconciliation, immutability, tape, and subprocess
    import-isolation coverage. No Warp, NumPy, or production resource module is
    imported.

- [x] **E8-F6-P5:** Compare analytical and observed peak device memory with integration tests
  - Issue: #1585 | Size: S | Status: Shipped
  - Delivered: Schema-v3 case-scoped artifact observations compare P4 logical
    steady-state bytes with the exact-device default-pool used-high allocation
    delta and retain the signed difference only. A lazy `ctypes` CUDA Runtime
    adapter uses only `cudaMemPoolAttrUsedMemHigh`; a private per-fixture monitor
    proves coverage with a reset sentinel and records synchronized before,
    post-capture peak, and post-cleanup after snapshots. Monitor faults retain
    deterministic all-null unavailable evidence without altering timing rows.
    The collector derives P4 inputs from the live fixture and appends exactly
    one finalized observation after cleanup for each executed case.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/execution/tests/resident_benchmark_cuda_support.py`,
    `particula/execution/tests/resident_benchmark_cuda_support_test.py`,
    `particula/gpu/tests/benchmark_test.py`, and
    `particula/gpu/tests/benchmark_safety_test.py`.
  - Tests: Hardware-free schema/adapter/monitor contracts cover import
    isolation, Runtime/API and snapshot failures, sentinel coverage, lifecycle
    ordering, and unavailable records. Injected collector tests cover live P4
    inputs, one observation per fixture, and no timing-loop monitor work; the
    optional CUDA row accepts valid evidence or structured unavailability.

- [x] **E8-F6-P6:** Publish benchmark and memory-budget evidence with documentation validation
  - Issue: #1586 | Size: XS | Status: Shipped
  - Delivered: Published the roadmap-linked resident benchmark/memory-budget
    record with the exact collection command, fixed matrix, planning inputs,
    accounting vocabulary, tape projections, and bounded non-claims. The only
    resident source is the schema-v3 artifact path; because it is absent, all
    current rows are explicitly unavailable and not measured.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/resident_benchmark_memory_budget.md`, and
    `particula/tests/resident_benchmark_docs_test.py`.
  - Tests: Hardware-free stdlib-only documentation assertions cover the roadmap
    link, source provenance, fixed configuration, unavailable rows, accounting
    terms, and limitations without artifact parsing, Warp import, or benchmark
    execution.
