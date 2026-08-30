# Phase Details

- [ ] **E8-F6-P1:** Define resident benchmark matrix and reproducible artifact schema with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add frozen case/result records and serializers in one concrete test
    support module; keep each record under 100 LOC and reject malformed rows
    before any CUDA probe or allocation.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/gpu/tests/benchmark_test.py`
  - Tests: Schema construction/validation, stable JSON serialization, command and
    device metadata, canonical case IDs, and malformed-record rejection.

- [ ] **E8-F6-P2:** Benchmark captured versus uncaptured repeated resident timesteps with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add <=100-LOC timing helpers that time one E8-F2 prepared fixture
    through uncaptured enqueue and E8-F4 replay, retaining raw samples separately
    from setup and capture costs.
  - Files: `particula/execution/tests/resident_benchmark_support.py`,
    `particula/execution/tests/resident_benchmark_support_test.py`,
    `particula/gpu/tests/benchmark_test.py`
  - Tests: Warmup/setup exclusion, synchronization placement, deterministic
    summary calculations, and identical fixture/identity routing.

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
