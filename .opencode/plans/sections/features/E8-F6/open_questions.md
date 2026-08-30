# Open Questions

- [ ] Which Warp/CUDA memory API is sufficiently stable to serve as the observed
  peak-memory source of record?
  - Open: Current code records device capacity and analytical bytes but no
    allocator high-water mark, and the Warp dependency is not version-bounded.
  - Recommendation: **A - Qualify one documented public allocator peak API by Warp/CUDA version**
  - Suggested answer: Choose **A** because source-of-record status requires known
    pool, graph-storage, and non-Warp allocation coverage; otherwise record the
    observed measurement as unavailable.
  - Options:
    - [ ] A. Qualify one documented public allocator peak API by Warp/CUDA version (Recommended)
    - [ ] B. Use a qualified CUDA or NVML process-level measurement with documented exclusions
    - [ ] C. Publish no observed peak and retain analytical and registry bytes only
  - Evidence considered:
    - `particula/gpu/tests/benchmark_test.py:246` - Existing metadata exposes
      total device memory but no runtime peak.
    - `particula/gpu/tests/benchmark_test.py:312` - Current memory accounting is
      deterministic shape-and-dtype analysis.

- [x] Should the 1000-box row use one canonical particle/species fixture or a
  budget-derived smaller capacity?
  - Resolved 2026-08-30: Attempt the same canonical particle/species fixture
    across the box-count sweep; if the 1000-box row exceeds budget, record it as
    unavailable and add only a separately labeled reduced-capacity row.
  - Rationale: This preserves cross-box comparability without hiding actual
    dimensions or discarding useful bounded evidence.
  - Evidence:
    - `particula/gpu/tests/benchmark_test.py:426` - Existing benchmarks gate
      oversized cases before allocation using explicit required bytes.
    - `.opencode/plans/sections/features/E8-F6/success_criteria.md:3` - E8-F6
      requires represented 1, 10, 100, and 1000-box rows where feasible.
  - Resolved by: plan-question-resolver

- [ ] What tape projection scenarios should be frozen before Epic I has measured
  storage records?
  - Open: The roadmap establishes fp64 state scaling and checkpointing as a
    concern but does not authorize empirical multipliers or checkpoint intervals.
  - Recommendation: **A - Publish symbolic full-retention and checkpoint-interval projections**
  - Suggested answer: Choose **A** because formulas can be auditable and clearly
    labeled projected without pretending to be measured tape storage.
  - Options:
    - [ ] A. Publish symbolic full-retention and checkpoint-interval projections (Recommended)
    - [ ] B. Publish only an fp64 full-retention lower bound
    - [ ] C. Defer every tape projection until Epic I produces measurements
  - Evidence considered:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1910` - Tape storage is expected
      to scale with timesteps and fp64 resident state and may need checkpointing.

- [x] Should benchmark JSON remain one aggregate file or add one file per run?
  - Resolved 2026-08-30: Retain one aggregate source-of-record JSON per benchmark
    invocation, with run metadata, schema version, and per-row status; distinct
    invocations may use controlled filename overrides.
  - Rationale: Existing incremental full-dictionary overwrite preserves partial
    results and keeps artifacts under one validated root.
  - Evidence:
    - `particula/gpu/tests/benchmark_test.py:438` - Results are accumulated under
      one benchmark map with metadata.
    - `particula/gpu/tests/benchmark_test.py:450` - Each save overwrites one file
      with the complete current result dictionary.
  - Resolved by: plan-question-resolver
