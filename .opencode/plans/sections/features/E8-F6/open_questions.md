# Open Questions

- [x] Which Warp/CUDA memory API is sufficiently stable to serve as the observed
  peak-memory source of record?
  - Resolved 2026-08-30: Use a documented public allocator high-water API only
    after qualifying it against the active Warp/CUDA version and documenting its
    pool, CUDA graph-storage, and relevant non-Warp allocation coverage. When no
    API meets that contract, record observed peak memory as unavailable.
  - Rationale: Analytical shape/dtype and registry bytes remain the portable
    authority. A version-qualified allocator probe supplies bounded observed
    evidence without substituting sampled NVML values or claiming coverage the
    API does not provide.
  - Options:
    - [x] A. Qualify one documented public allocator peak API by Warp/CUDA version (Selected)
    - [ ] B. Use a qualified CUDA or NVML process-level measurement with documented exclusions
    - [ ] C. Publish no observed peak and retain analytical and registry bytes only
  - Evidence:
    - `particula/gpu/tests/benchmark_test.py:246` - Existing metadata exposes
      total device memory but no runtime peak.
    - `particula/gpu/tests/benchmark_test.py:312` - Current memory accounting is
      deterministic shape-and-dtype analysis.
    - `.opencode/plans/sections/epics/E8/open_questions.md:47` - The epic-level
      policy keeps analytical/registry accounting portable and observed peaks
      conditional on a version-qualified API.
  - Resolved by: user decision

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

- [x] What tape projection scenarios should be frozen before Epic I has measured
  storage records?
  - Resolved 2026-08-30: Publish symbolic full-retention and checkpoint-interval
    scenarios. Define per-step retained-state bytes `S`, checkpoint bytes `C`,
    timesteps `T`, and checkpoint interval `K`; report `T * S` and
    `ceil(T / K) * C + K * S` as projected scenarios, not measured tape usage.
  - Rationale: Auditable formulas expose timestep and checkpoint scaling without
    inventing an empirical multiplier or claiming to represent unknown Epic I
    operation/intermediate storage.
  - Options:
    - [x] A. Publish symbolic full-retention and checkpoint-interval projections (Selected)
    - [ ] B. Publish only an fp64 full-retention lower bound
    - [ ] C. Defer every tape projection until Epic I produces measurements
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1910` - Tape storage is expected
      to scale with timesteps and fp64 resident state and may need checkpointing.
  - Resolved by: user decision

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
