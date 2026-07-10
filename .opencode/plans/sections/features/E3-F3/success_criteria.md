# Success Criteria

## Pass / Fail Criteria

- [x] Refreshed benchmark evidence covers the existing single-box and multi-box
  coagulation matrix, with shipped timings recorded in the roadmap note.
- [x] The plan captures the benchmark command, hardware/runtime context, and the
  measured point where one-thread-per-box behavior becomes a practical caution.
- [ ] Epic C documentation states whether the current design is accepted as-is
  for the roadmap or whether a bounded parallel-within-box follow-up is needed.
- [ ] User-facing docs explain when the low-level GPU coagulation path is
  appropriate, when it is not, and where future scaling work belongs.
- [x] CUDA stays optional for normal development: helper/unit tests remain fast,
  while benchmark execution stays opt-in.
- [ ] The feature does not introduce production graph-capture optimization or a
  parallel-within-box kernel rewrite.

## Evidence Metrics

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Benchmark coverage | Single-box and multi-box cases are rerun or explicitly skipped with reason | `particula/gpu/tests/benchmark_test.py` notes/results |
| Reproducibility | Recorded command, date, and hardware/Warp context accompany the decision | Roadmap section or small benchmark artifact |
| Acceptance decision | Docs state accept-current-design vs follow-up-needed with measured rationale | `docs/Features/Roadmap/data-oriented-gpu.md` |
| Optional CUDA policy | CPU-only workflows remain green while CUDA benchmark work stays opt-in | Focused helper tests and benchmark skip behavior |
| Scope control | Any helper changes are co-tested without widening into production optimization | Same-phase helper/unit coverage |
