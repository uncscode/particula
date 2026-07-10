# Phase Details

- [x] **E3-F3-P1:** Reproduce single-box and multi-box coagulation benchmark matrix
  - Issue: #1246 | Size: S | Status: Shipped
  - Depends on: E3-F1 finalized repeated-step RNG semantics and E3-F2 settling
    any sampler changes that would invalidate the benchmark baseline.
  - Goal: Refresh the existing CUDA opt-in coagulation benchmark evidence for
    representative single-box and multi-box configurations, then record the
    exact command and run outcome in one shipped docs location.
  - Files: `particula/gpu/tests/benchmark_test.py`,
    `particula/gpu/tests/benchmark_helpers_test.py`,
    `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: Shipped focused helper coverage in
    `pytest particula/gpu/tests/benchmark_helpers_test.py -q`, including the
    deterministic coagulation-only mixed-scale fixture contract, helper-routing
    isolation, benchmark result recording, and persistent RNG-state reuse.
    Captured benchmark evidence with
    `pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s` on CUDA
    hardware and recorded the artifact path plus single-box vs multi-box timing
    summary in the roadmap.
  - Delivered: `benchmark_test.py` now routes coagulation through
    `_make_coagulation_particle_data(...)` while leaving condensation on the
    generic helper; `benchmark_helpers_test.py` adds bounded regression
    coverage for the mixed-scale fixture and helper split; the roadmap note now
    records the 2026-07-10 UTC benchmark command, hardware context, artifact
    path, and timing summary.

- [x] **E3-F3-P2:** Record measured one-thread-per-box scaling limit and decision evidence
  - Issue: #1247 | Size: S | Status: Shipped
  - Depends on: E3-F3-P1 benchmark matrix and captured environment notes.
  - Goal: Convert benchmark results into a clear measured limit for large
    single-box and many-box coagulation workloads.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Theory/nvidia-warp/examples/gpu_benchmarks.py`
  - Tests: Read back the roadmap decision-record subsection and aligned
    notebook-backed example text; no benchmark rerun or kernel changes were
    required for this documentation-only issue.
  - Delivered: The roadmap evidence block now names the measured single-box
    caution band for the current one-thread-per-box path (`1x10k` to `1x50k`)
    and separately names the measured many-box effective region
    (`10x500`, `10x1k`, `50x1k`, `10x5k`, `50x5k`, `100x1k`, `10x10k`). The
    notebook-backed theory/example text was aligned to the controlled artifact
    path `.artifacts/benchmarks/gpu_benchmark_results.json` and to the same
    machine-bounded recommendation wording.

- [ ] **E3-F3-P3:** Document accepted GPU coagulation usage boundaries for Epic C
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F3-P2 decision evidence so the usage boundary cites measured
    limits rather than inferred expectations.
  - Goal: State whether one-thread-per-box is accepted for Epic C and explain
    when the current low-level coagulation path should be used.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`, benchmark example
    docs as needed.
  - Tests: Markdown/link validation and any paired notebook/source validation
    required by edited docs.

- [ ] **E3-F3-P4:** Scope parallel-within-box follow-up if scaling evidence requires it
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E3-F3-P3 concluding that documentation alone is insufficient for
    Epic C acceptance.
  - Goal: If measurements show the current design is not acceptable for Epic C,
    create a bounded follow-up plan or issue without implementing optimization,
    and update the roadmap/follow-up documentation so the final Epic C decision
    is explicit for future implementers.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `.opencode/plans/sections/features/E3-F3/open_questions.md`,
    `.opencode/plans/sections/features/E3-F3/change_log.md`
  - Tests: Documentation validation only; no production performance work.
  - Deliverable: Update developer-facing roadmap guidance whether the outcome is
    "accepted with boundaries" or "follow-up required," so the closeout phase
    leaves explicit documentation behind.
