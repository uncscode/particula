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

- [x] **E3-F3-P3:** Document accepted GPU coagulation usage boundaries for Epic C
  - Issue: #1248 | Size: S | Status: Shipped
  - Depends on: E3-F3-P2 decision evidence so the usage boundary cites measured
    limits rather than inferred expectations.
  - Goal: State whether one-thread-per-box is accepted for Epic C and explain
    when the current low-level coagulation path should be used.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`
  - Tests: Read back the touched roadmap/foundations sections and validate the
    updated links, paths, and cross-references; notebook/source validation was
    intentionally skipped because the notebook-backed benchmark source was not
    edited.
  - Delivered: The roadmap now records the Epic C outcome as accepted with
    caveat and points readers to the measured decision record, while the
    foundations guide now states the appropriate many-box/direct-kernel uses,
    the large-single-box caveat, and the existing explicit-transfer/Warp-
    optionality boundaries.

- [x] **E3-F3-P4:** Scope parallel-within-box follow-up if scaling evidence requires it
  - Issue: #1249 | Size: XS | Status: Shipped
  - Depends on: confirming whether future evidence has shown the
    accepted-with-caveat boundary is no longer sufficient.
  - Goal: Close the follow-up decision for Epic C by recording whether the
    shipped evidence requires a bounded future investigation, without
    implementing optimization work.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `.opencode/plans/sections/features/E3-F3/open_questions.md`,
    `.opencode/plans/sections/features/E3-F3/change_log.md`
  - Tests: Documentation validation only; no production performance work.
  - Delivered: The shipped P2/P3 evidence was re-read and still supports the
    accepted-with-caveat Epic C decision, so P4 closed with no new
    parallel-within-box follow-up track. The roadmap and `open_questions.md`
    already stated that bounded outcome, so this phase updated the plan-state
    record via `change_log.md` and `phase_details.md` only.
