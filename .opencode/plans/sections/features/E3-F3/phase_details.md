# Phase Details

- [ ] **E3-F3-P1:** Reproduce single-box and multi-box coagulation benchmark matrix
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F1 finalized repeated-step RNG semantics and E3-F2 settling
    any sampler changes that would invalidate the benchmark baseline.
  - Goal: Refresh the existing CUDA opt-in coagulation benchmark evidence for
    representative single-box and multi-box configurations.
  - Files: `particula/gpu/tests/benchmark_test.py`, benchmark result artifact or
    documentation note selected by implementer.
  - Tests: Run focused fast tests for benchmark helpers; run
    `pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s` only on
    CUDA-capable hardware and record skip status otherwise.

- [ ] **E3-F3-P2:** Record measured one-thread-per-box scaling limit and decision evidence
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F3-P1 benchmark matrix and captured environment notes.
  - Goal: Convert benchmark results into a clear measured limit for large
    single-box and many-box coagulation workloads.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, optional benchmark
    results artifact, `docs/Theory/nvidia-warp/examples/gpu_benchmarks.py`.
  - Tests: Validate documentation examples or metadata helpers touched by the
    result recording; keep CUDA-dependent benchmark execution optional.

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
  - Files: roadmap decision section, open questions/follow-up tracker, docs
    change log.
  - Tests: Documentation validation only; no production performance work.
  - Deliverable: Update developer-facing roadmap guidance whether the outcome is
    "accepted with boundaries" or "follow-up required," so the closeout phase
    leaves explicit documentation behind.
