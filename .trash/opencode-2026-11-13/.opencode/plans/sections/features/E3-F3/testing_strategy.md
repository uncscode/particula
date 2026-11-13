# Testing Strategy

## Fast Validation

- For the shipped issue #1248 docs-only scope, validate the touched markdown by
  readback against the current roadmap evidence block and the existing
  foundations-guide transfer/optionality wording.
- Verify touched links, anchors, file paths, and cross-references in:
  - `docs/Features/Roadmap/data-oriented-gpu.md`
  - `docs/Features/data-containers-and-gpu-foundations.md`
- Confirm the updated wording answers both “when should I use this?” and “when
  should I not?” without implying hidden transfers, hidden synchronization, or
  broader production support.
- Skip notebook sync/validation because
  `docs/Theory/nvidia-warp/examples/gpu_benchmarks.py` was not edited in this
  phase.

## Benchmark Validation

- No benchmark rerun is required for the shipped docs-only phase because the
  roadmap decision text is anchored to the existing P2 measured evidence.
- Preserve the exact measured-evidence framing already recorded in the roadmap
  rather than inventing new timings or broadening the recommendation.

## Co-located Testing Policy

This feature is now primarily implementation-record maintenance for shipped
documentation. No new pytest coverage is expected when the phase only updates
roadmap/foundations wording.

## Non-goals for Tests

- Do not require CUDA in default test runs.
- Do not rerun benchmarks or add helper coverage when the implementation change
  is limited to documentation wording.
- Do not add graph-capture or production optimization tests in this feature.
