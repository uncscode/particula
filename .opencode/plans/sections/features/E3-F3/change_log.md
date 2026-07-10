# Change Log

## 2026-07-08

- Created first-pass feature plan for E3-F3.
- Added four issue-sized phases covering benchmark reproduction, measured
  decision recording, user-facing usage-boundary documentation, and optional
  follow-up scoping.
- Anchored the plan to E3-F2 dependency, current one-thread-per-box
  coagulation kernel behavior, existing opt-in benchmark infrastructure, and
  CUDA optionality constraints.

## 2026-07-08 — Completeness Review

- Expanded success criteria into measurable pass/fail checks and evidence
  metrics tied to benchmark coverage, reproducibility context, and final
  acceptance guidance.
- Clarified the final phase so it always leaves explicit developer-facing
  roadmap/follow-up documentation, even when the outcome is a scoped follow-up.

## 2026-07-10

- Updated E3-F3 plan sections to reflect shipped E3-F3-P1 work for issue
  #1246.
- Recorded that `particula/gpu/tests/benchmark_test.py` now uses a dedicated
  coagulation-only mixed-scale fixture while condensation continues to use the
  generic helper.
- Recorded focused helper coverage in
  `particula/gpu/tests/benchmark_helpers_test.py` for the deterministic
  mixed-scale fixture, helper-routing split, benchmark result recording, and
  persistent RNG-state reuse.
- Recorded that `docs/Features/Roadmap/data-oriented-gpu.md` now carries the
  compact shipped benchmark-evidence note and artifact path.

## 2026-07-10 — Issue #1247

- Updated E3-F3 plan sections to reflect shipped E3-F3-P2 documentation work.
- Recorded the measured single-box caution band (`1x10k` to `1x50k`) and the
  measured many-box effective region
  (`10x500`, `10x1k`, `50x1k`, `10x5k`, `50x5k`, `100x1k`, `10x10k`) as the
  current decision record for the one-thread-per-box coagulation path.
- Recorded that the notebook-backed theory/example source text was aligned to
  the controlled benchmark artifact path and machine-bounded interpretation used
  in the roadmap.
