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

## 2026-07-10 — Issue #1248

- Updated E3-F3 plan sections to reflect shipped E3-F3-P3 docs-only work.
- Recorded that public documentation changes were limited to
  `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/data-containers-and-gpu-foundations.md`.
- Recorded the accepted-with-caveat Epic C outcome for the current low-level
  one-thread-per-box coagulation path and the user-facing guidance boundaries
  that shipped with it.
- Recorded that notebook-backed benchmark sources were intentionally left
  unchanged in this phase.

## 2026-07-10 — Issue #1249

- Updated E3-F3 plan sections to reflect shipped E3-F3-P4 closeout work.
- Recorded that the shipped P2/P3 evidence still stands, so this phase closed
  without opening a new parallel-within-box follow-up track.
- Recorded that roadmap wording and `open_questions.md` already matched the
  accepted-with-caveat outcome, so the closeout stayed limited to plan-state
  updates instead of adding new roadmap or kernel-scope changes.
