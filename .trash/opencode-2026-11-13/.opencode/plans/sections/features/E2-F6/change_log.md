# E2-F6 Change Log

## 2026-07-03

- Created first-pass feature plan for issue #1172 feature E2-F6.
- Added four issue-sized phases covering numerical case definition, precision
  candidate comparison, conservation/fidelity/memory evaluation, and final
  recommendation publication.
- Documented E2-F1 dependency and parent E2 context.
- Captured codebase research findings: current CPU and GPU mass storage are
  absolute per-species `fp64`; CPU condensation has conservation-limited
  references; GPU condensation uses `wp.float64` and clamps negative masses.

## 2026-07-03

- Completeness review clarified the final report phase as a development-doc
  update gate for downstream dtype and schema tracks.

## 2026-07-05

- Updated the feature plan after shipping issue #1208 / E2-F6-P1.
- Recorded that `particula/gpu/tests/mass_precision_cases_test.py` now defines
  deterministic baseline cases for `npf_cluster`, `five_to_ten_nm`,
  `accumulation_mode`, and `cloud_droplet`.
- Recorded the new roadmap page `docs/Features/Roadmap/mass-precision-study.md`
  plus added roadmap links in `index.md` and `data-oriented-gpu.md`.
- Marked the current absolute per-species `np.float64` / `wp.float64` storage
  policy as the implemented baseline, while leaving candidate-comparison and
  recommendation work for later phases.

## 2026-07-08

- Marked E2-F6 and E2-F6-P4 as shipped now that
  `docs/Features/Roadmap/mass-precision-study.md` is the final acceptance
  artifact for particle mass precision policy.
- Closed the feature lifecycle as completed so parent epic E2 can close cleanly.
