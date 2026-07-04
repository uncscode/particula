# E2-F6 Change Log

## 2026-07-03

- Created first-pass feature plan for issue #1172 track T6.
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
