# Change Log

## 2026-07-04

- Drafted feature plan `E2-F3` under parent epic issue `#1172` / feature
  `E2-F3`.
- Added four implementation phases covering Warp struct design, CPU-to-Warp
  conversion, Warp-to-CPU round trips, CUDA-parametrized coverage, and docs.
- Captured dependency on `E2-F2` for the CPU `EnvironmentData` schema.
- Recorded explicit transfer semantics and no-hidden-transfer acceptance
  criteria.
- Updated plan sections after implementation issue `#1192` shipped `E2-F3-P1`
  with
  `WarpEnvironmentData` in `particula/gpu/warp_types.py` and focused schema
  coverage in `particula/gpu/tests/warp_types_test.py`.
- Recorded that no conversion helpers, package export changes, or external docs
  landed in this implementation slice.

## 2026-07-03

- Completeness review aligned `success_criteria` with the accepted environment
  shape contract and the documented default `device="cuda"` transfer helper
  behavior.
