# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-12 | Initial E4-F1 plan drafted from issue #1272 acceptance criteria, E4 dependency gates, and codebase research; added five co-tested phases. | plan-feature-drafter |
| 2026-07-12 | Issue #1281 shipped E4-F1-P1 validation-only sidecar, required condensation boundary, executable-call migration, and regression coverage; formula/refresh and user-facing docs remain deferred. | plan-update-full |
| 2026-07-13 | Issue #1282 shipped E4-F1-P2 standalone `refresh_vapor_pressure_gpu`: validated Warp `float64` boundary, one pressure-matrix launch, constant and canonical Buck water/ice parity, concrete-module-only export, and co-located API/no-mutation tests; condensation integration remains P3. | plan-update-full |
| 2026-07-13 | Issue #1283 shipped E4-F1-P3: `condensation_step_gpu()` now requires keyword-only `ThermodynamicsConfig`, validates before mutation, casts direct float32 temperature on-device when needed, refreshes caller-owned vapor pressure once before environment preparation/mass transfer, and adds ordering, input-path, repeated-call, compatibility, and pre-refresh atomicity regressions. | plan-update-full |
| 2026-07-13 | Issue #1284 shipped E4-F1-P4 in commit `92b3797e7`: added Warp-backed public integration coverage for reusable thermodynamics/vapor-pressure/mass-transfer buffers, legacy positional mass-transfer compatibility, and atomic missing-sidecar/CUDA cross-device failures. Production code was unchanged. | plan-update-full |
| 2026-07-13 | Issue #1285 shipped E4-F1-P5 documentation-only work: recorded the canonical thermodynamics/derived-buffer refresh contract and E4-F2 through E4-F7 ownership. No production code or supported-model expansion was made. | plan-update-full |
