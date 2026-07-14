# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-12 | Initial four-phase plan; preserved issue #1272 signed whole-call energy criteria and E4-F1/F2/F3 gates | plan-feature-drafter |
| 2026-07-14 | Issue #1297 shipped P1 private fp64 thermal helpers, atomic latent/thermal-work sidecar preflight, and formula/contract regressions; correction integration and energy accounting remain deferred | plan-update-full |
| 2026-07-14 | Issue #1298 shipped P2 latent-rate correction in all four fixed GPU condensation substeps, shared activity/Kelvin surface pressure, exact omitted/zero isothermal preservation, CPU-oracle parity, atomic validation, deterministic scratch-reuse coverage, and deferred `thermal_work` semantics without public API expansion | plan-update-full |
| 2026-07-14 | Issue #1299 shipped P3 optional caller-owned device-only energy output: atomic preflight, one post-preflight clear, single-writer final box/species reduction of bounded accumulated transfer times latent heat, unchanged disabled path/two-item return, and Warp CPU plus optional CUDA regression coverage | plan-update-full |
