# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-21 | Initial E6-F2/T2 plan drafted with E6-F1 dependency, explicit-transfer boundaries, atomic validation, five issue-sized phases, and issue acceptance criteria | plan-feature-drafter |
| 2026-07-22 | Completed E6-F2-P1 for issue #1395: added concrete-only input/no-write contract and co-located Warp tests; P2 re-export/mutation and P3 full state validation remain deferred | plan-update-full |
| 2026-07-22 | Completed E6-F2-P2 for issue #1396: shipped in-place finite-step particle/gas concentration decay, protected-state and no-op paths, and the sole `particula.gpu.kernels` export; P3 coefficient-value validation, complete preflight, and rollback remain deferred | plan-update-full |
| 2026-07-22 | Completed E6-F2-P3 for issue #1397: shipped ordered read-only preflight for exact same-device float64 Warp schemas and finite/nonnegative coefficient/concentration values, with precedence, no-side-effect, and full-state rejection coverage; rollback after a launched kernel failure remains deferred | plan-update-full |
| 2026-07-22 | Completed E6-F2-P4 for issue #1398: added test-only deterministic independent NumPy-reference finite-step parity and invariant coverage on required Warp CPU, with the identical scalar/per-box matrix as optional CUDA evidence; production API and documentation remain unchanged | plan-update-full |
| 2026-07-22 | Completed E6-F2-P5 for issue #1399: published the delivered direct GPU dilution P1–P4 contract in the foundation guide, roadmap, maintainer reference, README, and documentation indexes; added hardware-free contract, lazy-export, and local-link/anchor regression coverage | plan-update-full |
