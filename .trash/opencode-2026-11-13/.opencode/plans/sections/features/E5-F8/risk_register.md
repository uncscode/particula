# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| CPU expected values accidentally reuse Warp state or production output | Medium | High | Separate builders, immutable fixture constants, identity/no-alias assertions, and perturbation tests | P1 implementer |
| One aggregate pass hides a failed conservation or energy invariant | Medium | High | Three result types, independent calculations, category-isolation tests, and all-categories-required exit gate | P2 implementer |
| Physics tolerances are invented, relaxed, or confused with invariant tolerances | Medium | High | Copy current canonical per-field parity tolerances during implementation, print them, and lock them in tests; retain strict invariant thresholds | Scientific reviewer |
| Example is mistaken for high-level CPU strategy parity | Medium | High | Name the NumPy fixed-four oracle explicitly and repeat the direct-kernel non-claim in output, report, and indexes | Documentation owner |
| A skipped Warp/CUDA test is reported as execution evidence | Low | High | Distinct PASS/SKIP states; Warp CPU required when installed; CUDA marked optional only | Test owner |
| Deferred capability is listed without an actionable owner or silently drops from docs | Medium | High | Required-row parser test with owner, entry gate, and non-claim columns | Roadmap owner |
| Phase-aware surface tension or BAT activity is assigned to an unrelated runtime epic | Medium | Medium | Route scientific-model additions through a separately approved condensation-physics plan before coding | Scientific governance owner |
| Ownership wording drifts across canonical docs | Medium | Medium | One canonical record, inbound links instead of duplicated tables, documentation content tests | E5-F9 owner |
| Walkthrough duplicates or destabilizes the existing broad quick start | Low | Medium | Keep a focused parity script with shared conventions, not shared mutable state or a replacement entry point | P1 implementer |
| Optional Warp absence reduces local coverage | Medium | Medium | Test deterministic import/no-Warp behavior and require Warp CPU in installed-Warp CI environments | CI owner |
