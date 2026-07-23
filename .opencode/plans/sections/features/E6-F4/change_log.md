# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-21 | Initial E6-F4/T4 plan drafted with six issue-sized phases; preserved E6-F3 dependency, CPU image-charge/field semantics, neutral fallback, deterministic and stochastic validation, fixed-slot/RNG invariants, and Epic G boundaries | plan-feature-drafter |
| 2026-07-23 | Recorded issue #1409 P1 shipment: concrete `NeutralWallLossConfig` charged-mode/potential/field validation, rectangular field ownership and ordering/atomicity tests, and deliberately unchanged neutral execution; retained P2-P5 charged physics as deferred | plan-update-full |
| 2026-07-23 | Recorded issue #1410 P2 shipment: private fp64 Warp Coulomb self-potential-ratio and image-charge helpers plus independent NumPy/Warp parity and clipping coverage in the colocated dynamics test module; no exports, direct-kernel integration, config/preflight, CPU, potential/field composition, or RNG changes | plan-update-full |
