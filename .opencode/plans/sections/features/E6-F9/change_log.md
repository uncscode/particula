# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-21 | Initial E6-F9 plan drafted with four dependency-gated phases, mandatory E6-F1 through E6-F8 dependencies, integrated validation, explicit-transfer example, roadmap cross-links, and Epic G scope exclusions | plan-feature-drafter |
| 2026-07-26 | Recorded #1446 completion of E6-F9-P1: private deterministic fp64 fixture/invariant coverage in `particula/gpu/tests/process_sequence_test.py`; no production/public API or integrated process-sequence behavior added | plan-update-full |
| 2026-07-26 | Recorded #1447 completion of E6-F9-P2: private test-only resident composition coverage for the five existing direct GPU boundaries in `particula/gpu/tests/process_sequence_test.py`, with persistent sidecars/RNGs, final-only conversion checks, Warp CPU baseline, and optional CUDA coverage; no production API or coordinator added | plan-update-full |
| 2026-07-26 | Recorded #1448 completion of E6-F9-P3: added the explicit-transfer five-step direct-Warp example and its focused regression suite. The example converts each CPU container once, uses caller-owned sidecars/RNG, synchronizes once, restores once at the final checkpoint, has deterministic no-Warp behavior, and never falls back to CPU work | plan-update-full |
| 2026-07-26 | Recorded #1449 completion of E6-F9-P4: published the E6 roadmap inventory, direct-boundary ownership guidance, and hardware-free documentation/closeout regression. P4 command evidence passed, while E6-F2, E6-F5, E6-F6, and E6-F8 keep E6-F9/E6 Draft and Epic G pending | adw-build-refine |
| 2026-07-26 | Reconciled completed upstream records, marked E6-F9 and P4 Shipped/completed, removed fragile mutable-status assertions from the documentation regression, and closed E6. | OpenCode |
