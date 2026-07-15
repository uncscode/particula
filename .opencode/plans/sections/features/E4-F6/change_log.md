# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-12 | Initial E4-F6 plan drafted with five phases for device parity, strict conservation, graph capture, bounded autodiff readiness, and documentation | plan-feature-drafter |
| 2026-07-14 | Recorded completed E4-F6-P1 / issue #1308: two deterministic fp64 device-aware parity cases use an independent NumPy four-substep/P2/gas-coupled oracle and separately compare particle mass and gas concentration on Warp CPU and optional CUDA | plan-update-full |
| 2026-07-14 | Recorded completed E4-F6-P2 / issue #1309: test-only Warp-CPU P2 contract regressions cover per-box/per-species conservation, transfer and unweighted latent-energy accounting, immutable inputs, atomic invalid buffers, and deterministic fresh runs; production APIs are unchanged | plan-update-full |
| 2026-07-15 | Recorded completed E4-F6-P3 / issue #1310: added only `particula/gpu/kernels/tests/condensation_graph_capture_test.py`, providing test-only reusable-buffer graph-capture/replay readiness coverage for the public four-substep condensation step; production APIs and behavior are unchanged | plan-update-full |
