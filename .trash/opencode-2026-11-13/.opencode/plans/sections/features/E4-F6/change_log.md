# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-12 | Initial E4-F6 plan drafted with five phases for device parity, strict conservation, graph capture, bounded autodiff readiness, and documentation | plan-feature-drafter |
| 2026-07-14 | Recorded completed E4-F6-P1 / issue #1308: two deterministic fp64 device-aware parity cases use an independent NumPy four-substep/P2/gas-coupled oracle and separately compare particle mass and gas concentration on Warp CPU and optional CUDA | plan-update-full |
| 2026-07-14 | Recorded completed E4-F6-P2 / issue #1309: test-only Warp-CPU P2 contract regressions cover per-box/per-species conservation, transfer and unweighted latent-energy accounting, immutable inputs, atomic invalid buffers, and deterministic fresh runs; production APIs are unchanged | plan-update-full |
| 2026-07-15 | Recorded completed E4-F6-P3 / issue #1310: added `particula/gpu/kernels/tests/condensation_graph_capture_test.py`, which records the public-step capture limitation: CPU capability skip and CUDA strict-xfail due to host validation readbacks; production APIs and behavior are unchanged | plan-update-full |
| 2026-07-15 | Recorded completed E4-F6-P4 / issue #1311: added only `particula/gpu/kernels/tests/condensation_autodiff_test.py`, providing bounded out-of-place raw-rate Warp Tape versus centered-fp64 checks, access-verification cleanup coverage, optional CUDA evidence, and forward-only P2 limitation tests; production APIs and published documentation are unchanged | plan-update-full |
| 2026-07-15 | Recorded completed E4-F6-P5 / issue #1312: documentation commit `2c6531adc` published the P1--P4 direct-condensation evidence matrix, focused commands, and explicit boundaries; P3 is recorded as CPU capability-skip/CUDA strict-xfail unsupported capture, not replay support | plan-update-full |
