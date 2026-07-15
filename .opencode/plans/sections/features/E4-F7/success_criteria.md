# Success Criteria

- [x] Canonical documentation enumerates the shipped bounded direct-condensation vapor-pressure, activity, surface-tension, latent-heat, gas-coupling, diagnostic, scratch, shape, precision, and device contract.
- [x] Unsupported BAT/staggered modes, high-level integration, automatic backend selection, implicit fallback, and hidden transfer are explicit.
- [x] The example runs on Warp CPU from a clean supported environment and visibly performs CPU-to-Warp conversion, low-level execution, and explicit checkpoint restore.
- [x] The example preserves ordered species metadata, uses reusable fixed-shape fp64 buffers, performs no per-substep host refresh, and regression coverage checks documented gas-conservation and energy-sidecar semantics.
- [x] Troubleshooting covers configuration, order, shape, device, environment exclusivity, physical validation, scratch, inventory limiting, diagnostics, and missing Warp/CUDA.
- [x] Every published command resolves to a real post-E4 file/marker and passes on its required backend; optional CUDA skips cleanly when unavailable.
- [x] Text-only publication tests deliberately cover the canonical foundations and migration wording and continue to reject unsupported high-level or hidden-transfer claims.
- [x] Issue #1317 published roadmap shipped status only after verifying E4-F1
  through E4-F6 and E4-F7-P1 through P3 completion records and passing the
  focused text-only documentation suite; later high-level GPU integration is
  not enabled by this direct-kernel milestone.

## Metrics

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Canonical support matrices | Pre-E4/general GPU contract | 1 final condensation matrix | `data-containers-and-gpu-foundations.md` |
| Runnable supported condensation examples | Pre-E4 particle-only path | 1 Warp CPU explicit-transfer workflow | `gpu_direct_kernels_example_test.py` |
| Hidden CPU/GPU transfers in example | Not contract-tested | 0 | No-Warp/import and mocked helper-call regressions |
| Required focused commands passing on Warp CPU | Not published as a set | 100% | Focused pytest runs |
| Optional CUDA behavior without CUDA | Partial/local | 100% clean skips | Marker run |
| Broken internal links or stale command paths | Unknown | 0 | Docs validation |
| Unsupported high-level backend claims | Guarded as future work | 0 | Documentation guardrail tests |
