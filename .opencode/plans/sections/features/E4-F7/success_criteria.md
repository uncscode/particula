# Success Criteria

- [ ] Canonical documentation enumerates every supported vapor-pressure, activity, surface-tension, latent-heat, gas-coupling, diagnostic, scratch, shape, precision, and device contract delivered by E4-F1 through E4-F6.
- [ ] Unsupported BAT/staggered modes, high-level integration, automatic backend selection, implicit fallback, and hidden transfer are explicit.
- [ ] The example runs on Warp CPU from a clean supported environment and visibly performs CPU-to-Warp conversion, low-level execution, and explicit checkpoint restore.
- [ ] The example preserves ordered species metadata, uses reusable fixed-shape fp64 buffers, performs no per-substep host refresh, and matches documented conservation/energy semantics.
- [ ] Troubleshooting covers configuration, order, shape, device, environment exclusivity, physical validation, scratch, inventory limiting, diagnostics, and missing Warp/CUDA.
- [ ] Every published command resolves to a real post-E4 file/marker and passes on its required backend; optional CUDA skips cleanly when unavailable.
- [ ] Issue 1272 documentation tests are revised deliberately and continue to reject unsupported high-level or hidden-transfer claims.
- [ ] Roadmap shipped status is withheld until E4-F1 through E4-F6 evidence and E4-F7 publication checks all pass.

## Metrics

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Canonical support matrices | Pre-E4/general GPU contract | 1 final condensation matrix | `data-containers-and-gpu-foundations.md` |
| Runnable supported condensation examples | Pre-E4 particle-only path | 1 Warp CPU explicit-transfer workflow | Example test |
| Hidden CPU/GPU transfers in example | Not contract-tested | 0 | Example source assertions/review |
| Required focused commands passing on Warp CPU | Not published as a set | 100% | Focused pytest runs |
| Optional CUDA behavior without CUDA | Partial/local | 100% clean skips | Marker run |
| Broken internal links or stale command paths | Unknown | 0 | Docs validation |
| Unsupported high-level backend claims | Guarded as future work | 0 | Documentation guardrail tests |
