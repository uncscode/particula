# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|---|---|---|---|---|---|
| Stale vapor pressure after latent temperature change | Medium | High | Refresh on-device before every substep; temperature-sensitive tests | E4-F1 | Open |
| Activity/surface edge cases produce NaN or division by zero | Medium | High | Dry/zero-water guards, positive-finite validation, CPU fixtures | E4-F2 | Open |
| Four-step buffers alias or retain stale data | Medium | High | Preflight validation, explicit zeroing, reuse/alias regression tests | E4-F3 | Open |
| Latent correction has wrong sign or unstable denominator | Medium | High | Port isolated CPU equations; zero-latent fallback and strict energy checks | E4-F4 | Open |
| Aggregate demand exceeds gas inventory | High | High | Per-box/species aggregation and deterministic inventory scaling | E4-F5 | Open |
| CPU/GPU parity hides mixed-scale error | Medium | High | Per-case and per-species tolerances plus strict conservation | E4-F6 | Open |
| CUDA-only behavior escapes CI | Medium | Medium | Warp CPU required; optional CUDA matrix with clean skips and commands | E4-F6 | Open |
| Documentation overstates supported integration | Low | High | Explicit low-level support matrix and non-goals review | E4-F7 | Open |
