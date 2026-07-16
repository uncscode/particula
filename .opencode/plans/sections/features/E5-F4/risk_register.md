# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| A radius-extrema shortcut does not bound composition-dependent differential settling rates | Medium | High | Use the exhaustive active-pair maximum initially; require proof and pairwise regression before optimization | P2 implementer |
| CPU reference defaults to an unimplemented collision-efficiency callback | High | High | Build expected values with `calculate_collision_efficiency=False`; encode GPU efficiency as constant 1 and expose no efficiency input | P1 implementer |
| Effective mixture density is derived inconsistently from species masses/densities | Medium | High | Define `total_mass / sum(mass_s / density_s)`, test pure and mixed compositions independently, and reject invalid species density preflight | P1 implementer |
| Non-finite properties or rates produce invalid scheduling or acceptance ratios | Medium | High | Validate caller domains before launch; guard device-derived values; assert finite non-negative rate and majorant invariants | P2 implementer |
| Exhaustive majorant scanning is expensive for large fixed slots | High | Medium | Accept the explicit correctness-first O(n^2) bound for E5-F4, scan active slots only, preserve trial caps, and defer optimization until a safe proof exists | E5 owner |
| Invalid sedimentation requests mutate RNG or output buffers before failing | Medium | High | Complete mechanism/domain/buffer validation before allocation, RNG initialization, or launch; add full-state snapshot tests | P3 implementer |
| Sedimentation work forks the shared sampler and breaks future additive semantics | Low | High | Depend on E5-F1 and add only property/rate/majorant branches; reserve combinations for E5-F6 | E5-F4 reviewer |
| Optional CUDA availability masks missing baseline evidence | Low | Medium | Require Warp CPU whenever Warp is installed and treat CUDA only as additive evidence with clean skips | Test owner |
| Documentation overstates parity or supported model variants | Medium | Medium | Use the explicit support-limit checklist and defer final combination claims/example to E5-F9 | P4 owner |
