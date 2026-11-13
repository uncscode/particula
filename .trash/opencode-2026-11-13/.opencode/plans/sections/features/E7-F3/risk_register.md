# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Adapter silently reseeds RNG each timestep because `rng_seed` is repeated | Medium | High | Require caller-owned persistent state; default normal execution to no reset; test progression and explicit reset replay | E7-F3 implementer |
| Selection promotes direct-kernel mechanisms beyond issue #1451 T3 | Medium | High | Admit Brownian only; parameterize charged, sedimentation, turbulent, and combined capability errors | E7-F3/E7-F6 owners |
| CPU and Warp stochastic trajectories are incorrectly treated as exact parity | Medium | Medium | Compare deterministic physics inputs and invariants; use aggregate statistical bounds; document no exact replay promise | Test owner |
| Hidden transfer, synchronization, or fallback enters adapter convenience code | Medium | High | Keep typed resident inputs; use conversion/sync spies; propagate failures; enforce E7-F6 policy | Adapter owner |
| Caller output or RNG buffers have wrong shape, dtype, device, or capacity | Medium | High | Preserve direct-kernel preflight; add selection schemas and negative identity/atomicity tests | Adapter owner |
| Result type implies RNG or collision counts are host values | Low | Medium | Return resource references and mutation metadata; prohibit implicit `.numpy()` or synchronization | API owner |
| Launch failure is documented as atomic despite in-place asynchronous work | Low | High | Distinguish pre-launch atomic rejection from post-launch no rollback in API/docs/tests | Documentation owner |
| E7-F3 duplicates E7-F8 stream/restart responsibilities | Medium | Medium | Limit T3 to per-box buffer ownership, seed/reuse/reset seam; defer stream identity and checkpoints explicitly | E7 epic owner |
| Warp-only imports break CPU-only environments | Medium | High | Follow E7-F6 lazy/scoped import policy and add fresh-process CPU-only import tests | API owner |
| Tests weaken existing direct-kernel validation or coverage | Low | High | Reuse fixtures, keep thresholds, run focused and existing regressions, require >=80% changed-module coverage | Test owner |
