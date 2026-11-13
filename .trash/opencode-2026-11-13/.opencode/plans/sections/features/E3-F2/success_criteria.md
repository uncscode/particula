# E3-F2 Success Criteria

## Pass / Fail Criteria

- [x] A deterministic mixed NPF/droplet fixture lands in the GPU coagulation
  test surface and reproduces the low-acceptance scenario that motivated E3-F2.
- [x] Acceptance-rate evidence is captured through explicit test/debug plumbing
  without introducing hidden production CPU↔GPU transfers.
- [ ] The chosen outcome is measurable: either the hardened path improves
  acceptance on the mixed-scale fixture, or the final docs record a quantified
  limitation and the accepted boundary for Epic C.
- [x] Mixed-scale collision application preserves mass and avoids invalid pair
  generation on Warp CPU, with CUDA-if-available following the same assertions.
- [x] The production and diagnostic kernels use the same bounded active-particle
  selector and keep accepted counts bounded by `collision_pairs.shape[1]`,
  `max_collisions`, and `n_particles // 2`.
- [x] Aggregate stochastic checks stay within the documented tolerance policy and
  use the finalized E3-F1 repeated-step RNG contract.
- [ ] Final documentation records the selected design, measured evidence,
  acceptance interpretation, and focused reproduction commands.

## Evidence Metrics

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Mixed-scale fixture coverage | At least one dedicated fixture/test path exists for the NPF-plus-droplet case | `particula/gpu/kernels/tests/coagulation_test.py` |
| Acceptance visibility | Baseline and chosen-path acceptance behavior can be inspected or asserted without hidden sync | Test-only diagnostics or bounded helper coverage from P1/P2 |
| Brownian correctness | Aggregate collision-rate checks remain within documented stochastic tolerance | `test_mixed_scale_brownian_collision_totals_match_expected_mean_within_sigma_tolerance(device)` |
| Conservation | Mixed-scale runs conserve total mass within the chosen stable tolerance, including zero-acceptance seeded trials | P2 conservation plus `test_mixed_scale_repeated_seeded_runs_conserve_total_mass_even_with_zero_acceptance_trials(device)` |
| Decision traceability | Docs state either the improvement achieved or the accepted limitation with commands to reproduce it | `docs/Features/Roadmap/data-oriented-gpu.md` or focused feature note |
