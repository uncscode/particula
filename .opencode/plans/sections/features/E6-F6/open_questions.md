# Open Questions

- [ ] Which deterministic resampling algorithm and tie-break key best preserve
  the required number, species-mass, charge, radius, and composition moments
  while guaranteeing a requested number of released slots?
  - Resolve in P1 before implementing CPU mutation.
- [ ] What exact distribution moments and mixed-scale error thresholds form the
  scientific acceptance contract?
  - Resolve in P1 with independent NumPy fixtures; do not infer thresholds from
    CPU/Warp agreement alone.
- [ ] What minimum/maximum representative-volume scale factor is physically and
  numerically acceptable, and how is indivisible source demand rounded?
  - Resolve in P1/P4; any remainder must be represented explicitly or cause a
    pre-mutation error, never truncation.
- [ ] Which diagnostics must E6-F7/E6-F8 retain: requested/admitted count,
  released slots, policy code, scale factor, and conservation residuals?
  - Resolve before P5 integration and keep buffers fixed shape/caller-owned.
- [x] Should scaling run before resampling when both are enabled?
  - Resolved 2026-07-21: No. Resampling-first precedence is mandatory; scaling
    is considered only when the planned resample remains insufficient.
- [x] What happens when both policies are disabled?
  - Resolved 2026-07-21: Capacity-sufficient calls proceed normally; actual
    exhaustion raises before mutation.
