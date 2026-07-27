# Success Criteria

- [ ] E7-F1 and E7-F6 contracts are consumed rather than duplicated, and all
  public names follow their capability/error/export policy.
- [ ] Setup validates the complete CPU state and performs exactly one particle,
  one gas, and one environment upload.
- [ ] A session owns same-device fixed-shape Warp particle, gas, and environment
  state plus validated reusable process resources.
- [ ] Container, array, and supplied sidecar identities and shapes remain stable
  across at least two successful timesteps.
- [ ] No bulk CPU-to-Warp or Warp-to-CPU conversion occurs during normal step
  lifecycle operations.
- [ ] A nonterminal checkpoint performs one explicit synchronization and one
  restore per CPU container, preserves ordered metadata, and leaves the live
  session active.
- [ ] Finalization uses the same explicit boundary and makes the session
  terminal without a duplicate restore on repeated calls.
- [ ] Restart from a compatible checkpoint matches an uninterrupted
  deterministic Warp CPU run for resident state and lifecycle metadata.
- [ ] Preflight failures preserve caller-owned state; post-launch uncertainty
  faults the session and never triggers hidden rollback or CPU fallback.
- [ ] Missing Warp/CUDA and incompatible checkpoint/device requests produce
  E7-F6-defined explicit errors.
- [ ] Existing direct-kernel contracts and narrow scratch/configuration export
  boundaries remain unchanged.
- [ ] Warp CPU tests, focused coverage, linters, and strict documentation pass;
  optional CUDA rows skip cleanly when unavailable.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Setup uploads per CPU container | Caller-managed | Exactly 1 | Conversion spies |
| Bulk restores during normal timesteps | Caller-managed | 0 | Conversion guard |
| Explicit checkpoint synchronizations | Caller-managed | Exactly 1 per checkpoint | Warp sync spy |
| Resident container identity changes | Not standardized | 0 across steps | Identity assertions |
| Supplied sidecar identity changes | Not standardized | 0 across reuse | Registry tests |
| Deterministic restart state mismatch | No session contract | 0 for covered Warp CPU cases | Restart regression |
| Changed-module coverage | N/A | At least 80% | pytest-cov |
| Mandatory CUDA CI rows | 0 | 0; optional rows skip cleanly | pytest markers |
