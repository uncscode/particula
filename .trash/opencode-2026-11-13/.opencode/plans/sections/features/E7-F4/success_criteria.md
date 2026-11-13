# Success Criteria

- [ ] E7-F1 and E7-F6 contracts are consumed rather than duplicated, and all
  public names follow their capability/error/export policy.
- [x] P1 provides concrete-only immutable resident dimensions, metadata,
  lifecycle vocabulary, and identity-retained Warp containers with O(1)
  metadata-only validation (issue #1484).
- [x] P1 construction has regression coverage for no payload access, transfer,
  synchronization, kernel launch, allocation, conversion, fallback, migration,
  lifecycle operation, or export change.
- [x] P2 setup validates the complete local CPU state before conversion and
  performs exactly one particle, one gas, and one environment upload in order
  (issue #1485).
- [x] P2 publishes only a complete validated `ACTIVE` session, retains converted
  containers by identity, and preserves ordered CPU gas names in metadata
  (issue #1485).
- [x] P4 provides direct-import-only identity-token timestep bookkeeping for one
  exact active session/registry binding; only matching completion advances
  guard-owned count/time state (issue #1487).
- [x] P4 rejects nested/mismatched lifecycle operations and uses
  `assert_step_closed()` as the explicit future-boundary gate without adapter
  execution, transfer, synchronization, allocation, resize, restore, or
  fallback (issue #1487).
- [x] P4's metadata-only `validate_pinned_session()` rejects session/lifecycle/
  primary identity drift without acquiring or mutating registry resources
  (issue #1487).
- [ ] A session owns same-device fixed-shape Warp particle, gas, and environment
  state plus validated reusable process resources.
- [ ] Container, array, and supplied sidecar identities and shapes remain stable
  across at least two successful timesteps.
- [ ] No bulk CPU-to-Warp or Warp-to-CPU conversion occurs during normal step
  lifecycle operations.
- [x] A nonterminal checkpoint performs one explicit synchronization and one
  restore per CPU container, preserves ordered metadata, and leaves the live
  session active (P5, issue #1488).
- [x] Finalization uses the same explicit boundary and makes the session
  terminal without a duplicate restore on repeated calls (P5, issue #1488).
- [x] Restart from a compatible checkpoint restores fresh same-device resident
  state, canonical vapor pressure, acquired sidecars, and lifecycle metadata
  (P5, issue #1488).
- [x] Explicit read-only failure cleanup preserves reusable `ACTIVE` session and
  payload identities; writer-may-have-launched failure releases its exact token,
  faults the session, preserves the original error and observable mutation, and
  never triggers hidden rollback or CPU fallback (P6, issue #1489).
- [x] Concrete close/discard terminally closes valid active/faulted bindings
  without runtime work; closed/finalized calls are write-free idempotent and P5
  finalized-checkpoint identity is retained (P6, issue #1489).
- [ ] E7-F6 availability seam: native-device availability remains an explicit
  upstream precondition for P2; unavailable-device rejection awaits E7-F6's
  public runtime API and is not emulated by this factory.
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
