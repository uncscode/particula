# Success Criteria

- [ ] E6-F3 is recorded as the upstream dependency and its neutral API,
  preflight, fixed-slot, identity, and RNG contracts remain passing.
- [x] P1 accepts only neutral/charged modes; validates signed finite potential,
  charged spherical scalar fields, and charged rectangular caller-owned
  same-device `wp.float64` `(3,)` fields without changing package exports.
- [x] P1 rejects malformed charged configurations before particle value scans,
  environment/RNG work, allocation, launch, or caller mutation; rectangular
  field errors precede coexisting charge errors.
- [x] P1 valid charged calls preserve rectangular field identity and bytes and
  execute the existing neutral coefficient/removal/RNG path.
- [ ] Nonzero particle charge receives CPU-equivalent image-charge enhancement
  even when wall potential is exactly zero.
- [ ] Explicit field magnitude and potential-derived field preserve CPU
  geometry scaling and signed charge drift semantics.
- [ ] Every zero-charge active slot reproduces the E6-F3 neutral coefficient and
  survival probability, including when potential or field is configured.
- [ ] Charged spherical and rectangular coefficients match the CPU array oracle
  at recorded fp64 `rtol`/`atol` on required Warp CPU.
- [ ] Charged and neutral-fallback survival counts pass predeclared statistical
  bounds without requiring exact CPU/Warp RNG sequences.
- [ ] Supplied per-box RNG persists by identity without hidden reseeding;
  explicit reset is reproducible and invalid/no-op calls do not advance it.
- [ ] Removed slots clear every species mass, concentration, and charge;
  survivors preserve all fields and all shapes/devices/dtypes/identities remain
  stable.
- [ ] All detectable invalid calls fail before allocation, RNG mutation, or
  particle mutation.
- [ ] CUDA validation skips cleanly when unavailable, and documentation states
  all explicit ownership and deferred boundaries.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Charged deterministic CPU/Warp cases | 0 direct GPU cases | 100% within recorded fp64 tolerances | `wall_loss_parity_test.py` |
| Zero-charge fallback cases | E6-F3 neutral only | 100% match E6-F3 neutral result | Device/kernel tests |
| Statistical survival scenarios | 0 charged GPU scenarios | 100% pass predeclared confidence bounds | `wall_loss_parity_test.py` |
| Invalid-call mutation incidents | N/A | 0 particle writes and 0 RNG advances | Preflight snapshots |
| Removed-slot clearing fields | E6-F3 mass/concentration/charge | 100% retained in charged mode | Kernel tests |
| Required backend | Warp CPU | Warp CPU passes; CUDA optional | Backend test matrix |
