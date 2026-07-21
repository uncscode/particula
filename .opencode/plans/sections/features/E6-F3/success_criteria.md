# Success Criteria

- [ ] `wall_loss_step_gpu` supports neutral spherical and rectangular,
  particle-resolved execution on Warp CPU and is importable from
  `particula.gpu.kernels`.
- [ ] Device coefficients match independent CPU spherical/rectangular references
  at recorded fp64 `rtol`/`atol` across the accepted state matrix.
- [ ] Observed survivor counts satisfy predeclared statistical bounds for
  `p = exp(-k * time_step)`; acceptance does not require CPU/GPU RNG identity.
- [ ] `time_step == 0` and all-inactive inputs are exact particle no-ops; the
  documented RNG behavior for those no-op paths is explicit and tested.
- [ ] Every removed slot has all species masses, concentration, and charge set
  exactly to zero; no inactive slot is sampled or reactivated.
- [ ] Surviving slot values, particle density and volume, fixed shapes, devices,
  dtypes, object identities, and array identities remain unchanged.
- [ ] Supplied `(n_boxes,)` `wp.uint32` RNG sidecars retain identity, advance
  across repeated successful calls without hidden reseeding, and reset only on
  explicit initialization.
- [ ] All contractually detectable invalid configuration, environment, state,
  time, shape, dtype, and device inputs fail before allocation/RNG initialization
  where promised and before any caller-owned particle or RNG mutation.
- [ ] No hidden CPU transfer/fallback, dynamic resize, charged behavior,
  high-level runnable, scheduler/backend selection, graph-capture, or performance
  claim is introduced.
- [ ] Warp CPU parity is required; the same CUDA matrix passes when available or
  skips cleanly when unavailable. Existing CPU wall-loss and GPU RNG/slot tests
  remain green, and configured coverage remains at least 80%.
- [ ] Documentation states units, equations/citations, RNG ownership, slot
  invariants, support/deferred boundaries, and focused reproduction commands.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Supported neutral GPU geometries | 0 | 2: spherical and rectangular | API/config tests |
| Deterministic coefficient parity | No GPU path | All matrix cases within recorded fp64 tolerances | `wall_loss_funcs_test.py`, parity tests |
| Statistical survival agreement | No GPU path | Every case inside predeclared binomial confidence/sigma bound | `wall_loss_parity_test.py` |
| Removed slots completely cleared | No GPU path | 100% of mass components, concentration, and charge exactly zero | Kernel invariant tests |
| Invalid-call caller mutations | N/A | 0 across particle and RNG snapshots | Preflight tests |
| Hidden per-step RNG reseeds with supplied state | N/A | 0 unless `initialize_rng=True` | Repeated-step RNG tests |
| Required GPU backend evidence | N/A | Warp CPU 100%; CUDA optional clean skip | `warp_devices()` matrix |
