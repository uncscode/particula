# Success Criteria

- [x] E5-F1/F2 records an explicit approved charged-model subset; the approved
  hard-sphere model has an internal scalar fp64 Warp pair helper and excluded
  models remain deferred.
- [x] Approved charged pair rates meet documented deterministic CPU/Warp
  tolerances for neutral, same-sign, opposite-sign, and mixed-scale fixtures.
- [x] Supported pair-helper inputs produce finite, symmetric, non-negative rates
  and preserve the neutral limit without divide-by-zero or overflow.
- [x] Charge shape, dtype, device, NaN, and infinity failures occur before any
  particle/output-buffer mutation or persistent RNG advancement.
- [x] Every accepted merge adds donor charge to recipient charge and clears
  donor mass, concentration, and charge; zero-collision calls are no-ops.
- [x] Per-box species mass and total charge are separately conserved in
  one-box, multi-box, multi-species, zero-charge, and mixed-sign fixtures.
- [x] Existing Brownian behavior, return tuple, output-buffer identity, and
  persistent RNG tests remain compatible.
- [x] Warp CPU evidence passes when Warp is installed; optional CUDA either
  passes the same deterministic checks or skips cleanly.
- [x] Documentation describes only pair-helper/merge support and identifies
  E5-F3 as the owner of charged stochastic execution and majorants.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| GPU charged pair models with parity evidence | 0 | 100% of approved subset | `coagulation_funcs_test.py` |
| Supported parity fixtures producing non-finite/negative rate | Not covered | 0 | Pair-helper tests |
| Charge validation failures occurring after mutation/RNG advance | Charge not validated | 0 | State-preservation tests |
| Accepted merges conserving per-box species mass | Mass-only baseline | 100% | Kernel/step tests |
| Accepted merges conserving per-box total charge | Donor charge retained | 100% | Kernel/step tests |
| Donor charge after an accepted merge | Unchanged | Exactly `0.0` | Direct merge tests |
| Existing Brownian focused regressions passing | Current suite | 100% | GPU coagulation tests |
