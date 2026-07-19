# Documentation Updates

## Status

**P4 shipped/completed in #1365.**

The [GPU coagulation validation record](../../../../../docs/Features/Roadmap/coagulation-validation.md)
and `.opencode/guides/testing_guide.md` publish the existing direct-path
validation contract; neither adds a production API, physics, runnable, CPU
fallback, or performance claim.

- **P1 deterministic evidence:**
  `particula/gpu/kernels/tests/coagulation_validation_test.py`, with independent
  support in `_coagulation_validation_support.py`. Brownian uses
  `rtol=1e-7, atol=0`; other positive/additive rate, property, and majorant
  comparisons use `rtol=1e-6, atol=0`; physical zeros are exact.
- **P2 invariant/ownership evidence:**
  `_coagulation_public_step_support.py` covers concentration-weighted
  per-box/per-species inventory at `rtol=1e-12, atol=1e-30`, applicable charge
  conservation, merge/inactive-slot bookkeeping, caller-buffer identity,
  persistent RNG lifecycle, and atomic preflight failures.
- **P3 bounded stochastic evidence:**
  `coagulation_stochastic_validation_test.py` uses one-proposal capacity, 100
  unique fresh seeds, an independent initial-state expectation, and
  `3 * sqrt(expected_mean)`. It is not conservation or exact replay evidence.

The executable masks are exactly `1`, `2`, `3`, `4`, `5`, `6`, `8`, `9`, `10`,
`12`, and `15`; three-way masks `7`, `11`, `13`, and `14` are deferred/fail
closed. Published warning-clean commands cover deterministic P1, CPU-only P3,
optional CUDA P3, and full `coagulation_test.py` regression. Warp CPU is the
baseline when Warp is installed. CUDA is optional local/manual additive
evidence and cleanly skips when unavailable; marker selection is not device
selection, and P2 enumerates available Warp devices.
