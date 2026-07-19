# GPU Coagulation Validation Record

This maintainer and release-review record documents evidence for the existing
direct GPU-coagulation path. It does not introduce a production API, physics,
public export, runnable integration, CPU fallback, graph capture/replay,
autodiff, adaptive stepping, or a performance claim.

## Supported Evidence Matrix

The fixed-mask matrix has exactly 11 executable rows. P1 is deterministic
validation in
`particula/gpu/kernels/tests/coagulation_validation_test.py`, backed by the
Warp-free independent support equations in
`particula/gpu/kernels/tests/_coagulation_validation_support.py`. P2 public-
step invariant and ownership assertions are in
`particula/gpu/kernels/tests/coagulation_validation_test.py`, supported by
`particula/gpu/kernels/tests/_coagulation_public_step_support.py`. P3 is the
bounded stochastic evidence in
`particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py`.

| Mask | Canonical mechanism tuple | P1 deterministic | P2 invariants/ownership | P3 stochastic |
| --- | --- | --- | --- | --- |
| 1 | `("brownian",)` | P1 | P2 | P3 |
| 2 | `("charged_hard_sphere",)` | P1 | P2 | P3 |
| 4 | `("sedimentation_sp2016",)` | P1 | P2 | P3 |
| 8 | `("turbulent_shear_st1956",)` | P1 | P2 | P3 |
| 3 | `("brownian", "charged_hard_sphere")` | P1 | P2 | P3 |
| 5 | `("brownian", "sedimentation_sp2016")` | P1 | P2 | P3 |
| 6 | `("charged_hard_sphere", "sedimentation_sp2016")` | P1 | P2 | P3 |
| 9 | `("brownian", "turbulent_shear_st1956")` | P1 | P2 | P3 |
| 10 | `("charged_hard_sphere", "turbulent_shear_st1956")` | P1 | P2 | P3 |
| 12 | `("sedimentation_sp2016", "turbulent_shear_st1956")` | P1 | P2 | P3 |
| 15 | `("brownian", "charged_hard_sphere", "sedimentation_sp2016", "turbulent_shear_st1956")` | P1 | P2 | P3 |

## Deferred Masks

The three-way masks `7`, `11`, `13`, and `14` are deferred and fail closed.
They are not supported combinations, and this record makes no claim for any
other combination.

## Evidence Boundaries

- **P1 deterministic:** Brownian pair-rate comparisons use
  `rtol=1e-7, atol=0`. Brownian property and selector-majorant checks, along
  with other applicable positive or additive rate, property, and majorant
  comparisons, use `rtol=1e-6, atol=0`. Physical zeros are exact.
- **P2 invariants and ownership:** concentration-weighted, per-box,
  per-species inventory uses `rtol=1e-12, atol=1e-30`. The same public-step
  evidence covers applicable charge conservation, legal pairs, merge
  bookkeeping, inactive slots, caller-buffer identity, persistent RNG
  lifecycle, and atomic preflight failures.
- **P3 stochastic:** each executable row uses one-proposal capacity, 100
  unique fresh seeds, an independent initial-state expectation, and the
  `3 * sqrt(expected_mean)` bound. P3 is not conservation evidence and is not
  an exact accepted-pair, seed, RNG, CPU/Warp, or CUDA replay guarantee.

Warp CPU is the required baseline when Warp is installed. CUDA is optional,
local/manual additive evidence and skips cleanly when unavailable. The `warp`,
`cuda`, `gpu_parity`, and `stochastic` markers describe tests; selecting markers
does not select a device. P2 enumerates available Warp devices, so it includes
available CUDA even when P2 tests are not `cuda`-marked.

## Warning-Clean Reproduction Commands

```bash
pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror
```

Missing Warp or unavailable CUDA is valid only when an existing guarded test
skips. A skip is not CPU/Warp or CUDA parity evidence.
