# GPU Coagulation Validation Record

This maintainer and release-review record documents evidence for the existing
direct GPU-coagulation path. It does not introduce a production API, physics,
public export, runnable integration, CPU fallback, graph capture/replay,
autodiff, adaptive stepping, or a performance claim.

## Supported Evidence Matrix

The fixed-mask matrix has exactly 11 executable rows. Input validation is deterministic
validation in
`particula/gpu/kernels/tests/coagulation_validation_test.py`, backed by the
Warp-free independent support equations in
`particula/gpu/kernels/tests/_coagulation_validation_support.py`. numerical validation public-
step invariant and ownership assertions are in
`particula/gpu/kernels/tests/coagulation_validation_test.py`, supported by
`particula/gpu/kernels/tests/_coagulation_public_step_support.py`. integration validation is the
bounded stochastic evidence in
`particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py`.

| Mask | Canonical mechanism tuple | input validation deterministic | numerical validation invariants/ownership | integration validation stochastic |
| --- | --- | --- | --- | --- |
| 1 | `("brownian",)` | input validation | numerical validation | integration validation |
| 2 | `("charged_hard_sphere",)` | input validation | numerical validation | integration validation |
| 4 | `("sedimentation_sp2016",)` | input validation | numerical validation | integration validation |
| 8 | `("turbulent_shear_st1956",)` | input validation | numerical validation | integration validation |
| 3 | `("brownian", "charged_hard_sphere")` | input validation | numerical validation | integration validation |
| 5 | `("brownian", "sedimentation_sp2016")` | input validation | numerical validation | integration validation |
| 6 | `("charged_hard_sphere", "sedimentation_sp2016")` | input validation | numerical validation | integration validation |
| 9 | `("brownian", "turbulent_shear_st1956")` | input validation | numerical validation | integration validation |
| 10 | `("charged_hard_sphere", "turbulent_shear_st1956")` | input validation | numerical validation | integration validation |
| 12 | `("sedimentation_sp2016", "turbulent_shear_st1956")` | input validation | numerical validation | integration validation |
| 15 | `("brownian", "charged_hard_sphere", "sedimentation_sp2016", "turbulent_shear_st1956")` | input validation | numerical validation | integration validation |

## Deferred Masks

The three-way masks `7`, `11`, `13`, and `14` are deferred and fail closed.
They are not supported combinations, and this record makes no claim for any
other combination.

## Evidence Boundaries

- **input validation deterministic:** Brownian pair-rate comparisons use
  `rtol=1e-7, atol=0`. Brownian property and selector-majorant checks, along
  with other applicable positive or additive rate, property, and majorant
  comparisons, use `rtol=1e-6, atol=0`. Physical zeros are exact.
- **numerical validation invariants and ownership:** concentration-weighted, per-box,
  per-species inventory uses `rtol=1e-12, atol=1e-30`. The same public-step
  evidence covers applicable charge conservation, legal pairs, merge
  bookkeeping, inactive slots, caller-buffer identity, persistent RNG
  lifecycle, and atomic preflight failures.
- **integration validation stochastic:** each executable row uses one-proposal capacity, 100
  unique fresh seeds, an independent initial-state expectation, and the
  `3 * sqrt(expected_mean)` bound. integration validation is not conservation evidence and is not
  an exact accepted-pair, seed, RNG, CPU/Warp, or CUDA replay guarantee.

Warp CPU is the required baseline when Warp is installed. CUDA is optional,
local/manual additive evidence and skips cleanly when unavailable. The `warp`,
`cuda`, `gpu_parity`, and `stochastic` markers describe tests; selecting markers
does not select a device. numerical validation enumerates available Warp devices, so it includes
available CUDA even when numerical validation tests are not `cuda`-marked.

## Warning-Clean Reproduction Commands

```bash
pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror
```

Missing Warp or unavailable CUDA is valid only when an existing guarded test
skips. A skip is not CPU/Warp or CUDA parity evidence.

## Historical performance context

[Recorded coagulation benchmark evidence](coagulation-benchmarks.md) describes
a dated, machine-specific tradeoff. It is separate from the physics validation
above and from current resident/capture qualification.

## Adaptive outer intervals on native CUDA

Two synthetic, unit-weight cloud populations reached 600 s in six 100 s
observation intervals. Both used seed 17, 283.15 K, 101325 Pa, combined
Brownian, SP2016 settling and ST1956 shear, dissipation 10⁻⁴ m² s⁻³ and
fluid density 1.2 kg m⁻³. Total concentration was 2,000 cm⁻³, with 10% of
the slots assigned to droplets. Each run exhausted a one-pass limit,
checkpointed the committed state, and resumed with a 128-pass ceiling. Each
accepted pass was limited to 256 scheduled trials per box. These two
resolutions exercise the issue's mechanisms and trajectory length. The
original campaign comprised 300 cases across a wider concentration range.
[Worked CUDA example and saved results](../../../examples/gpu/prepared/adaptive_coagulation.py)

| Initial slots | Accepted passes in successive 100 s intervals | Mergers | Live + rollback bytes | Synchronous step time |
| ---: | --- | ---: | ---: | ---: |
| 5,000 | 8, 9, 10, 10, 10, 10 | 184 | 503,876 + 503,876 | 0.86–1.06 s |
| 10,000 | 16, 18, 18, 19, 22, 22 | 433 | 964,516 + 964,516 | 6.33–8.35 s |

Each final active count equalled initial count minus accepted mergers.
Survivors retained unit weight, charge remained zero and both relative
mass drifts rounded to zero in fp64. Times include the restore-point copies,
graph replay and status readback. An additional status check immediately
after each committed step took 92–166 µs on the idle stream. This measures
host observation overhead separately from time waiting for pending physics.
The run used an RTX 5060, Warp 1.11.1, Python 3.12.12 and NumPy 2.3.4.

A separate two-box, three-material fixture isolates evolving disjoint-pair
capacity. At seed zero, a fixed 0.1 s step passes in both boxes. After the
first merger in box zero, five active particles permit only two disjoint
trials. The next interval requests three trials. The adaptive controller
subdivides the interval using the updated majorant.

The focused tests compare complete particle, field, collision, RNG and
checkpoint state after a rejected composed step. They check ordinary versus
captured continuation, ordered direct collisions against manual replay,
independent work limits, ULP boundaries, nonfinite demand and cohort
accounting. A separate Warp CPU comparison uses 32 independent seeds per
arm over 0.2 s. The adaptive 0.1 s and fixed 0.0125 s arms compare merger
counts and mean squared active radius against four pooled standard errors.
Passing that test establishes agreement for those two statistics and inputs.
It does not establish general timestep convergence.

```bash
python -m pytest particula/execution/tests/coagulation_subcycling_test.py -q -Werror
python scripts/docs_examples.py refresh adaptive-coagulation
```
