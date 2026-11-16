# Adaptive integration and splitting assessment

The selected development methods are RK3(2) for the measured continuous
condensation regimes, an explicit CPU ROS2 option for stiff continuous systems,
and sequential/Strang composition of declared deterministic flows. Existing
finite-step strategies and fixed-four GPU condensation remain available.
This record establishes a bounded numerical choice, not experimental validation
or a general GPU speedup. See the [usage and support matrix](../../../guide/processes/numerical-integration.md).

## Evidence and reproducibility

The [preliminary assessment data](https://github.com/uncscode/particula/blob/main/docs/data/numerics/stage-02a-preliminary.json)
retain 60 generic solver runs, 66 native comparisons, 15 splitting refinements,
18 conservative-transfer runs and four CUDA scalar-ensemble runs. Their original
source hashes identify the executed scripts; later source changes do not relabel
those historical results.

The [native adaptive assessment](https://github.com/uncscode/particula/blob/main/docs/data/numerics/native-adaptive.json)
was completed on 2026-11-16 UTC, on an NVIDIA GeForce RTX 5060, sm_120,
Warp 1.11.1, Python 3.12.12, NumPy 2.3.4 and SciPy 1.16.2. CUDA driver 13.4
and toolkit 12.9 were reported by Warp. It includes 27 baseline runs, eight
additional target checks and shared/independent 32-box comparisons. All completed;
completion alone does not mean an accuracy target passed.

The implementation base is `0d21d18b734ac840dfaf2c1f360db8daec780c80` plus the
source additions identified by the report's SHA-256 map. Every recorded source
hash was compared with the executable file before retaining the report.
Reproduce from the repository root with the selected environment:

```bash
PYTHONPATH=. .venv/bin/python scripts/assess_native_adaptive.py \
  --device cuda:0 --boxes 32 \
  --output .artifacts/numerics/native-adaptive-qualified.json
```

The script requires actual CUDA. It separates setup, accepted-step work
(including rejections and scalar readbacks), and full observation time. Timings
are one warmed sample, not a distribution, matched-hardware throughput campaign,
or profiler measurement. Array byte counts exclude context, compiler,
registers/spills and transient peak allocation.

## Cases, independent references and weights

The native case begins with 1 micrometre radius, density 1000 kg/m3, number
density 1e9 m^-3, vapor 1e-5 kg/m3, diffusivity 2e-5 m2/s, accommodation one,
molar mass 0.018 kg/mol, temperature 298.15 K and pressure 101325 Pa. Resolved
volume is non-unit; the case contains an inert gas and an inactive slot.
Surface tension is 0.072 N/m and the Kelvin correction is retained. The new
assessment uses Sutherland viscosity consistently in all three methods and
references; the earlier native comparison fixed viscosity at 1.8e-5 Pa s.
Do not use timings from these two studies as an equal-physics speed ratio.

Smooth growth runs to 1.4 s. Equilibrium and near-depletion runs reach 20 s,
with flat-interface equilibrium vapor of 1e-6 and zero kg/m3 respectively.
The independent radius equation uses the closed inventory to eliminate gas.
DOP853 and Radau at tight tolerances cross-check the reference; inverse-radius
quadrature independently checks both the initial and noninitial local flow.
Maximum baseline radius-reference disagreement is below 2.5e-12, well below
1% of the tightest declared global target, 1e-7.

Radius error is normalized by initial radius. Vapor error is normalized by its
own initial concentration, not by the particle-plus-gas reservoir. The global
metrics cover accepted trajectory points, rather than endpoints alone. Absolute
tolerances in the assessment are initial particle mass or initial vapor times
`rtol * 1e-3`; these are part of the result, not universal default values.

For up to 16 accepted steps per run, a separate local reference starts from the
numerical pre-step state. DOP853/Radau uncertainty is retained in the same error
weights as the true local error. This distinguishes a bad local estimate from
accumulated global error. It samples accepted steps and is not an exhaustive
bound over every possible stage or physical regime.

## Accuracy and estimator limits

Representative RK23 results below apply to both CPU and CUDA within the shown
precision. The report preserves their small numerical differences.

| Case | Local rtol | Maximum radius error | Maximum vapor error / initial vapor |
| --- | ---: | ---: | ---: |
| Smooth growth | 1e-3 | 2.51e-6 | 3.82e-6 |
| Smooth growth | 1e-5 | 1.09e-7 | 1.75e-7 |
| Smooth growth | 1e-7 | 1.36e-9 | 2.19e-9 |
| Equilibrium | 1e-3 | 6.61e-4 | 1.67e-3 |
| Equilibrium | 1e-5 | 3.60e-6 | 8.34e-6 |
| Equilibrium | 1e-7 | 4.74e-8 | 1.03e-7 |
| Near depletion | 1e-3 | 2.91e-3 | 7.52e-3 |
| Near depletion | 1e-5 | 3.06e-6 | 7.36e-6 |
| Near depletion | 1e-7 | 3.96e-8 | 9.03e-8 |

The loose equilibrium and depletion settings fail the corresponding 1e-3
global vapor target. The sampled true local weighted error reaches about 4.65
and 22.7 respectively even though the embedded estimate admitted the steps.
Domain rejection protects physical inventory, but does not establish estimator
reliability. This is why positivity and conservation are separate from accuracy.
At `rtol=1e-5`, sampled true local weighted errors are below 0.36 for these
cases; at `1e-7` they are below 0.18. This is empirical case evidence, not a
new acceptance rule or a global guarantee.

Additional checks establish these settings for **both** radius and vapor in
the two 20 s cases:

| Global target | Qualified local rtol | Largest measured error across both cases/backends |
| --- | ---: | ---: |
| 1e-3 | 1e-4 | 1.34e-4 |
| 1e-5 | 1e-5 | 8.34e-6 |
| 1e-7 | 1e-8 | 1.19e-8 |

The absolute tolerances above must accompany these settings. Near-depletion
accuracy also has an independent inverse-quadrature regression at radius ratio
1.49, with less than 4% of the initial vapor remaining. No result here covers
complete active-slot evaporation, discontinuous forcing without boundaries,
or arbitrary high-stiffness chemistry.

All baseline inventories remain nonnegative. Maximum relative closed-budget
residual is below 1.1e-12 across methods, including the long ROS2 trajectories.
Physical domain failures trigger shorter trials; the adaptive owners do not
apply inventory clipping. The preliminary finite-step study separately counts
full-step limiting proposals and retains failed external-solver trials.

## Work at a common observable accuracy

At `rtol=1e-5` all three native methods meet the 1e-5 radius and vapor targets in
these cases. Work and measured step-only seconds are:

| Case | Method | Accepted / rejected | Physical rate evaluations | Step seconds |
| --- | --- | ---: | ---: | ---: |
| Smooth growth | CPU RK23 | 8 / 0 | 32 | 0.0093 |
| Smooth growth | CPU ROS2 | 114 / 1 | 805 | 0.239 |
| Smooth growth | CUDA RK23 | 8 / 0 | 32 | 0.0050 |
| Equilibrium | CPU RK23 | 56 / 2 | 232 | 0.0639 |
| Equilibrium | CPU ROS2 | 1257 / 1 | 8806 | 2.46 |
| Equilibrium | CUDA RK23 | 56 / 2 | 232 | 0.0351 |
| Near depletion | CPU RK23 | 90 / 2 | 368 | 0.103 |
| Near depletion | CPU ROS2 | 2433 / 1 | 17038 | 4.66 |
| Near depletion | CUDA RK23 | 90 / 2 | 368 | 0.0815 |

ROS2 physical-rate counts include its numerical Jacobian evaluations. The report
also records Jacobian/time-derivative calls, factorizations and linear solves;
ordinary controller rate-call counters alone omit this extra work. CUDA RK23
retains 648 bytes of counted device arrays for this tiny case. Setup and
per-accepted-step observation costs are reported separately, so the table does
not claim end-to-end acceleration.

For 32 boxes with vapor spanning a factor of 100 and volumes from 0.1 to
0.9 m3, a shared maximum-error controller took 17 accepted steps and 153 scalar
readbacks. It retained 14,784 array bytes and took 0.0100 s to step. Sequential
independent CUDA owners needed 5--17 accepted steps each, 2169 scalar readbacks,
20,736 array bytes and 0.142 s. Both agree with a tighter CPU reference within
2.37e-7 in active particle mass and 2.56e-8 in vapor.

This comparison favors shared host control for the measured native ensemble.
It does not measure warp divergence in a native per-box device controller. The
separate scalar CUDA prototype does: at 4096 boxes, accepted work ranges from
104--1028 for RK23 and 7450--34456 for ROS2, with endpoint errors below 4.3e-8.
Its warmed kernel times are 0.0106 and 0.316 s, excluding uploads/downloads and
compilation. Unequal work illustrates divergence pressure; it is not a measured
hardware occupancy statistic or native aerosol throughput result.

## Method decisions

| Candidate | Decision and evidence |
| --- | --- |
| Existing finite-step isothermal/staggered methods | Retain explicit compatibility choices. Native refinements expose trajectory error despite good budgets; fixed substeps are not adaptive accuracy control. |
| RK3(2) | Select for the supported CPU/CUDA continuous condensation profile. Four rates, resident trial buffers on GPU, clear domain rejection and qualified observable targets. |
| RK5(4) | Retain as an assessment comparator, not another production default. Preliminary results do not demonstrate a need for a second explicit production method in the chosen accuracy range. |
| ROS2 | Retain explicit CPU stiff branch and rate/Jacobian contract. Native cases are accurate but substantially more expensive; no native GPU ROS2 claim. |
| Radau/BDF | Independent stiff/reference solvers in the assessment. Robertson screening exposes the limitations of explicit candidates; these references do not qualify a production chemistry model. |
| MPRK22 | Retain research prototype only. Linear zero-donor limits conserve mass and stay nonnegative; smooth/depletion orders approach two, but stiff transient error remains 0.0591 at 256 steps. Positivity is insufficient for selection. |
| SSPRK | Defer until an owning process supplies a demonstrated forward-Euler positivity restriction and a need beyond the selected error-controlled methods. No SSP production or measured-cost claim. |
| Spectral deferred correction | Defer: the selected case meets the declared targets with simpler methods, with no measured unresolved accuracy gap justifying additional quadrature/sweep storage. No SDC benchmark claim. |

These are engineering decisions inferred from the retained measurements and
stated support boundary. The references describe method families, not Particula
qualification: [SciPy integration methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
[KPP ROS2 coefficients](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html),
[modified Patankar order conditions](https://arxiv.org/abs/1702.04589), and
[spectral deferred correction convergence](https://arxiv.org/abs/1706.06245).

## Splitting and failure qualification

Independent noncommuting matrix flows give finest-grid observed orders 0.9994
and 1.0004 for the two sequential orderings and 1.9998 for Strang. Production
scheduler tests separately verify first/second order, nested groups, component
absolute times, and whole-cycle rejection. Native condensation plus exact
dilution has a separate refinement and species-budget test. None of these
results extends second-order claims to arbitrary equilibrium projections or
stochastic events.

The focused core/native/GPU/splitting/existing-condensation regression command
passed 128 tests with `-Werror`. After adding near-depletion and local-oracle
checks, the affected GPU and assessment subset passed 26 tests. Actual Warp CPU
and CUDA ran; capture rejection was tested with native CUDA capture. Failures,
caller detachment, inactive slots, non-unit volumes, mapping, kappa/Buck physics,
forcing stage times and scalar-only ordinary-step downloads are covered.

Coverage of the five new numerical modules is 85% aggregate (normal 80% gate).
CPU/GPU owners and core are 93%, splitting 91%; Python line coverage of the Warp
algebra is 21% because compiled kernel bodies execute outside Python tracing.
Actual kernel behavior is checked on both devices against independent oracles.
This focused evidence does not replace the full repository validation and
cross-workflow restart gates.

The broader standard package suite subsequently passed **10574 tests**, with
19 skips, 57 nonstandard workload deselections and one expected failure; full
package coverage was **91%**. Seven changed runtime modules together reached
**86%**. Standard full-repository Ruff, format and Mypy checks passed, as did
all **324** tooling tests. All three usage-guide snippets executed, including
CUDA. The website build and rendered-link verification passed. Cross-workflow
forcing and restart integration remains a separate required gate.
