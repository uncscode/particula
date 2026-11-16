# Adaptive integration and splitting assessment

The selected development methods are RK3(2) for the measured continuous
condensation regimes, CPU ROS2/ROS3 and a single-exchange GPU ROS3 option for stiff continuous systems,
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

## Cloud/droplet follow-through: sparse ROS2 and worked references

The following additional evidence qualifies the sparse CPU linear solve and
three controlled worked examples. Full cloud-chamber, respiratory-droplet,
biomass and coupled GPU migrations remain a separate unfinished requirement;
these small cases do not substitute for their complete workloads.

CPU ROS2 now accepts sparse Jacobians. For known particle-local native activity
and surface models above 64 coordinates, material-colored finite differences
retain each particle's composition block and weighted gas-reservoir coupling.
Unknown/custom models retain dense column differences. Physical mass
perturbations use current particle mass, including dry cores, rather than the
initial wet scaling. A regression reproduces the incorrect derivative after a
water-mass change from 1e-12 to 1e-22 kg and verifies invariance to a freshly
scaled owner at the same accepted state.

Seven sparse-specific tests cover independently perturbed columns, two boxes,
reordered gas/material maps, non-unit volumes, inactive slots, cross-material
kappa derivatives, a matrix-exponential stiff oracle, derivative-work scaling,
exact disk continuation, and a custom nonlocal activity whose cross-particle
derivatives must remain present. The combined ODE/native/sparse subset passed
26 tests before the custom-model case was added; the expanded sparse suite
then passed all seven tests. No tolerance or workload was weakened.

The complete package subsequently passed **10,690 tests**, with 19 skips,
57 slow/performance deselections and one expected failure, using `-Werror` and
actual CUDA. Coverage was **91.21%**, above the normal 80% gate. Standard Ruff
and formatting checks passed for 755 files; ordinary Mypy passed for 675 source
files. Checked-body Mypy also passed for the three changed numerical modules
with imported-module diagnostics suppressed. That additional local check does
not claim checked-body cleanliness for the full repository. Tooling and example
checks excluding the separately running native-cloud test module passed
**552 tests**, with one skip. The native-cloud module remains a required
outstanding check for the example migrations.

### Independent trajectory comparisons

All three examples save compact numerical histories and work counters. Full
refreshes and separate saved-data replots completed, and all figures were
visually inspected. Their manifests retain execution versions, device, dirty
source provenance and the common package-source SHA-256
`fcd3e6e2e8939a530ae4aa3aaf4bd708172dca7cd6696678c501e9bf37822b06`.
The CUDA download was extracted to a separate directory and executed with its
bundled helper on the RTX 5060. Replots ran on CPU without advancing physics.

| Example / profile | Maximum relative radius error | Other evidence |
| --- | ---: | --- |
| [Wet–dry relaxation](../../../examples/simulations/adaptive_cloud_relaxation.py), fixed 0.005 s | 3.70 | Artificial dry-phase oscillations despite mass closure below 1e-14 |
| Wet–dry ROS2, rtol 1e-5 | 6.05e-6 | 3,667 accepted, 2 rejected trials |
| Wet–dry ROS2, rtol 1e-7 | 6.98e-8 | 36,161 accepted, 4 rejected trials |
| [Finite reservoir](../../../examples/simulations/adaptive_reservoir_depletion.py), fixed 0.5 s | 1.31e-2 | Vapor error / initial vapor: 5.17e-2 |
| Finite reservoir, fixed 0.005 s | 1.22e-4 | Vapor error / initial vapor: 4.71e-4 |
| Finite reservoir, CPU RK23 rtol 1e-7 | 2.57e-8 | Vapor error: 9.15e-8; 430 accepted trials |
| Finite reservoir, CPU ROS2 rtol 1e-5 | 7.74e-7 | Vapor error: 2.86e-6; 2,574 accepted trials |
| [CUDA reservoir](../../../examples/simulations/adaptive_reservoir_depletion_cuda.py), RK23 rtol 1e-4 | 1.82e-5 | Vapor error: 6.74e-5; 62 accepted trials |
| CUDA reservoir, RK23 rtol 1e-7 | 2.57e-8 | Vapor error: 9.15e-8; 430 accepted trials |

Wet–dry reference uncertainty is cross-checked with independently coded scalar
water-volume equations solved by Radau and BDF. Finite-reservoir reference
uncertainty is cross-checked with a separate scalar mass-ratio equation solved
by DOP853 and Radau. The example sources contain the explicit equations,
constants, comparison assertions and observable targets. Both calculations
include Kelvin curvature and kinetic correction, and keep external resets
separate from closed budgets. They do not validate measured chamber material
properties, thermal coupling, stochastic removal, or stiff CUDA condensation.
The one-representative-particle CUDA calculation is accuracy evidence only.

### Higher-order stiff follow-through: ROS3

The full cloud migration exposed a practical cost limit in ROS2's first-order
embedded estimator. A bounded comparison therefore revisited the original
Stage 2A ROS3 candidate before selecting it for larger workloads. The concrete
CPU solver now accepts `method="ros3"`; RK23 remains the default and the GPU
owner remains RK23-only. The three-stage method shares one factorization and
uses two rate evaluations per trial, with partial time derivatives for forcing.
Coefficients were checked against the `Ros3` routine in
[KPP's primary implementation](https://github.com/KineticPreProcessor/KPP/blob/main/int/rosenbrock.f90),
file blob `baf1ef64538a5128db5e1154ac7fd1b39c41f665`.

The exact nonautonomous and nonlinear solutions in `ode_test.py` demonstrate
third-order convergence. Dense and sparse stiff transfer tests compare the
whole transient with matrix exponentials: maximum errors decrease from
8.69e-6 to 1.07e-7 to 1.13e-9 at relative tolerances 1e-4, 1e-6 and 1e-8.
Independent budgets remain closed. Additional checks cover domain rejection,
discontinuous/linear forcing, event-side observations and bitwise disk restart
with both dense and sparse native condensation coordinates.

The full wet–dry example was refreshed with both methods and unchanged physical
inputs, absolute tolerances and observation times. Radau/BDF independently
cross-check its reference. These are measured global errors, not promises
derived from the local tolerance:

| Method | Relative tolerance | Maximum relative radius error | Accepted / rejected |
| --- | ---: | ---: | ---: |
| ROS2 | 1e-5 | 6.04634e-6 | 3,667 / 2 |
| ROS3 | 1e-5 | 4.07160e-6 | 330 / 0 |
| ROS2 | 1e-7 | 6.98075e-8 | 36,161 / 4 |
| ROS3 | 1e-7 | 3.41392e-8 | 1,247 / 2 |

An additional eight-particle polydisperse screening case completed the same
2 s wet/3 s dry schedule with unchanged tolerances. Final masses agreed between
ROS2 and ROS3 to approximately 2e-13 relative. This endpoint agreement supports
trying ROS3 on the complete examples; it is neither an independent transient
oracle nor a replacement for those workloads. No full-cloud speedup claim is
based on this small screening case.

Focused checks passed 39 numerical/sparse/native/split-restart tests, 50 existing
workflow tests, and five additional ROS3 forcing/restart cases. The full package
then passed **10,703 tests**, 19 skips, 57 slow/performance deselections and one
expected failure, with actual CUDA and **91.21% coverage**. Standard Ruff/format,
Mypy (675 source files), and checked-body Mypy for the ODE module passed. The
558-page website built with no link errors. The updated comparison figure was
inspected and its compact saved data successfully replotted.

The full single-cycle chamber completed with both ROS2 and ROS3, retaining
1,000 particles over 60 s. Both report peak diameter 23.19 µm, 788 survivors and
180.6% final wet/initial dry mass at the displayed precision. Both inspected
plots show smooth dry relaxation. Elapsed refresh times were 1,869.4 s and
299.3 s respectively on the same machine with other jobs running. These are
end-to-end example timings, not an isolated solver benchmark; printed agreement
does not establish a bound on every trajectory coordinate.

Publication review found stale installed distribution metadata (0.2.14) while
the executed checkout version is 0.3.1, and a library hash sampled only after
execution. The publisher now records the checkout version and the distinct
installed distribution version, captures library identity before execution,
and refuses publication if that hash differs afterward. Twenty-two focused
publisher tests passed, including preservation of the prior bundle after a
mid-run library change. Fresh runs with these guards are pending for the full
chamber/droplet/biomass workloads. Their coupled GPU migration and the complete
native-cloud example test module remain outstanding.

The guarded full single-cycle repeat subsequently completed in 293.0 s with
the same printed peak and survivor count and source version 0.3.1. All 326
tooling tests passed. The complete native-example module then exposed three
dry-inventory failures despite passing the broader package suite. Two causes
were reproduced independently: pivoted linear solves contaminated a zero-rate
coordinate, and scaling/unscaling an unchanged mass changed its last bit.

The production solver now applies two residual corrections with the existing
LU factors at each Rosenbrock stage. All correction solves are included in the
work counters. The condensation adapter preserves the original physical value
only when its scaled coordinate is exactly unchanged; there is no tolerance,
clipping, or small-rate suppression. The caches reconstruct from the original
checkpoint base, preserving the existing restart representation. A separate
three-coordinate linear system tests passive inventory, an exponential-matrix
reference and weighted conservation with dense and sparse ROS2/ROS3 solves.

After both corrections, the **unmodified full native-example test module passed
all 13 tests in 313.76 s**, including three compositions, both output schedules,
and shifted absolute times. The focused numerical/restart/forcing command passed
73 tests with four CUDA skips in its restricted environment. The subsequent
full source-CI package selection ran with actual CUDA: **10,710 passed, 19
skipped, 57 slow/performance deselected and one expected failure** in 193.85 s.
The remaining example/tooling suites passed **554 tests with one skip** in
115.90 s. Ruff/format and ordinary Mypy passed; checked-body Mypy passed for both
modified numerical modules. The full unfiltered coverage run passed **10,766 tests**, with 20 skips and one
expected failure, in 276.26 s; coverage is **91.41%**, above the 80% gate.
Logs are retained under `.artifacts/ros3-refined-*`; the complete native run is
`.artifacts/cloud-native-refined-production.log`.

The three live multi-cycle, respiratory and biomass publishers using the earlier
solver were deliberately stopped before changing runtime source. Their outputs
are not qualification of the corrected solver. All four full workloads and the
CPU comparison examples require fresh guarded publication from a fixed source
revision; earlier successful bundles remain historical evidence.

An isolated resident ROS3 feasibility prototype now executes physical GPU rates,
colored derivatives and conservative arrowhead solves without downloading the
trial state. An independent inverse-radius quadrature, with two boxes and
non-unit volumes, gives CUDA relative radius errors 1.30e-6, 1.40e-8 and 1.40e-10
at rtol 1e-4, 1e-6 and 1e-8. Weighted budgets, inert gas and inactive slots pass.
The first prototype exposed float32 rounding of kernel coefficient literals;
explicit module-level float64 constants restored refinement. This is prototype
evidence only: forcing, restart, integration into the runtime, coupled event
ownership, and the complete GPU chamber/forcing workloads remain unqualified.

The corrected production source was committed as `12742e1c394c7a836d31aff3aeecbd8c0ff2b523`.
A detached validation checkout pins that revision while full examples execute;
ongoing GPU development cannot change their imported source. The first CPU
workflow, finite-reservoir comparison and wet–dry comparison completed guarded
refreshes from the clean revision, with package hash
`6d2e24246ad67da3f2dd23afebccde41af3e1ed32de51992c99ef898b3f88d13`.
Their bundle hashes and matching source were verified before import, figures
inspected, and saved data replotted. Corrected wet–dry ROS3 maximum radius
errors are 4.07786e-6 and 3.39075e-8, still requiring 330/1,247 accepted steps.
The corrected full single-cycle chamber subsequently completed in 304.95 s,
retaining 1,000 particles over 60 s. Its inspected plot shows smooth dry
relaxation, with peak diameter 23.19 µm, 788 survivors and 180.6% final
wet/initial dry mass at printed precision. Its source and bundle hashes were
verified before importing the clean-revision result. The multi-cycle,
respiratory and biomass workloads remain running at this checkpoint.
The current 558-page website build and rendered structure check both pass
with zero structural errors.

The isolated GPU prototype also completed the full 81-observation wet–dry
reference case on Warp CPU and actual CUDA. On CUDA, rtol 1e-5/1e-7 gives
maximum radius errors 4.06940e-6/3.24342e-8, with 330/1,251 accepted and 0/2
rejected trials. Closed-phase relative mass residuals stay below 4e-15.
Independent Radau/BDF agreement and the explicit humidity-reset accounting are
retained. This remains prototype evidence, not coupled GPU qualification.

## Integrated GPU ROS3 qualification

The resident ROS3 method is now integrated into `GPUCondensationODE`, including
stage-time forcing and exact same-device disk restart. The supported profile
has exactly one exchanging material; unsupported multiple exchanges reject
before device allocation. The stage solve uses conservative arrowhead
elimination, with linear scratch storage and no dense factorization. This
supersedes the prototype-only status above for the isolated condensation owner;
coupled source/collision ownership remains unfinished.

Independent dense-matrix tests cover scaled, reversed species maps, non-unit
volumes, discrete/resolved measures and inactive slots on Warp CPU and CUDA.
Inverse-radius quadrature verifies refinement. Exact passive-coordinate tests
retain dry masses and inert gases bit for bit. Existing forcing, rejection,
capture prohibition and restart tests now exercise ROS3 as well as RK23.
Adaptive GPU checkpoint schema 3 rejects earlier numerical semantics rather
than silently reinterpreting their scaled state.

The full [CUDA wet–dry example](../../../examples/simulations/adaptive_cloud_relaxation_cuda.py)
completed its 81-observation case on the RTX 5060. Relative tolerances 1e-5 and
1e-7 give maximum radius errors 4.069403e-6 and 3.243418e-8 against independent
Radau/BDF references. Accepted/rejected counts are 330/0 and 1,251/2; maximum
closed-phase budget residuals are 7.09e-16 and 3.72e-15. The explicit humidity
reset is accounted for separately. Autonomous trials use six physical rate
calls, three linear solves and nineteen scalar readbacks, including derivative
and domain checks. A shared host controller remains necessary; this is neither
whole-loop CUDA capture nor a GPU acceleration claim.

The guarded publication took 7.72 s and retained package source hash
`d3dbf2bbc42010dc580e48635a3056b40dfcfede7b20b6405fae6cc8921002f6`.
The figure was inspected, saved data replotted on CPU, and the extracted ZIP
executed on CUDA using its bundled reference helper. The CUDA reservoir example
also completed a fresh guarded publication and CPU replot.

Validation of this integration passed **10,788 package tests**, with 20 skips
and one expected failure, in 305.19 s; unfiltered coverage was **91.30%**.
Tooling and examples excluding the separately qualified full native-cloud
module passed **554 tests**, with one skip, in 124.75 s. The combined GPU
numerics/restart/forcing subset passed 75 tests. Four later-added parameter
cases extend the transfer/capture checks to ROS3; their focused selection passed
all eight cases. Ruff/format and ordinary Mypy passed; checked-body Mypy passed
for the four changed GPU modules. The 561-page website built successfully,
with zero rendered structure errors. Logs are in `.artifacts/ros3-gpu-*` and
`.artifacts/ros3-cuda-*`.

The full coupled GPU cyclic chamber and forcing workloads are still required.
These isolated-method results do not qualify the original fixed stochastic
event schedule under adaptive deterministic condensation.

Validation commands (from the Stage 2B checkout with `PYTHONPATH=.`):

```bash
.venv/bin/pytest particula/execution/tests/ode_test.py particula/execution/tests/condensation_sparse_test.py particula/execution/tests/cpu_restart_test.py particula/execution/tests/forcing_test.py -q -Werror --no-cov
.venv/bin/pytest examples_tests/native_simulation_examples_test.py -q -Werror --no-cov
.venv/bin/pytest -q -Werror -m 'not slow and not performance'
.venv/bin/pytest --cov=particula --cov-report=term-missing --cov-fail-under=80 -q -Werror
.venv/bin/pytest examples_tests scripts/tests -q -Werror --no-cov --ignore=examples_tests/native_simulation_examples_test.py
.venv/bin/ruff check particula/ examples_tests/ scripts/ --no-fix
.venv/bin/ruff format particula/ examples_tests/ scripts/ --check
.venv/bin/mypy particula/ --ignore-missing-imports
.venv/bin/mypy particula/execution/ode.py particula/execution/condensation_ode.py --ignore-missing-imports --check-untyped-defs --follow-imports=silent
```
