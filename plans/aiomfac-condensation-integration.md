# Milestone 6 — Direct AIOMFAC condensation in a closed box and a rising parcel

The two neutral vapors are water and 1,4-butanediol. A dissolved salt and a
nonvolatile organic influence both rates. Particula 0.3.4 and the optional
GPL backend 0.1.3 run this chemistry on CPU and NVIDIA CUDA. The qualification
uses AIOMFAC-web v3.15 at upstream revision
`24041a1b0db090de1285008814a5285b4b9b746b`.
[Parent plan](aiomfac-activity-package.md) · [Activity and speciation qualification](aiomfac-speciation-and-cuda.md)

## From activities to flux

`ActivityFactory().get_strategy("aiomfac", ...)` binds each analytical component
to a material lane. Each occupied row supplies its full dissolved mass vector
and current trial temperature. The versioned binding evaluates the mixture
once for all requested neutral vapors. It retains an explicit material map,
neutral-output order, parameter digest, precision, solver convention and
backend identity. The GPL package owns parameter tables, speciation and device
kernels. Particula owns the Kelvin factor, pure vapor pressures, transport
and both reservoirs. [Provider contract](../particula/particles/prepared_activity.py)
· [Backend binding](../packages/particula-aiomfac/src/particula_aiomfac/adapter.py)

The fixed CUDA path reevaluates activities before each of four simultaneous
inventory-limited updates. It checks the final composition before publishing
the step. RK23 evaluates each stage and each retry. ROS3 also evaluates the
perturbed compositions used to build the coupled Jacobian and checks the
accepted candidate. Occupancy masking keeps vacant slots out of the activity
solve. The native and resident tests exercise permuted maps, inert material,
different droplets in one population, sulfate re-equilibration, and a failed
trace carbonate–sulfate row. The failed fixed-step test reads the original
device arrays to check rollback. [Transport and event regressions](../examples_tests/aiomfac_closed_box_test.py)
· [Parcel restart and failed-row tests](../packages/particula-aiomfac/tests/test_parcel_cuda.py)

## Finite two-vapor box

At 298 K the water, vanillic acid, butanediol and NaCl mass fractions are
0.977, 0.001, 0.002 and 0.020. The pinned Fortran neutral activities are
0.9878567852129456 for water and 0.0011898857970537608 for butanediol.
Multiplying by the prescribed pure pressures (3167 and 2 Pa) and the
species-specific Kelvin factors gives 3131.6318969763 and 0.00239155087572 Pa.
With water 2% above equilibrium and butanediol 20% below it, their first
fluxes are +1.4316331e−13 and −1.8082919e−18 kg per particle per second.
Reversing the offsets reverses both fluxes. The pressure errors against the
independent calculation are below 1.4e−12 Pa. [Worked source](../docs/studies/examples/thermodynamics/activity/aiomfac_closed_box.py)
· [Figure](../website/results/aiomfac-closed-box/figure-5-1.png) · [Numerical records](../website/results/aiomfac-closed-box/data.json)

Over 4 ms, the worst errors against the refined CPU Radau trajectory across
both reservoir directions were as follows. Each phase-inventory error uses the
initial total for that species. The example also checks each class, each flux
and each activity history separately. Its absolute inventory floor is
1e−30 kg/m³.

| Solver | Relative wet-radius error | Normalized phase-inventory error | Maximum species budget residual |
| --- | ---: | ---: | ---: |
| Native CPU, eight steps per output interval | 3.83e−5 | 1.97e−6 | 4.66e−16 |
| Fixed CUDA, four internal substeps per step | 9.56e−6 | 4.92e−7 | 1.20e−15 |
| Host-controlled CUDA RK23 | 7.15e−9 | 4.48e−10 | 3.11e−16 |
| Host-controlled CUDA ROS3 | 1.61e−9 | 1.99e−10 | 6.21e−16 |
| Resident CUDA RK23 | 7.15e−9 | 4.48e−10 | 3.11e−16 |
| Resident CUDA ROS3 | 1.57e−9 | 2.44e−10 | 5.19e−16 |

Doubling native CPU steps per output interval roughly halves the radius error.
Both nonvolatile materials retain their initial masses. Between resident
segments a reservoir update reverses the tendencies. A separate nucleation
test fills a vacant slot with vanillic acid and checks the first subsequent
two-vapor uptake against the new composition. Both adaptive methods restore
the accepted gas and material inventories bit for bit from disk.

## Moist rising parcel

The opt-in parcel state specifies wet analytical masses (kg/particle), class
counts, ordered vapor mixing ratios (kg/kg dry air), dry-air mass, height,
pressure and temperature. At each trial the ideal gas law uses all vapor
mixing ratios. Each wet radius comes from the summed material volumes.
For each class one conductive surface heat balance couples the signed water
and organic mass fluxes. The parcel first law uses expansion work and the
sum of latent heating over both species. The independent CPU Radau system
integrates vapor mixing ratios and particle masses as distinct variables.
CUDA recovers gas mixing ratios from conserved analytical inventories. The
water-only, no-exchange and organic-only limits exercise separate terms.
[CPU balances and reference](../particula/dynamics/multivapor_parcel.py) · [Limit tests](../particula/dynamics/tests/multivapor_parcel_test.py)

The first comparison starts at 298.15 K and 101325 Pa with two wet radii
(1 and 2.5 µm), 50 and 20 million droplets per kg dry air, and a constant
1 m/s updraft. At 150 m the reference temperature is 297.338816 K. Organic
vapor falls from 32.2332 to 1.43264 µg/kg dry air. The final radii are 9.39026
and 10.67942 µm. Suppressing only organic exchange lowers the final
temperature by 0.000302 K. The organic vapor pressure and vaporization
enthalpy use the measurements in Toktonov's Table B4. Component densities,
unit accommodation and continuum diffusivities are prescribed benchmark
properties. [Worked parcel and thermophysical inputs](../docs/studies/examples/thermodynamics/parcel-experiments/aiomfac_multivapor_parcel.py)
· [Ascent figure](../website/results/aiomfac-multivapor-parcel/figure-5-1.png) · [Numerical records](../website/results/aiomfac-multivapor-parcel/data.json)

| 150 m comparison with CPU Radau | Resident RK23 | Resident ROS3 | Acceptance |
| --- | ---: | ---: | ---: |
| Maximum relative radius error | 2.83e−6 | 2.26e−7 | 1e−3 |
| Maximum phase inventory error / initial species total | 6.88e−7 | 1.94e−7 | 1e−3 |
| Maximum temperature error (K) | 1.08e−7 | 2.47e−8 | 0.01 |
| Maximum absolute water supersaturation error | 8.76e−9 | 2.01e−9 | 1e−4 |
| Maximum species budget residual | 1.88e−16 | 1.88e−16 | 1e−10 |
| Accepted / rejected steps at relative tolerance 2e−6 | 409 / 22 | 687 / 3 | All 151 output times |
| Activity evaluations | 1,724 | 6,210 | Fresh stage and retry values |

Reducing the relative tolerance from 2e−4 to 2e−6 reduced the maximum radius
error from 1.07e−4 to 2.83e−6 for RK23 and from 1.58e−5 to 2.26e−7 for ROS3.
An independent pressure–volume quadrature and species-resolved latent heats
give a maximum first-law discrepancy of 4.61e−7 K at 5 m sampling and
2.04e−8 K at 1 m. Parcel species budgets use a mixing-ratio floor of 1e−30.
At 75 m both methods wrote a checkpoint, restored on the same device,
recaptured, and matched every remaining observation exactly. A checkpoint
includes the ordered state, inventories, time, controller, thermophysical
laws, precision, model revisions, parameter hash and device identity.

## Cloud activation during a 1,500 m ascent

A second closed parcel starts at **82% water RH**, 298.15 K and 101325 Pa.
Water and butanediol are equilibrated with two haze classes before ascent.
The nonvolatile cores have radii of 0.32 and 0.48 µm with a 1:20 vanillic
acid to NaCl mass ratio. Their counts are 50 and 20 million per kg dry air.
Initial butanediol vapor is 3 µg/kg dry air. A 1 m/s updraft takes 25 min
to lift this parcel through saturation and into cloud.

The first upward crossing of 100% ambient water RH is **399.97 m**. An
independent dry ascent holding both vapor mixing ratios fixed gives 398.31 m
from the ideal gas law, hydrostatics and the same saturation-pressure law.
The coupled Radau run reaches its maximum water supersaturation of
**0.42971% at 422 m**. For each class, the diagnostic varies water while
holding that height’s temperature and nonwater masses fixed. It locates the
maximum of the resulting AIOMFAC water activity times the Kelvin factor.
Class activation requires an upward crossing of that critical wet radius,
ambient supersaturation above the critical value, and continued water uptake
and radius growth over 10 m. [Köhler maximum definition](https://amt.copernicus.org/articles/7/2227/2014/amt-7-2227-2014.html)

| Dry core radius | Activation height | Critical wet radius | Ambient / critical supersaturation at crossing |
| ---: | ---: | ---: | ---: |
| 0.32 µm | 463.47 m | 7.656 µm | 0.29210% / 0.00922% |
| 0.48 µm | 707.24 m | 13.726 µm | 0.16793% / 0.00516% |

Both classes reach the growing branch after the ambient supersaturation peak.
At 1,500 m all 70 million represented droplets per kg dry air have activated.
Their wet radii are 20.44 and 20.74 µm, liquid water content is 2.5433 g/m³,
water vapor is 13.8343 g/kg dry air, and temperature is 289.584 K.
The final organic vapor mixing ratio is 0.00575 µg/kg dry air.
The maximum water and butanediol budget residuals are below 1.7e−15.
Sampling the first-law terms at 0.5 m gives a residual of 9.33e−9 K
(9.33e−7 K at 5 m). Refining height sampling and the critical-radius search
moves either activation height by less than 0.002 m. An independent bounded maximization agrees with the refined critical
radii to 3.21e−6 relative. This calculation extends the organic
(subcooled-liquid) vapor-pressure law to 289.6 K, below the lowest
measurement at 298.4 K. The near-surface transport uses the prescribed
continuum model and the parcel is closed to entrainment.
[Cloud activation figure](../website/results/aiomfac-multivapor-parcel/figure-5-2.png) · [Cloud profile, activation thresholds and balances](../website/results/aiomfac-multivapor-parcel/data.json)

Resident CUDA ROS3 reached 1,500 m in 2,178 accepted steps with 5 rejections
and 19,647 activity evaluations. RK23 required 27,056 accepted steps with
6,312 rejections and 133,472 evaluations at the same relative tolerance of
2e−6. Both predicted class activation heights within 0.00214 m of CPU Radau.
Across methods the largest wet-radius error was 2.73e−6 relative, temperature
error 2.14e−6 K, supersaturation error 1.87e−7 absolute, and phase-inventory
error 3.62e−6 of that species’ initial total. Observed warmed replay times
were 9.3 s for ROS3 and 61.7 s for RK23 during local qualification.

## Local qualification

The combined numerical and regression run passed 340 tests on Linux. Two
Warp CPU parameterizations skipped resident CUDA conditional graphs. Their
NVIDIA counterparts executed. The suite included the pinned Fortran fixture
comparisons, cloud activation, sulfate and trace-failure paths, parcel and
box restart, existing condensation solvers and the documentation runner.
Both wheel/source pairs passed the package audit. Fresh Python 3.12 installs
passed pip dependency checks for base, standalone backend, combined extra and
standalone backend with CUDA. The combined install ran the two-vapor box,
150 m parcel and 1,500 m cloud ascent using the installed scripts. The
development GPU run used Warp 1.11.1 and the clean combined install resolved
Warp 1.19.0.

## Hardware and reproduction

The numerical records include setup, accepted and rejected steps, activity
evaluations, warmed timings, observation readbacks and Warp allocator
high-water marks. On the RTX 5060 (8 GiB, sm_120, NVIDIA driver 615.78.08), the 150 m refined replay
took about 0.8 s for RK23 and 3 s for ROS3. The segment interface performed
300 explicit observation readbacks over 150 intervals. Warp pool peaks of
16,272 and 17,104 bytes include the duplicate restart owner. These metrics
exclude driver, JIT and graph-executable memory. Timings depend on the GPU,
driver and Warp version. [Box run manifest](../website/results/aiomfac-closed-box/manifest.json)
· [Parcel run manifest](../website/results/aiomfac-multivapor-parcel/manifest.json)

A separate Nsight Systems capture bracketed four warmed graphs (each parcel
and box method once) between the profiler start and stop. Graph-node tracing
measured the activity kernels, device control and on-device buffer clears.
The trace recorded four graph launches, 116 activity kernel executions,
18 device memsets, zero host/device copies and zero allocation calls. The
bracket excluded initial upload, explicit output reads and restart I/O.
[Profiling script](../packages/particula-aiomfac/tools/profile_condensation.py)

Run these commands from the repository root in a Python 3.12+ environment
with the combined local development install. The example runner uses the
plotting dependencies. The installation probe directly executes both
simulation functions from the installed base wheel outside the checkout.

```sh
AIOMFAC_TEST_CUDA=1 python -m pytest -q packages/particula-aiomfac/tests examples_tests/aiomfac*test.py particula/dynamics/tests/multivapor_parcel_test.py
python scripts/docs_examples.py refresh aiomfac-closed-box
python scripts/docs_examples.py refresh aiomfac-multivapor-parcel
python packages/particula-aiomfac/tools/qualify_packages.py --cuda --dist-dir .artifacts/m6-dist
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda --cuda-graph-trace=node -o .artifacts/m6-profile python packages/particula-aiomfac/tools/profile_condensation.py
nsys stats --report cuda_api_sum,cuda_gpu_mem_size_sum,cuda_gpu_kern_sum .artifacts/m6-profile.nsys-rep
```

The root extra requires `particula-aiomfac>=0.1.3,<0.2`. Clean package checks
build from each source archive, audit contents, and test base-only, backend-only,
combined, and standalone CUDA installs. The public extra requires a published
compatible backend. Milestone 7 owns sparse activity tables. The planning
estimate for this expanded milestone was 8–12 engineer-weeks.
