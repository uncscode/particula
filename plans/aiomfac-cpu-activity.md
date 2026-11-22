# Milestones 2 and 3 — Direct CPU activities and native water condensation

[Milestones 4 and 5](aiomfac-speciation-and-cuda.md) extend this baseline to closed equilibrium systems and direct CUDA activities.

## Delivery and scope

One prepared float64 model evaluates independent liquid mixtures in bounded
NumPy batches. The optional Particula provider binds a dissolved-material map
once and evaluates current masses at the actual temperature on every native
pressure call. The qualified workflow uses native isothermal kinetic stepping
and a finite gas reservoir. The release candidate is backend 0.1.1 with a local
Particula 0.3.4 wheel. Public distribution requires publication of the matching
backend and a Particula release that carries the updated extra.

| Step | Implementation | Acceptance evidence |
| --- | --- | --- |
| Generate active parameters | Initialize the hash-verified v3.15 build, export arrays in Fortran orientation, record units, axis offsets, branch and hashes. | Regeneration verifies the installed arrays. Source and runtime data belong to the GPL distribution. |
| Prepare and evaluate neutral mixtures | Resolve subgroup counts, molar masses and surface/volume terms. Calculate the combinatorial and residual terms at the requested temperature, including pure reference subtraction and PEG. | Pure water, water–organic binary, organic-only and mixed temperature sweep agree with the Fortran cases. |
| Add fully dissociated ions | Aggregate shared ions, compute molalities on electrolyte-free solvent mass, evaluate LR/MR and convert ionic log coefficients to the aqueous molal reference. | Qualified individual terms, ion pairs, mean coefficients, activity products and charge balance. Four systems that activate equilibrium raise a preparation error. |
| Connect native pressure and rates | Select the v1 provider lazily with the activity factory. Bind full material order including nonvolatile solutes and inert lanes. Exchange only configured neutral targets. | Compare equilibrium pressure and initial water flux with the stored Fortran activity, then advance and close the water budget in both directions. |
| Qualify distributions | Exercise ideal, κ, BAT, custom and missing-provider behavior on base alone. Run the mixed calculation with backend alone and both trajectories in the combined environment. | Build each wheel from its source archive, check contents and hashes against accidental bundling, install and run probes outside the import tree. |

## Equation and fixture coverage

The qualification uses the upstream CLI as the independent implementation.
Agreement measures port fidelity. Experimental accuracy depends on the assigned
subgroups, parameters, liquid phase and reference conventions. [Upstream output conventions](https://aiomfac.lab.mcgill.ca/help.html)

| Branch or contract | Reference or test |
| --- | --- |
| Pure liquid, absent neutral, neutral-only batch and organic solvent | `water-limit`, `neutral`, `organic-only` |
| Dataset 1 SR temperature dependence, water reference, salting effects and molality conversion | `mixed-cool`, `mixed`, `mixed-warm` at identical composition, plus `mixed-reordered` |
| Dilute electrolyte, cold warning, shared/absent ion and salt, crossed ion pairs | `nacl-dilute`, `nacl`, `nacl-cold`, `shared-ion`, `absent-salt`, `reciprocal-salts` |
| PEG volume/surface and PEG–ammonium/sulfate overrides, sulfate without a proton source, multivalent stoichiometry | `peg-sulfate`, `peg-ammonium-sulfate`, `multivalent` |
| Supported analog parameters and unavailable Ca/methanesulfonate pair | `analog`, `missing-pair` |
| Like-cation and three-ion interactions | `like-ion` and direct MR evaluation at the Fortran sulfate solver’s final individual molalities |
| Preparation and input failures | `sulfate`, three carbonate fixtures, sulfate plus HCl, `invalid-subgroup-pair`, `negative-input`, indexed nonfinite masses and invalid temperatures |
| Native compatibility and ownership | Custom pressure override, material and gas permutations, heterogeneous particles, explicit inert material, empty slot, inadmissible transfer and unchanged state on an active-row failure |

## Results on Linux

GNU Fortran 16.2.1 regenerated **25 cases**. Eighteen direct cases produced
numerical predictions. Four require equilibrium speciation and three exercise
input or parameter errors. Across the 18 direct states the maximum absolute
differences from Fortran were **6.67e-15** for SR, **4.45e-16** for MR,
**6.67e-16** for LR, **6.67e-15** for defined log coefficients and **1.00e-15**
for neutral activities (values rounded up). The largest normalized charge
residual was **5.90e-17**. Qualification checks each species by ID, masks,
warnings, coefficients, mean salt coefficients, products and basis conversion.
[Direct tests](../packages/particula-aiomfac/tests/test_direct.py)

| Mixed case at 298 K | Stored Fortran | NumPy CPU |
| --- | ---: | ---: |
| Water activity | 0.9878567852129456 | 0.9878567852129452 |
| NaCl mean molal coefficient | 0.7037048661941817 | 0.7037048661941815 |

With pure water pressure 3167 Pa, surface tension 0.072 N/m, density
1000 kg/m³ and initial radius 1.0607844 µm, the equilibrium pressure was
**3131.631897 Pa**. The example sets gas water to 102% or 98% of this value.
Each run advances 20 steps of 0.1 ms with 10¹⁰ droplets in 1 m³. Water mass
in the particle column below refers to the full droplet population.
[Time series](../packages/particula-aiomfac/examples/closed_water.csv) · [Executable example](../packages/particula-aiomfac/examples/closed_water.py)

| Gas reservoir | Initial flux (kg/s per droplet) | Final water activity | Final radius (µm) | Particle water change (kg) | Maximum absolute water budget residual (kg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 102% | +1.4316331e-13 | 0.98852339 | 1.080468 | +2.83536e-6 | 6.94e-18 |
| 98% | −1.4316331e-13 | 0.98712254 | 1.040752 | −2.77951e-6 | 1.39e-17 |

Dry component masses were exactly conserved. The independent rate calculation
uses the stored water activity, Kelvin correction, gas partial pressure and
Fuchs–Sutugin transport correction. The native comparison includes a permuted
layout with an inert material, two active compositions and a vacant slot.

## CPU baseline

AMD Ryzen AI 9 HX 370, Linux, single OpenBLAS thread, mixed chemistry at 298 K
with varying water amounts, diagnostics disabled. Times are the median of seven
warmed calls. Peak traced allocation includes new result arrays and temporary
arrays and excludes the existing input, model and interpreter.
[Benchmark](../packages/particula-aiomfac/tools/benchmark_cpu.py)

| Independent states | Time (ms) | Peak allocation (MiB) |
| ---: | ---: | ---: |
| 1 | 0.523 | 0.011 |
| 128 | 0.353 | 0.188 |
| 4,096 | 4.834 | 1.186 |
| 65,536 | 74.118 | 9.038 |

Use the [backend guide](../packages/particula-aiomfac/README.md) to repeat the
three clean installations and the [reference guide](../packages/particula-aiomfac/reference/README.md)
to repeat numerical qualification and table regeneration. Milestone 4 owns
equilibrium speciation. CUDA, hybrid tables, resident execution and viscosity
are outside this CPU qualification.

## Installation and regression evidence

The local Linux qualification built both wheels from their respective source
archives and passed the base-only, backend-only and combined installations
under Python 3.12. The base wheel and source archive passed the backend payload
audit. The runtime backend wheel was 48 KiB including tables, provenance and
notices. The combined installation ran the two closed reservoirs and the
permuted material layout from the unpacked backend source archive. Retained
local wheels are in `.artifacts/aiomfac-cpu-dist/{base,backend}`.

The compiler-independent package suite passed 13 tests, including the
parameterized reference corpus and a single native pressure call checked
against two independently recorded Fortran compositions. The Particula particle
and condensation suites passed 1,109 tests. Three additional execution suites
passed 82 tests with nine skips. CI is configured to repeat the package checks
and compiler-backed regeneration on Linux. These results are from local runs.
