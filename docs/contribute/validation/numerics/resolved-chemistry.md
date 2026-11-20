# Resolved chemistry qualification

Particula advanced the same version 2 mechanism on NumPy, Warp CPU and
resident CUDA. The reference cases measure individual material inventories,
gas depletion and extensive mass in separate boxes. All cases use prescribed
temperature, spherical particles, fixed represented counts and synthetic
reaction parameters. The CUDA host used an NVIDIA GeForce RTX 5060 and an AMD
Ryzen AI 9 HX 370. [Mechanism, units and usage](../../../guide/processes/resolved-chemistry.md)

## Bulk material conversion

A + B → C uses molar masses 0.01, 0.02 and 0.03 kg/mol, an inert core,
two active sizes, an empty slot and initially zero product. Initially each
active particle has equal reactant amounts $n_0$. The two boxes have volumes
1 and 2 m³ and temperatures 298 and 318 K. The Arrhenius coefficient is
$0.0006(T/298)\exp[-200/(RT)]$ m³/(mol s).

Write $n$ for the amount of either reactant in mol/particle,
$d=M_A/\rho_A+M_B/\rho_B-M_C/\rho_C$ and $V_\infty=V_0-dn_0$.
The instantaneous volume is $V_\infty+dn$ and the integrated time satisfies

$$
kt=V_\infty(1/n-1/n_0)+d\ln(n_0/n).
$$

An independent scalar root finder recovers each final material mass.
The equal-density case gives $d=0$. Material densities of 1200, 800 and
1600 kg/m³ for A, B and C give a changing volume. Every backend passed
both cases at 1.5 s within relative error $4\times10^{-8}$ and absolute
error $10^{-32}$ kg/particle. The latter case detects freezing the volume
during a trial. Reordering gas and material columns and rebinding retains the
physical solution.

## Competition and near depletion

Two uptake channels convert the same gas X to different particle products
with probabilities 0.006 and 0.002. Both products have molar mass 0.04 kg/mol.
The closed-box reference uses the common radial increment of the particles.
If $v_*$ is the product molar volume averaged with branching fractions 0.75
and 0.25, a radial increment $z$ deposits
$4\pi z(r_0^2+r_0z+z^2/3)/v_*$ mol per particle. This determines the
remaining gas from the initial inventory and represented counts alone.
SciPy DOP853 integrates this single radial variable independently of the
production RK23 mass equations.

- At 1.5 s each product, gas species, size and box meets the independent
  reference. Extensive gas plus particle kg agrees with initial kg to
  `rtol=1e-12, atol=1e-22`. Inert material, spare gas, counts and free
  slots also pass exact comparisons.
- At 3 s tightening `rtol` from $10^{-4}$ to $10^{-8}$ reduces maximum
  gas and particle errors by more than tenfold on each backend. At 45 s less
  than one millionth of the original X remains in each box. The retained
  gas is nonnegative and independently matches the inventory gained by the
  two particle products.
- A separate sequence condenses a different vapor onto a fourth material
  lane before X reacts. The test measures intermediate geometry from a
  condensation-only run on each backend. The coupled history agrees with
  the uptake reference evaluated on that geometry. Both gas and particle
  changes enter the joint budget.
- Mixed gas conversion, uptake and particle aging also run together with an
  empty second population. The second box independently recovers exponential
  gas conversion while all of its particle slots stay empty.

## Continuation and failure boundaries

Each backend reproduced uninterrupted gas and particle arrays bit for bit
after checkpointing halfway through the bulk and uptake cases. CUDA graph
recording leaves simulation time unchanged. Eager and captured runs agree
exactly, including recapture after restoring a checkpoint on the same GPU.
Changing material mappings rejects continuation. The tests also exercise
missing mappings, stale densities, elemental or mass imbalance, probabilities
summing above one, invalid slots and insufficient workspace.

The CPU box transaction discards staged writes when another box exhausts its
trial budget. The resident solver checks all trials before its separate
publication kernel. An unsuccessful device interval faults the owner and
rejects a subsequent snapshot. Resident checkpoints carry accepted and
rejected work counts per box. Warp CPU tests exercise resident code generation.
All 76 chemistry and worked-example checks passed on the RTX 5060 host
with both Warp 1.11.1 and Warp 1.19.0. The published outputs use Warp 1.11.1.

## Worked chamber and full-workload cost

The [uptake and aging example](../../../examples/processes/customization/resolved_reactive_uptake.py)
tracks both populations and both products at 17 observations. At 8 s, 0.2863%
of initial gas mass remains. The 35 nm population contains 40.98% of the
original mass and the 80 nm population contains 58.73%. The maximum absolute
error across all five mass fractions is $2.57\times10^{-10}$ on each backend.
The numerical data and measured wall times are downloadable with the example.
The published run measured 0.325 s on NumPy and 1.032 s on CUDA. These
timings include construction, preparation, all snapshots and teardown,
including graph recording in the CUDA case. At this one-box size NumPy
completed the full workflow sooner.

The separate stiff gas ensemble varies initial A from 0.1 to 2 mol/m³
across boxes, with initial B and C zero. The single-box case starts at
1 mol/m³. Each run integrates 0.01 s with ROS3, compares every box and
species to a Radau reference and checks the conserved inventory. The timer
covers state construction, binding, preparation, integration, snapshot,
checkpoint and close. The table gives medians of three runs following one
warm-up, with one OpenBLAS thread. Times are specific to this development
host and workload.

| Boxes | NumPy (s) | CUDA (s) | CUDA accepted trials per box | CUDA factorizations per box |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0279 | 0.0262 | 286 | 289 |
| 8 | 0.2038 | 0.0281 | 237–297 | 239–300 |
| 32 | 0.8051 | 0.0289 | 237–297 | 239–300 |
| 128 | 3.2837 | 0.0346 | 237–297 | 239–300 |

The ensembles with varying initial concentrations took 2–3 rejected trials and 2151–2700 triangular
solves per box. The spread measures differing numerical work among threads.
GPU instruction-level occupancy was outside this measurement. For three gas species
and three reactions, ROS3 budgets 396 scratch bytes per box, including two
dense 3 × 3 matrices and pivot indices. This reaches 50,688 bytes at 128
boxes. Mechanism arrays, native state and checkpoint storage are additional.
The timing study qualifies this small stiff system; larger reaction networks
need their own scaling and accuracy evidence.

## Reproduce

The scientific implementation used for the published example and ensemble
assessment is `b90bab07a9`. The commands below require the checkout and actual
CUDA. The case suite runs NumPy, Warp CPU and CUDA.

```bash
python -m pytest -q -Werror --no-cov particula/dynamics/chemistry/tests examples_tests/native_gas_chemistry_test.py examples_tests/resolved_reactive_uptake_test.py
OPENBLAS_NUM_THREADS=1 python -m scripts.assess_chemistry_cost
python scripts/docs_examples.py refresh resolved-reactive-uptake
```

The [gas-only qualification](gas-chemistry.md) records the independent Radau
and BDF refinement study and the two-temperature NCAR import cases.
