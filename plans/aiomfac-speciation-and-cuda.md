# Milestones 4 and 5 — CPU equilibrium speciation and direct CUDA activities

The standalone GPL backend 0.1.2 extends the [direct CPU calculator](aiomfac-cpu-activity.md)
to closed equilibria and adds float64 NVIDIA CUDA activities. Particula 0.3.4
selects the backend through its optional `aiomfac` extra. GPU condensation,
resident graph replay and restart belong to milestone 6 in the
[parent delivery table](aiomfac-activity-package.md#delivery-and-release).

## Milestone 4 — CPU equilibrium

| Step | Delivered behavior | Acceptance |
| --- | --- | --- |
| 4A. Resolve the analytical mixture | Preparation retains caller material indices and constructs separate neutral and ion indices, including implicit proton, hydroxide and aqueous CO₂. Bisulfate activates sulfate equilibrium. Proton or bicarbonate in a sulfate mixture also activates it. Bicarbonate, or carbonate with proton or bisulfate, activates closed carbonate equilibrium. | Sodium sulfate alone remains on the direct path. Explicit and implicit CO₂ give the same named products. Reversed declarations and batched temperatures pass. |
| 4B. Solve sulfate | SciPy Brent solves for bisulfate on the attainable sulfur/proton interval. Trial activities include SR, MR, LR and the molality conversion. The fallback freezes coefficients at a candidate from the upstream dissociation estimate and solves the mass-action quadratic. The final evaluated residual gates both paths. | At 298.15 K and 12 wt% ammonium bisulfate, water activity is 0.9602314418143285 and sulfate molality is 0.55462164145014 mol/kg. Tests include inventories below and above the upstream trace cutoff. |
| 4C. Close carbonate and sulfur inventories | SciPy Powell hybrid solves for bicarbonate, carbonate, CO₂, hydroxide and proton in log amounts. A sixth log bisulfate-to-sulfate ratio preserves both positive sulfur amounts. Water changes with CO₂ formation and hydroxide consumption. Each trial recalculates electrolyte-free solvent mass and all activity terms. The verified exporter supplies the upstream ion-specific CO₂ table. | Independent Fortran fixtures cover trace carbon, carbonate-rich conditions, implicit CO₂, coupled sulfate–carbonate and the upstream removal of CaSO₄ in a calcium-containing coupled system. The trace fixture retains a CO₂ mole fraction of 1.06508 × 10⁻²⁰. |
| 4D. Return the final state | Final neutral and ion amounts are mol per kg of analytical input. Results expose a per-row status. Diagnostics report solver work, mass-action and inventory errors, and any CaSO₄ removed. The CPU provider exchanges only mapped volatile neutral material and reports the failed particle slot before committing a step. | The native 298.15 K condensation test starts at the sulfate reference. After 0.1 ms, water activity reaches 0.9603560487906229 and sulfate molality falls to 0.55285317205658 mol/kg. Analytical salt stays at 6 × 10⁻¹⁶ kg per particle. Gas-plus-particle water remains 0.02260844297029946 kg in the unit-volume box. Injected solver exhaustion leaves both reservoirs intact. |

Each row cold-starts from an ideal charge-balanced composition. Up to five
deterministic attempts perturb that starting point by 0, +0.25, −0.25, +1
and −1 in the prescribed log directions. SciPy uses unit variable scaling,
`xtol = 0.01√ε` and `maxfev = 500(n+1)` per attempt. The five/six-equation
formulation replaces the upstream scaled hydrogen and carbon balance rows
with electroneutrality and log carbon inventory. Tracking reacted water closes
hydrogen and oxygen for the closed reactions. The independent reference
records the original Fortran equations and MINPACK return code. Compare final
species, mass action and physical balance errors across implementations.
Iteration counts are implementation-specific.

## Milestone 5 — Direct CUDA activities

A prepared model uploads compact mixture-specific tables to the selected
device once. The host-facing `evaluate` chunks populations with bounded
device workspace and restores the leading batch axes. The stream-facing
`evaluate_into` accepts caller-owned contiguous device arrays and workspace.
One Warp CUDA thread evaluates one row. It recomputes activities locally at
each trial, writes status at that row's original position, and retires when
the row succeeds or exhausts its budget. The iteration requires no host
callbacks or device-to-host transfers.

| Step | Implementation and evidence |
| --- | --- |
| 5A. Device interface | Explicit CUDA selection requires a usable NVIDIA device. The installed backend owns the Warp kernels and packed result schema. Tests use a separate caller stream and buffers and check a failed middle row alongside two successful rows. |
| 5B. Direct checkpoint | For 298 K water/vanillic acid/butanediol/NaCl the RTX 5060 produced water activity 0.9878567852129458 and NaCl mean molal coefficient 0.7037048661941816. The Fortran values are 0.9878567852129456 and 0.7037048661941817. The direct corpus also exercises PEG, reciprocal salts, shared ions, absent species and the temperature branches. |
| 5C. Equilibrium | Sulfate uses bounded bisection with a residual-checked quadratic fallback. The carbonate kernel computes a finite-difference Jacobian, a pivoted dense Newton step and a Powell dogleg trust-region step in unit scaling. It shares the CPU cold-start chemistry, retry directions, verified physical balances and evaluation budget. A reduced-budget device test forces indexed nonconvergence. |
| 5D. Installed capability | The standalone backend exposes direct CPU, direct CUDA and closed speciation. The Particula adapter exposes native isothermal CPU coupling. The base wheel carries the optional dependency metadata. Kernel source, parameters, oracle cases and GPL notices belong to their respective backend artifacts. |

## Numerical evidence

The pinned reference is AIOMFAC-web 3.15 at
`24041a1b0db090de1285008814a5285b4b9b746b`. The 28 case definitions
include 18 direct numerical states, six converged equilibrium states, one upstream equilibrium failure and three
rejected inputs or parameter combinations. Across converged fixtures the measured
maximum absolute discrepancies and residuals were as follows.

| Quantity | CPU | RTX 5060 | Gate |
| --- | ---: | ---: | --- |
| Defined log coefficient | 1.12e-11 | 7.15e-13 | 1e-8 absolute |
| SR / MR / LR terms | 2.38e-13 / 1.03e-11 / 3.97e-12 | 4.62e-14 / 7.62e-13 / 3.80e-13 | 1e-8 absolute each |
| Neutral activity | 8.43e-14 | 2.89e-15 | atol 1e-10, rtol 1e-8 |
| Positive ion molality or CO₂ mole fraction, absolute log ratio | 2.51e-10 | 5.95e-12 | 1e-8 |
| Log mass action at returned composition | 1.59e-11 | 1.78e-13 | 1e-8 |
| Normalized charge / carbon / sulfur balance | 1.31e-11 / 3.62e-15 / 0 | 4.38e-16 / 1.07e-13 / 0 | 1e-10 each |

The tests also compare species IDs, masks, warnings, salt products and ionic
strength. The fresh-process coupled Fortran oracle returned MINPACK status 1
and a largest equation residual of 3.37e-12. Its analytical carbon balance
closed to 5.77e-14 relative and its dissolved sulfur balance to floating-point
zero. The calcium fixture exercises the same coupled equations after the
pinned upstream calcium sulfate removal. [Coupled oracle](../packages/particula-aiomfac/reference/expected/coupled-carbonate-sulfate.json)
and [calcium oracle](../packages/particula-aiomfac/reference/expected/coupled-calcium.json).

A fixed-composition temperature sweep repeated the sulfate, trace-carbonate and
coupled cases at 280, 298.15 and 315 K in fresh Fortran processes. Both
backends agreed on the temperature warning category at each point. The maximum
absolute log-coefficient difference was 1.06e-12 on CPU and 9.20e-14 on CUDA.

At 295 K with mass fractions 1e-9 each of NH₄HSO₄ and KHCO₃ in water, the
pinned hydroxide upper bound is 6.54e-8 mol/kg input. An unconstrained
solution would require 8.58e-8. Fortran returned error 17 and MINPACK status
4, with a maximum equation residual of 0.371 and a 7.26% carbon balance
error. CPU and CUDA raise indexed convergence errors for this case. The
separate fixture preserves the upstream failed iterate for review.
[Measured failure](../packages/particula-aiomfac/reference/expected/coupled-trace-bound.json).

## Reproduction and distribution

From the repository root, first install the two local projects in a Python
3.12+ environment. The CPU package and Fortran reference run in ordinary
Linux CI. For CUDA qualification use a host with a compatible NVIDIA driver.

```sh
pkg=packages/particula-aiomfac
python -m pip install . "./$pkg[cuda]"
python "$pkg/tools/reference.py" fetch
python "$pkg/tools/reference.py" build
python "$pkg/tools/export_parameters.py"
python "$pkg/tools/reference.py" verify
python "$pkg/tools/reference.py" fetch --offline
AIOMFAC_TEST_CUDA=1 python -m unittest discover -s "$pkg/tests"
python "$pkg/tools/qualify_packages.py" --cuda --dist-dir .artifacts/aiomfac-speciation-dist
```

The backend suite passed all 22 tests with CUDA enabled. The native activity
and species-mapping regression suites passed 140 tests. The offline pinned
Fortran and parameter checks passed locally.

Local qualification covers base alone, standalone backend, standalone with
`[cuda]`, and Particula with `[aiomfac]`. The installed probes run outside
the checkout, check package identities, execute mixed and equilibrated
systems, and run native CPU water condensation in the combined install.
The standalone and combined CUDA wheel probes used Warp 1.19.0 on the RTX
5060. The clean archives preserve the MIT/GPL package boundary. A public
installation of this version requires publishing the matching backend and a
Particula release carrying the updated extra. [Backend API and buffer contract](../packages/particula-aiomfac/README.md).

## Timing and memory

The qualification machine has an AMD Ryzen AI 9 HX 370 and an 8 GiB GeForce
RTX 5060 (sm_120). The timing environment used Python 3.12.12, NumPy 2.3.4,
SciPy 1.16.2, Warp 1.11.1, CUDA toolkit 12.9 and driver API 13.4. Each size
runs in a fresh process with one BLAS thread. Analytical water mass varies
linearly from 90% to 110% of its case value. The one-row case uses 90%.
Timings include diagnostics. CPU and CUDA host times include result assembly.
CUDA kernel times sum stream-event intervals across chunks. Peak device memory
is Warp's measured pool high-water mark during host-facing execution, including
the prepared workspace and outputs. It excludes driver and compiler memory.

```sh
OPENBLAS_NUM_THREADS=1 python packages/particula-aiomfac/tools/benchmark_activity.py sulfate 4096
OPENBLAS_NUM_THREADS=1 python packages/particula-aiomfac/tools/benchmark_activity.py sulfate 4096 --device cuda:0
```

A separate cold mixed-case run measured 0.465 s for device preparation and
9.717 s for the first call, including 9.714 s for empty-cache JIT and module
load. Reuse the prepared model for population and timestep evaluations.
Pass `--kernel-cache` with a new directory to reproduce cold compilation.

| Chemistry | States | CPU host ms | CUDA kernel ms | CUDA host ms | Peak device MiB | Evaluations/row CPU / CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed | 1 | 0.574 | 0.483 | 0.795 | 8.42 | 0.00 / 0.00 |
| mixed | 128 | 1.194 | 0.824 | 1.339 | 8.43 | 0.00 / 0.00 |
| mixed | 4,096 | 5.696 | 1.155 | 2.136 | 8.58 | 0.00 / 0.00 |
| mixed | 65,536 | 130.223 | 18.387 | 36.910 | 8.71 | 0.00 / 0.00 |
| sulfate | 1 | 3.182 | 2.413 | 2.735 | 4.42 | 12.00 / 42.00 |
| sulfate | 128 | 226.453 | 4.088 | 4.446 | 4.43 | 12.52 / 43.02 |
| sulfate | 4,096 | 8,080.562 | 6.166 | 8.492 | 4.52 | 12.49 / 43.37 |
| sulfate | 65,536 | 136,002.425 | 96.938 | 116.964 | 4.58 | 12.47 / 43.31 |
| carbonate | 1 | 2.561 | 2.285 | 2.703 | 6.61 | 12.00 / 20.00 |
| carbonate | 128 | 254.863 | 3.410 | 3.809 | 6.62 | 12.00 / 17.70 |
| carbonate | 4,096 | 9,234.860 | 5.273 | 6.779 | 6.80 | 12.00 / 17.70 |
| carbonate | 65,536 | 153,255.118 | 84.533 | 116.322 | 6.96 | 12.00 / 17.75 |

Across these fresh processes CPU preparation took 3.4–6.8 ms. CUDA preparation
took 0.44–0.59 s, including Warp initialization. With a populated kernel cache,
the first CUDA call took 3.5–7.3 ms. The three CPU chemistry sweeps ran concurrently
in separate single-threaded processes. Timing samples include normal machine load.

In an additional host-to-host sweep, mixed fixed chemistry first favored CUDA
at 512 rows (2.19 ms CPU, 1.83 ms CUDA). Batches of 256 rows favored CPU (0.67 ms
versus 1.77 ms). For sulfate and carbonate, one-row timings overlapped
across runs. Two rows already favored CUDA for both systems in the finer
sweep (sulfate 5.54 versus 3.07 ms, carbonate 4.90 versus 2.94 ms).
These are observed crossovers on this machine with prepared models. Hardware,
temperature, chemistry and nonlinear iteration count can shift them.

The original planning estimate was 3–5 engineer-weeks for each milestone
with GPU access and scientific review. Broader coverage of concentrated and
trace coupled mixtures remains relevant when qualifying new aerosol compositions.
