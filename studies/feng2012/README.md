# Coexisting water, 2-propanol and ammonium sulfate liquids

Feng et al. measured ternary cloud points and coexisting compositions. The two accompanying CSVs contain the 81 binodal compositions and 13 tie lines for **2-propanol** at 303.15 and 333.15 K. The original article also reports separate 1-butanol measurements. [Feng et al. (2012)](https://doi.org/10.1016/j.fluid.2011.12.023)

## Measurement record

The CSVs under `docs/data/thermodynamics/feng2012-isopropanol-*.csv` transcribe the public ThermoML archive for the paper. Its SHA-256 is `78b0674305b4f97795ea111c52c260335867982ef3390854cb2c276cdbba795c`. All compositions are dimensionless mass fractions. `alcohol` denotes 2-propanol and `salt` denotes analytical (NH₄)₂SO₄. The `u95_` columns give the ThermoML compiler's reported expanded uncertainties at 95% confidence when supplied. Blank entries mean the archive gives no uncertainty for that field. [NIST ThermoML XML](https://trc.nist.gov/ThermoML/10.1016/j.fluid.2011.12.023.xml)

- **Binodal** uses data set 1, at 101 kPa. Variable 1 is temperature and variable 2 is alcohol mass fraction. Property 1 is salt mass fraction. The unreported water fraction is the complement.
- **Tie lines** use data set 2, at 101.3 kPa. “Liquid mixture 2” has the larger water fraction on all 13 lines and is labeled alpha. Its water, alcohol and salt mass fractions are property 4, property 1 and variable 2. “Liquid mixture 1” is beta. Its corresponding fractions are properties 2, 3 and 5. The CSV preserves values in the source, including deviations of the summed fractions from unity.

## Reproduce the comparison

Run from the repository root with its paired Particula and AIOMFAC packages on the import path.

```sh
python studies/feng2012/compare.py --output studies/feng2012/comparison.json
```

The model uses the alcohol subgroup assignment for 2-propanol in Zuend et al., two CH₃[alc] groups, one CH[OH] group and one OH. The input salt represents two ammonium ions and one sulfate ion. The solver receives a synthetic feed halfway between the independently normalized measured endpoints by mass. This choice specifies an overall composition within each observed miscibility gap without assuming a measured phase yield. [Zuend et al. (2011), Table 3](https://doi.org/10.5194/acp-11-9155-2011)

For each measured tie line, the report gives the phase count predicted at that feed and temperature. Where two liquids form, it subtracts each original measured endpoint from its same-phase prediction. A separate boundary comparison evaluates the observed cloud-point curve by linear interpolation at each predicted alcohol fraction, then subtracts that observed salt fraction from the predicted one. Duplicate measured alcohol coordinates are averaged for interpolation; points outside the measured range receive no interpolated comparison. The method compares **composition** boundaries. The archive supplies no separation RH, and the feed construction supplies no independent observed phase-amount test.

Numerical diagnostics give the largest analytical mass-balance error on the unit-mass feed, dimensionless transfer-potential mismatch and minimum tangent-plane distance found. Model–measurement errors are reported separately in mass-fraction units. Solver failures and single-phase predictions remain in the record.

## Results for the pinned AIOMFAC-web 3.15 parameters

At 303.15 K the model returned two liquids for six of the eight measured tie-line midpoints. At 333.15 K it returned two for three of five. At each temperature the two most water-rich tested feeds were predicted to be homogeneous. The endpoint and binodal errors below include only the feeds predicted to split. All comparisons pool mass fractions with equal weight. The archive supplies five or fewer reported uncertainties per six endpoint values, so these scores are unweighted.

| Temperature | Two-liquid predictions / measured tie lines | Endpoint RMS error, all three species and both phases | Cloud-point salt fraction RMS error | Largest endpoint discrepancy |
| --- | --- | --- | --- | --- |
| 303.15 K | 6 / 8 | 0.1025 | 0.0896 | 0.2307 |
| 333.15 K | 3 / 5 | 0.1289 | 0.1033 | 0.2647 |

At the available alcohol fractions, predicted binodal endpoints contain **more salt** than the measured cloud points by 0.059–0.144 mass fraction. The salt model flags the 333.15 K calculations as temperature extrapolations. This comparison assumes the declared salts stay dissolved in each liquid. It cannot establish behavior with crystallization.

Across all nine split states the maximum component potential mismatch was below $8\times10^{-14}RT$ and the largest inventory discrepancy was below $3\times10^{-17}$ per unit feed mass. An additional grid samples every strictly positive ternary composition on a 0.01 mass-fraction mesh (4,851 points per temperature) using the public single-liquid routine. It independently evaluates tangent-plane distances from the reported alpha phase using the two neutral activities and the ammonium sulfate ion activity product. All sampled distances were nonnegative, including for the four feeds predicted to be homogeneous. A finite grid cannot exclude an unsampled minimum. Agreement of the computed phase potentials therefore coexists with substantial model–measurement differences.

## A rising parcel crosses the liquid boundary

The parcel example here is a numerical test using the experimental chemical system. It tracks water transport, pressure and temperature at five observation times over a 0.15 m ascent at 2 m/s. There are $10^8$ identical droplets in 1 kg of dry air and one vacant model slot. The 2 µm wet particles initially contain 0.515 water by mass, with a 70:30 alcohol-to-salt dry mass ratio. The finite vapor reservoir starts at 86.5% RH. Water exchanges with the droplets; the two dry material inventories stay fixed. The setup uses an ideal vapor, a constant liquid density of 1000 kg/m³ and 0.072 N/m surface tension. The numerical comparison assumes instantaneous internal equilibrium.

Run the two commands from the repository root. The second requires an NVIDIA device. The first command records the independent SciPy Radau reference for both tolerances; the second exercises a device-resident ROS3 trajectory and verifies its observables against that reference.

```sh
python studies/feng2012/parcel.py cpu --output studies/feng2012/parcel-cpu.json
python studies/feng2012/parcel.py cuda --cpu-reference studies/feng2012/parcel-cpu.json --output studies/feng2012/parcel-cuda.json
```

Both integrators predicted dissolution of the organic-rich phase between the 5 and 15 ms observations as the particle absorbed water. The equilibrium organic-rich mass share fell from 0.555 initially to 0.517 at 5 ms, then reached zero by 15 ms. At 75 ms the predicted water fraction was 0.5962. These times measure transport across the phase boundary under the imposed internal-equilibrium assumption. They give no estimate of a nucleation or mixing delay.

| Local solver tolerance | CPU Radau trajectory (s) | RTX 5060 ROS3 trajectory (s) | CUDA accepted / rejected steps |
| --- | ---: | ---: | ---: |
| $2\times10^{-5}$ | 6.11 | 4.04 | 17 / 2 |
| $2\times10^{-6}$ | 17.14 | 7.53 | 28 / 0 |

Wall times each describe one run, including the four requested observation boundaries. They exclude the separate restart and diagnostic solves. CUDA preparation took 7.90 s including compilation on first use and 0.69 s on the second run with cached kernels. The integrators use different error estimators and take different steps.

At the finer tolerance the maximum CPU–CUDA discrepancy at the five output times was $1.6\times10^{-6}$ in particle water mass fraction, $2.5\times10^{-6}$ in organic-rich mass share, and $3.2\times10^{-8}$ K in temperature. The two methods kept relative total water inventory errors below $4\times10^{-16}$. A separate first-law calculation integrates expansion work from output pressure and specific volume and sums latent heat from condensed water. Its temperature-equivalent discrepancy was below $1.2\times10^{-10}$ K. Restoring the serialized 5 ms state on a fresh resident owner reproduced the terminal device coordinates and controller counts bit for bit at both tolerances.

## Closed-box kinetic benchmark

A closed 1 m³ box at 303.15 K and 101325 Pa contains one occupied droplet class at 10⁶ m⁻³ and one empty slot. Each droplet initially contains 5 × 10⁻¹⁵ kg with mass fractions 0.41 water, 0.413 2-propanol, and 0.177 ammonium sulfate. Water exchanges with an initial vapor concentration of 0.026 kg m⁻³. Both kinetic models use direct stable-liquid activities, the same Kelvin correction and the same reservoir. The regression compares final masses at 100 µs to relative tolerance 2 × 10⁻⁵ and checks that particle-plus-gas water closes to 10⁻¹⁵ kg m⁻³.

| Execution | Integration over 100 µs | Wall time |
| --- | --- | ---: |
| CPU | 96 equal condensation substeps | 21.53 s |
| NVIDIA GeForce RTX 5060, Warp 1.11.1 | Resident adaptive ROS3, relative tolerance 10⁻⁷ | 1.686 s |

These are single-run wall times with prepared providers and a warm Warp kernel cache. The GPU timing includes the final snapshot. The two integrators perform different numbers of rate evaluations, and the CPU run uses refined substeps as a trajectory reference. The single occupied class does not measure throughput for a large particle population.

Run this regression from the repository root with the optional AIOMFAC package and Warp installed.

```sh
AIOMFAC_TEST_CUDA=1 PYTHONPATH=packages/particula-aiomfac/src:. python -m pytest -q -s packages/particula-aiomfac/tests/test_stable_cuda.py::test_two_liquid_resident_ros3_box
```
