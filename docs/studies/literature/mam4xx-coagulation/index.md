# Particle-resolved coagulation leaves 6.7% fewer particles for nearby modes but agrees in number for widely separated modes

![At number ratio four, Particula retains 93.3% of modal number at 60 nm and 100.0% at 220 nm after one hour.](../../../images/mam4xx-coagulation/graphical-abstract.svg)

Two equally initialized aerosol models can lose similar numbers of particles while retaining different size spectra. In a dry box with four 26 nm Aitken particles per accumulation particle, Particula leaves 6.7% fewer particles than MAM4xx after one hour when the accumulation mode starts at 60 nm. Moving it to 220 nm reduces the difference to 0.04%, within sampling variability. At initialization the Particula Brownian kernel predicts **19–23% faster number loss in all nine mixtures**. A smaller discrepancy at the endpoint consequently cannot establish agreement on the underlying collision rates. [Number trajectories and spectra](mam4xx_coagulation.py)

## Mode separation changes the number discrepancy

The table gives the high-resolution Particula mean minus MAM4xx at one hour, as a percentage of the modal number. Values after ± are 95% sampling half-widths for the mean of 12 seeds. They exclude model and input uncertainty.

| Initial accumulation diameter | Aitken/accumulation 0.25 | Ratio 1 | Ratio 4 |
| --- | ---: | ---: | ---: |
| 60 nm | −2.03 ± 0.51% | −3.20 ± 0.56% | −6.73 ± 0.81% |
| 110 nm | −0.62 ± 0.70% | −1.00 ± 0.67% | −3.75 ± 0.99% |
| 220 nm | −0.20 ± 0.71% | +0.36 ± 0.69% | −0.04 ± 0.80% |

With 220 nm accumulation particles, the MAM4xx endpoint is inside all three Particula sampling intervals. Normalizing each final spectrum by its own number still gives total-variation distances of 0.045–0.070 across these cases. The metric is half the sum of absolute differences in bin fractions. At equal initial counts, it falls from 0.094 for the 60 nm mode to 0.045 for the 220 nm mode. Size-resolved output can therefore distinguish predictions that a total particle counter would struggle to separate.

## Independent model and inputs

Fast et al. provide the EAMxx source used for their regional MAM4xx simulation. This independent comparison runs the archived one-subarea coagulation calculation for synthetic dry boxes. Its Aitken and accumulation modes start at the same total number and material volume as the Particula particles. The reference uses the archive's simulation branch `78fb724b939f7e5e6300719ddf73682ea1bc3320` and its MAM4xx gitlink `0f51633242168c90018f6e296d02776f86dc9be6`. The coagulation header in the deposited working directory has the same SHA-256 as the pinned Git source. The example bundle includes the reference outputs, host driver and generator. [Fast et al.](https://doi.org/10.5194/gmd-19-11103-2026) · [Archived setup](https://doi.org/10.5281/zenodo.19140449) · [Pinned MAM4xx source](https://github.com/eagles-project/mam4xx/blob/0f51633242168c90018f6e296d02776f86dc9be6/src/mam4xx/coagulation.hpp)

The dry Aitken number-median diameter is 26 nm. Accumulation diameters of 60, 110 and 220 nm describe progressively separated modes. Their geometric standard deviations are 1.6 and 1.8. The synthetic boxes contain 200,000 particles cm⁻³ at 298.15 K and 101325 Pa, with uniform dry density 1770 kg m⁻³. The high number concentration makes coagulation measurable over one hour. Varying the initial Aitken-to-accumulation number ratio (0.25, 1, 4) changes the relative importance of intra- and intermodal collisions. The coarse and primary-carbon modes start empty.

## Matched moments and size spectra

For each mode, Particula places unit-count particles at midpoint lognormal quantiles and rescales their volumes to match the analytical dry mass. MAM4xx updates its modal number and component mass with the published scalar coagulation routines. After each step the driver reconstructs number-median diameter from the third and zeroth moments at the prescribed width. Particula lets the individual dry particles merge. All 49 interior bin edges are common to both models and number closure includes particles outside the plotted range. [Runnable example and spectra](mam4xx_coagulation.py)

Each of nine mixtures uses 480, 1,920 and 3,840 initial particles and 12 independently seeded runs per resolution. Three mixtures also use half the 60 s outer step with 12 fresh seeds. The modal reference steps at 10, 30 and 60 s with the 10 s result used for the trajectory comparison. Independent CPU Gauss–Hermite quadrature integrates the initial Particula collision kernel over both distributions. These calculations distinguish an initial difference in collision rates from subsequent changes in distribution shape.

## Numerical checks and interpretation

Halving the particle outer timestep from 60 to 30 s changes the endpoint mean by 0.23–0.47 percentage points of modal number across three checks, including sampling variability. Increasing from 1,920 to 3,840 particles shifts the mean by at most 1.04 percentage points across the nine mixtures. Changing the modal step from 10 to 60 s changes reference number by at most 0.27% at the observation times. The largest Particula relative dry-mass drift is below 1e-12. The reference generator checks its header checksum and the analysis verifies mass conservation across all modal timesteps. A separate replay of the authors’ archived E3SM one-subarea test case reproduced four final modal numbers within 2.0e-10 relative error and all 28 material-by-mode outputs within 7.8e-10 for nonzero values, with exact agreement at zero. [E3SM validation case](https://github.com/eagles-project/mam_x_validation/blob/3eb380561d71f7fdbbef902f8ea36850704a4e69/coagulation/mam_coag_1subarea.py)

The initial rate check integrates the Particula kernel over the specified continuous lognormals. It establishes an initial model difference before any spectrum evolves. The time series then reflects both collision modeling and subsequent shape evolution. A kernel-matched modal experiment would isolate how much of the remaining size-spectrum difference arises from holding the modal widths fixed. These synthetic mixtures quantify a process difference under the stated conditions. They do not evaluate field accuracy or the cost of the coupled atmosphere model.

## Reproduction

From a checkout, run `python scripts/docs_examples.py refresh mam4xx-coagulation` with CUDA. Run `python scripts/docs_examples.py replot mam4xx-coagulation` to redraw the saved results. To regenerate the frozen reference, obtain the pinned MAM4xx header above and run `python scripts/mam4xx_coagulation/build_reference.py PATH/TO/coagulation.hpp docs/data/mam4xx-coagulation/modal.csv` with a C++17 compiler. The generator verifies the archived header hash and compiles the scalar routines without modifying their bodies. Its host adapter supplies double-precision math and the modal layout. The result is an isolated microphysics calculation. The regional model's transport, humidity changes, modal redistribution by other processes, and host-model computational costs are outside the comparison.

## Matching the thermal kernel changes the interpretation of the 220 nm benchmark

At 220 nm the original agreement in total number concealed two effects of opposite sign. In the 4:1 benchmark, MAM4xx leaves 64,570 particles cm⁻³ after one hour. Integrating Particula’s Brownian kernel over the same evolving fixed-width modes gives 58,642. Allowing the individual particles to merge gives 64,544 ± 515 cm⁻³. At 60 nm the corresponding results are 100,272, 90,736 and 93,520 ± 812 cm⁻³. Intervals are 95% Monte Carlo intervals for 12 native CUDA replicates at fixed input. The matched closure updates number and dry volume separately and assigns the small particle’s entire volume to the receiving mode in every unlike-mode collision. [Runnable benchmark and atmospheric comparisons](regimes.py)

The matched initial number losses are 67.85 and 157.21 cm⁻³ s⁻¹ for 60 and 220 nm, compared with 55.46 and 127.94 in the archive. Over the first hour the archive transfers 3.977% and 0.153% of all dry volume from Aitken to accumulation, compared with 4.506% and 0.163% under matched Brownian integration. The initial donor-to-receiver dry-volume rates are 4.298 × 10⁻¹⁶ and 9.688 × 10⁻¹⁶ m³ m⁻³ s⁻¹. A separate calculation with a volume-weighted donor lognormal verifies the transfer moment. Direct number quadrature verifies the loss moment. Using 24 versus 32 quadrature nodes per mode changes the number trajectories by less than two parts in ten million.

## What the archived atmospheric model carries

| Candidate | Status in the archived calculation | Source or host coupling |
| --- | --- | --- |
| Size distribution and transfer | Simplified. Fixed lognormal widths, separate number and volume moments, three intermodal transfer pairs. | Aitken–accumulation, primary carbon–accumulation and Aitken–primary carbon enter the coagulation routine. The host supplies wet diameters and densities. |
| Electrostatic collision, ion attachment and recombination | Omitted from this coagulation path. | The host calls the scalar coagulation routine without free-ion concentrations or aerosol charge states. The nucleation and aging routines have distinct responsibilities. |
| van der Waals attraction and dielectric polarization | Omitted from the audited pair-rate calculation. | The archived thermal kernel uses modal Brownian rates. |
| Coarse-mode pair collection, differential settling and shear | Omitted as aerosol–aerosol encounter mechanisms in this routine. Surface sedimentation and removal by precipitation are represented elsewhere. | Dry deposition transfers aerosol to the surface. Wet scavenging and convection remove or transport aerosol through separate host processes. |

The comparisons below hold humidity, transport, cloud processing and emissions fixed in a closed dry box. A stored dry-volume transfer into particles above 100 nm and a surface deposition flux answer different questions. [Archived coagulation](https://github.com/eagles-project/mam4xx/blob/0f51633242168c90018f6e296d02776f86dc9be6/src/mam4xx/coagulation.hpp) · [Host subarea processing](https://github.com/eagles-project/mam4xx/blob/0f51633242168c90018f6e296d02776f86dc9be6/src/mam4xx/mam4_amicphys.hpp) · [Dry deposition](https://github.com/eagles-project/mam4xx/blob/0f51633242168c90018f6e296d02776f86dc9be6/src/mam4xx/seq_drydep.hpp) · [Wet removal](https://github.com/eagles-project/mam4xx/blob/0f51633242168c90018f6e296d02776f86dc9be6/src/mam4xx/wet_dep.hpp)

## Atmospheric process comparisons

The three published trimodal spectra have coarse diameters of 0.58, 0.93 and 0.86 µm and coarse concentrations of 3.1, 0.72 and 5.4 cm⁻³, respectively. Our primary comparison uses a common 298.15 K to isolate spectrum effects. Field-informed controls use 13.9–14.6 °C for remote marine air, −4 to 17 °C for boreal background and 8.2–34 °C for urban polluted air. These ranges and the size distributions come from different campaigns. Concentration varies by ±30% around the published modal values. Ion production spans 2–10 pairs cm⁻³ s⁻¹, with equal mobility for both signs and recombination at 1.6 × 10⁻⁶ cm³ s⁻¹. [Size spectra, Lance et al., Table 1](https://doi.org/10.1029/2004JD004596) · [Marine event temperatures](https://doi.org/10.5194/egusphere-2026-6891) · [Boreal temperatures](https://doi.org/10.1002/2017GL076394) · [Urban temperatures](https://acp.copernicus.org/articles/20/12721/2020/) · [Measured ion production](https://www.borenv.net/BER/archive/pdfs/ber12/ber12-323.pdf)

The coarse comparison adds the entire coarse mode to the fine-only Brownian population. Settling and shear each add one mechanical collision process to neutral Brownian encounters. Settling uses geometric collection efficiency of one. Turbulent dissipation varies from 0.001 to 0.1 m² s⁻³. The charge experiment compares a neutral hard-sphere reference to the same kernel averaged over a size-dependent stationary bipolar charge distribution. The dynamic experiment starts with neutral aerosol and an initially recombination-balanced free-ion pool. Charge moves through attachment, recombination and particle mergers. Neutral dispersion replaces the neutral hard-sphere law. The combined dielectric/dispersion law replaces the charge-only thermal law on the same stationary charged population. Each pair therefore receives exactly one thermal kernel.

Material controls span Hamaker constants of 4.45 × 10⁻²⁰ to 1.78 × 10⁻¹⁹ J and relative permittivities of 2.1–12. The nominal values are 8.9 × 10⁻²⁰ J and 5.9. Uncovered radius/charge pairs retain the bare thermal rate. They account for roughly 22%, 13% and 14% of the initial charge-averaged encounter rate. A neutral two-radius CPU calculation checks the GPU capture kernel. Concentrating all charge probability at zero recovers the neutral hard-sphere rate. Refinement increases size bins per decade from 8 to 12, extends the charge range from ±3 to ±5 and tightens the integration tolerance. Collisions that would exceed the finite state domain are excluded. The 24-hour dry-volume relative error stays below 10⁻⁸ and the free-ion plus particle charge residual stays below 2 × 10⁻⁶ elementary charges cm⁻³. The ion source introduces equal positive and negative charge.

## Ranked changes in sub-100 nm retention at 24 hours

Signed percentages compare the fraction of the initial ultrafine number remaining in each process pair. Absolute differences use final concentrations directly. The range spans the stated inputs and refinement. The grid column gives the absolute shift in percentage points after numerical refinement. Rows are ordered by central magnitude. The full charged interaction is an alternative to neutral dispersion and uses its own charged reference. Times are seconds per central 24-hour box on this machine, including available table setup. The clean marine coarse row also pays for the shared capture-table construction used by later rows. Subsequent timings reflect warm tables. These timings do not estimate coupled MAM4xx cost.

### Clean marine

Neutral dispersion and coarse Brownian collection have overlapping ranges. Pilot the coarse pairs and a neutral attraction correction together. Settling and shear make small additions at geometric efficiency one.

| Process representation | ΔN <100 nm (cm⁻³) | Change (%) | Input and grid range (%) | Grid shift (points) | Box time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral dispersion | -11.4 | -3.47 | -5.34 to -1.93 | 0.0086 | 0.004 |
| Full charged interaction | -11.1 | -3.38 | -5.16 to -1.9 | 0.01 | 0.003 |
| Add coarse Brownian pairs | -8.31 | -2.48 | -3.02 to -1.69 | 0.019 | 0.702 |
| Equilibrium bipolar charge | -0.552 | -0.168 | -0.209 to -0.118 | 0.00097 | 0.003 |
| Differential settling | -0.313 | -0.0952 | -0.126 to -0.0696 | 0.0017 | 0.003 |
| Fixed-width closure | +0.283 | +0.0839 | +0.0406 to +0.102 | 0.043 | 0.007 |
| Evolve free ions and charge | +0.166 | +0.0504 | +0.0218 to +0.0577 | 0.029 | 0.135 |
| Turbulent shear | -0.00296 | -0.000901 | -0.00377 to -0.00021 | 1.3e-05 | 0.003 |

### Continental background

Neutral dispersion leads across the input range. Check the two-mode closure against the particle reference next. Equilibrium charge and coarse collection have overlapping effects.

| Process representation | ΔN <100 nm (cm⁻³) | Change (%) | Input and grid range (%) | Grid shift (points) | Box time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral dispersion | -109 | -11.4 | -15.6 to -7.14 | 0.008 | 0.008 |
| Full charged interaction | -104 | -11 | -14.9 to -7.04 | 0.00048 | 0.008 |
| Fixed-width closure | -18.6 | -1.95 | -2.55 to -1.03 | 0.14 | 0.014 |
| Equilibrium bipolar charge | -5.53 | -0.58 | -0.665 to -0.455 | 0.0079 | 0.008 |
| Add coarse Brownian pairs | -5.09 | -0.534 | -0.591 to -0.398 | 0.00012 | 0.008 |
| Evolve free ions and charge | +0.762 | +0.0804 | +0.0243 to +0.081 | 0.056 | 0.154 |
| Differential settling | -0.235 | -0.0248 | -0.0319 to -0.0198 | 0.00066 | 0.009 |
| Turbulent shear | -0.00387 | -0.000408 | -0.00167 to -0.000104 | 5.5e-06 | 0.008 |

### Polluted air

Neutral dispersion is the largest persistent effect. The mode-width effect changes sign across inputs. A diagnostic bipolar state captures most of the smaller charge effect. Its incremental sensitivity to prognostic ions is comparable to numerical refinement in some cases.

| Process representation | ΔN <100 nm (cm⁻³) | Change (%) | Input and grid range (%) | Grid shift (points) | Box time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral dispersion | -961 | -35.3 | -44.8 to -26.3 | 0.21 | 0.030 |
| Full charged interaction | -867 | -32.5 | -41.6 to -24.3 | 0.25 | 0.030 |
| Fixed-width closure | -104 | -3.74 | -14.2 to +8.59 | 0.64 | 0.081 |
| Equilibrium bipolar charge | -51.4 | -1.89 | -2.01 to -1.77 | 0.047 | 0.030 |
| Evolve free ions and charge | +12.3 | +0.462 | +0.144 to +0.472 | 0.32 | 0.323 |
| Add coarse Brownian pairs | -10.9 | -0.391 | -0.451 to -0.324 | 0.0039 | 0.030 |
| Differential settling | -1.56 | -0.0562 | -0.0562 to -0.0515 | 0.0024 | 0.036 |
| Turbulent shear | -0.0504 | -0.00181 | -0.00438 to -0.00061 | 2.2e-05 | 0.032 |

### What the particle-resolved reference changes

To compare with the archived widths, we preserve the observed number and dry volume of each fine mode and recompute its median diameter at the MAM4xx width. Archived and kernel-matched boxes use those same moments and widths. Native Particula particles start from the same projected distributions and then evolve freely. A second native run starts with the published widths. Marine and continental boxes start with 1,520 and 1,800 particles, six seeds and a 300 s step. The polluted box starts with 414 particles, four seeds and a 600 s step. The field-width refinement doubles each particle count and halves its step with four fresh seeds. A larger three-regime ensemble exceeded the two-hour notebook limit. The reported intervals use the two completed particle counts in each regime. A pilot with the broad polluted spectrum exhausted the initial limit of 128 internal steps. The polluted runs allow up to 4,096 internal steps. The cleaner regimes use 128, with the same 0.01 s minimum internal step and 64 MiB device-memory ceiling. The GPU schedules guarded work up to the cap, which makes an unnecessarily high cap costly even in dilute air.

| Regime | Archived total at 24 h (cm⁻³) | Matched modal total (cm⁻³) | Particle total, same initial widths (cm⁻³) | Sub-100 nm, field widths (cm⁻³, 95% MC) | Refined field widths (cm⁻³, 95% MC) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Marine | 356.4 | 351.8 | 351.0 | 337.4 ± 2.2 | 338.7 ± 2.6 |
| Continental | 1,195.2 | 1,125.3 | 1,167.7 | 940.3 ± 7.9 | 950.9 ± 17.3 |
| Polluted | 8,866.0 | 8,102.4 | 9,333.3 | 2,000.0 ± 1,145.8 | 2,416.7 ± 459.3 |

At the archived initial widths, the following differences test whether changing the rate integral alone recovers the particle result. The last column bounds Monte Carlo uncertainty in total number using the sum of the two size-class interval half-widths. That bound allows for covariance between the size classes.

| Regime | Archived minus native total (%) | Kernel-matched modal minus native total (%) | Native 95% MC half-width upper bound (cm⁻³) |
| --- | ---: | ---: | ---: |
| Marine | +1.55 | +0.23 | 3.1 |
| Continental | +2.36 | -3.63 | 19.7 |
| Polluted | -5.01 | -13.19 | 2190.6 |

Replacing the archived rate integral brings the marine total within 0.23% of the particle mean. In continental air the same substitution moves the error from +2.36% to −3.63%. A revised rate integral alone therefore does not recover both central cases. In polluted air the sampling interval is too wide to choose between the two modal predictions.

The measured-width Brownian comparison checks the neutral sectional baseline used for screening. It excludes the coarse mode on both sides. The deviations below use the independent, refined native ensembles.

| Regime | Sectional sub-100 nm (cm⁻³) | Native sub-100 nm (cm⁻³, 95% MC) | Sectional minus native (%) |
| --- | ---: | ---: | ---: |
| Marine | 337.5 | 338.7 ± 2.6 | -0.36 |
| Continental | 953.2 | 950.9 ± 17.3 | +0.24 |
| Polluted | 2,796.3 | 2,416.7 ± 459.3 | +15.71 |

Each sectional endpoint falls within the sampling interval of the refined neutral particle ensemble. The polluted interval is wide enough to include substantial differences. A particle-resolved ensemble with each force enabled would be needed to measure sampling uncertainty on the paired force effects directly.

Native ensemble wall times include all seeds and all four output times. Hardware is an NVIDIA GeForce RTX 5060. The process comparisons use GPU collision tables and CPU integration of the size and charge distributions.

| Regime | Archived-width native (s) | Field-width native (s) | Refined field-width native (s) |
| --- | ---: | ---: | ---: |
| Marine | 50.0 | 50.1 | 66.8 |
| Continental | 50.3 | 50.3 | 66.7 |
| Polluted | 521.3 | 520.3 | 1090.0 |

The saved numerical data contain spectra, number below and above 100 nm, volume above 100 nm, and absolute and percentage differences at 1, 6 and 24 h. The archived two-mode outputs also retain dry-volume exchange and timestep sensitivity. Regenerate the regional archive with `python -m scripts.mam4xx_coagulation.regime_reference PATH/TO/coagulation.hpp docs/data/mam4xx-coagulation/regional-modal.csv`. Run `python scripts/docs_examples.py refresh regimes` on CUDA to reproduce the ensembles and `python scripts/mam4xx_coagulation/export_regimes.py` to regenerate the two CSV tables. A long workstation run can set `PARTICULA_MAM4XX_CHECKPOINTS` to a fresh directory for this checkout. It saves each regional particle case, its hardware and exact inputs before advancing.

The neutral dispersion calculation reproduces the full charge-averaged interaction’s central sub-100 nm count to +0.07% in marine air, +0.12% in continental background and −2.35% in polluted air. The retention metric counts all particles below the threshold, including newly merged particles in that size range. It does not follow individually tagged initial particles. The intervals on the native boxes quantify Monte Carlo sampling at fixed inputs. The process tables use deterministic integrations of published representative spectra. Campaign-to-campaign sampling uncertainty is unavailable from modal means alone. Before implementing a global parameterization, integrate the neutral dispersion rate over the host modes and compare it with the resolved response, first with polluted spectra and then with marine coarse particles. Collect material properties and spectra from the same air mass to test transfer to field conditions.

The CSV tables contain [ranked retention effects](../../../../website/public/study-data/mam4xx/priorities.csv) and [time-resolved paired number and dry-volume differences](../../../../website/public/study-data/mam4xx/paired-effects.csv).
