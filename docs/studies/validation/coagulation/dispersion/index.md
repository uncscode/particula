# Van der Waals and dielectric coagulation validation

The CPU strategy and direct and prepared CUDA solvers evaluate a complete
thermal encounter kernel for homogeneous spheres in air. The strongest
experimental agreement in this comparison is for unequal neutral salt
particles. The equal-size acid measurements show a larger size-dependent
residual. An independent trajectory calculation found a 13.8% underestimate
for a strongly attractive opposite-charge pair in the transition regime.
This result sets a boundary on the supported charge combinations.

## Model and operating range

Supply a nonnegative effective Hamaker constant $A$ in joules and a common
relative permittivity $1\le\epsilon_r\le100$. Each pair requires radius
ratio at most 32, $A/k_BT\le200$, and
$\max(z_i^2,z_j^2)\ell_B/R\le60$, where $R=a_i+a_j$ and
$\ell_B=e^2/(4\pi\epsilon_0 k_BT)$. Opposite-sign pairs additionally require
$z_i z_j\ell_B/R\ge-0.5$. The implementation raises a domain error outside
these limits. They bound the implemented calculation. Physical evidence
within this range comes from the individual cases below. Material constants
are fixed throughout a run, including after a merger.

- The CPU supports discrete bins, radius PDFs and particle-resolved populations.
  Bin populations retain their prescribed charge. Resolved particles carry
  signed charge and merge each species separately.
- Native CUDA supports equal positive counts within each box of a
  particle-resolved population. Admission and sampling use the same actual
  pair law, including its sum with settling or turbulent shear. Preparation
  transfers the material response table once. Restart checks its version and
  material inputs. The CPU resolved path allocates a dense pair matrix.
- Setting $A=0$ removes dispersion. With unit permittivity and zero charges
  the control is the dimensionless hard-sphere transition law. Its continuum
  and ballistic limits follow analytically. The dimensionless interpolation
  differs from the Fuchs interpolation in the ordinary Brownian strategy.

The [theory page](../../../../theory/dynamics/interparticle-forces.md) defines
the potential, transport limits and assumptions. Sceats gives an alternative
all-regime treatment. Near-contact hydrodynamic resistance, retardation,
molecular-scale structure and composition-dependent optical properties would
change the interpretation of an inferred Hamaker constant.
[Sceats (1989)](https://doi.org/10.1016/0021-9797(89)90419-0) ·
[Alam (1987)](https://doi.org/10.1080/02786828708959118)

## Numerical verification

The 44 material and geometry cases sample radius ratios 1, 3.17, 5 and
31.9. Controls include hard spheres, pure Coulomb capture, a charged sphere
on either side of a neutral partner, repulsive barriers, and weak and strong
dispersion. Analytic checks also cover the dipole and dispersion far-field
powers, charge-sign reversal and pair exchange.

| Quantity | Reference and result |
| --- | --- |
| Continuum enhancement | Adaptive diffusion-resistance integral. Maximum relative difference across 44 CPU cases was 0.056%. |
| Ballistic enhancement | Adaptive search for critical impact parameters followed by Maxwell-flux quadrature. Maximum relative difference was 0.267%. |
| Reference convergence | Six equal or highly unequal charged pairs were refined from multipole order 160 to 224 and 256 to 512 radial intervals, with tighter orbital quadrature. The largest change in either enhancement was 0.024%. A quadrature roundoff warning occurred at the tightest setting. These solvers share the electrostatic boundary equations. |
| Dimensional CUDA kernel | All 132 comparisons at diffusive Knudsen numbers $10^{-5},1,10^5$ passed. Maximum relative error for rates above $10^{-8}$ of hard spheres was 0.029% (criterion 1%). Below that threshold, maximum absolute error divided by the hard-sphere rate was $2.6\times10^{-16}$ (criterion $10^{-10}$). |

[CPU limiting factors](https://github.com/uncscode/particula/blob/main/docs/data/coagulation/van-der-waals-reference.json) ·
[Reference refinement](https://github.com/uncscode/particula/blob/main/docs/data/coagulation/van-der-waals-refinement.json) ·
[CUDA pair grid](https://github.com/uncscode/particula/blob/main/docs/data/coagulation/van-der-waals-cuda.json)

## Independent transition-regime trajectories

A relative-coordinate Langevin solver followed inward Maxwellian flux from a
finite outer sphere to either contact or escape. An exterior diffusive
resistance accounts for supply to the outer boundary. The solver uses the
force profile directly and estimates absorption from hit counts. The campaign
used 524,288 launches per case at three bath-radius and timestep settings,
20,447,232 trajectories in total. Zero trajectories reached the time cutoff.
Binomial sampling uncertainty is conditional on the chosen force and
boundary model. Varying outer radius from 5 to 8 contact distances and
halving the step scale tests sensitivity to the numerical boundary and step.

| Pair and conditions | Model minus trajectory rate, relative to trajectory rate at finest setting |
| --- | --- |
| Hard spheres, $q=1$ | +0.7% |
| Neutral with dispersion, equal and 4:1 sizes, $q=1$ | +2.9% and +2.8% |
| Charged–neutral, equal and 3:1 sizes, $q=1$ | +2.3% and +3.2% |
| Like charges, equal sizes at $q=1$ and 3:1 at $q=2$ | −0.6% and −2.3% |
| Opposite charges, bare attraction $0.5\,k_BT$, equal or 4:1 sizes, $q=0.5,1,2$ across four cases | −3.0% to −1.0% |
| Opposite charges, bare attraction $1.5\,k_BT$, equal sizes at $q=0.5$ and 4:1 at $q=1$ | −8.3% and **−13.8%** |

Here $q$ is the unmodified diffusive Knudsen number. Interacting cases used
$A/k_BT=12$ and $\epsilon_r=5.9$. The finest-setting full 95% sampling
intervals span 2.1–6.9% of each rate. The stronger unlike unequal pair misses
the 10% physical criterion under all three numerical settings. Intermediate
strengths between 0.5 and 1.5 remain unresolved. The implementation uses
0.5 as a conservative bound.
[Trajectory inputs, seeds, hit counts and intervals](https://github.com/uncscode/particula/blob/main/docs/data/coagulation/van-der-waals-trajectories.json)

## Chamber measurements

The [CPU example](van_der_waals.py)
plots absolute rates and residuals with the experimental bars. Residuals
below are $100(K_{model}/K_{measured}-1)$. The input file identifies
the figure, axis calibration, unit conversion and digitizing precision.

| Data series | Mean residual | Range | Predictions inside plotted bars |
| --- | ---: | ---: | ---: |
| Chan and Mozurkewich Fig. 5, equal-size acid droplets | −15.0% | −46.8% to +21.9% | 10 of 23 |
| Boren et al. Fig. 2d, 20 nm neutral NaCl probe | −0.1% | −5.2% to +5.7% | 3 of 4 |
| Boren et al. Fig. 4a, 20 nm charged probe, larger neutral partner | −6.6% | −9.2% to −2.7% | 2 of 5 |

The acid series uses the distinguishable data points and the authors'
fitted $A=(6.4\pm2.6)\times10^{-20}$ J. Some markers overlap. Converting
their identical-particle dimer-production coefficient requires
$K=2k_c\times10^{-6}$ from cm³/s to m³/s. The example propagates the
reported Hamaker range and assumes a droplet density of 1650 kg/m³.
This comparison reuses a parameter inferred from those measurements.
The two Boren series use a dry NaCl material scenario, 2160 kg/m³,
$A=8.9\times10^{-20}$ J and $\epsilon_r=5.9$. Treat that material assignment
for Fig. 4a as an assumption. Plotted bars and digitization have different
sources of uncertainty. The comparison does not determine a transferable
material constant. [Chan and Mozurkewich (2001)](https://doi.org/10.1016/S0021-8502(00)00081-1) ·
[Boren et al. (2026)](https://doi.org/10.1016/j.jaerosci.2026.106780) ·
[Cai et al. (2022)](https://doi.org/10.5194/acp-22-11529-2022) ·
[Digitized values](https://github.com/uncscode/particula/blob/main/docs/data/coagulation/van-der-waals-measurements.json)

Cai et al. provide five tabulated NaCl rates for 3.6–10 nm particles
colliding with a 100 nm partner. Applying the same neutral salt scenario at
298 K and 1 atm gives absolute rates 37–44% below the measured values.
Particula predicts enhancements of 1.21–1.31 relative to its hard-sphere law.
Using the paper's tabulated Knudsen numbers and fitted limiting factors in
Eqs. S7–S14 gives 1.44–1.51 with the Alam interpolation and 1.50–1.56 with
Sceats. Those calculations use different limiting factors and hard-sphere
baselines. This disagreement matters when using the mechanism for a
sub-10 nm coagulation sink. Particle shape, material parameters and charge
are among the uncertainties described by the authors.
[Cai et al., Fig. S1 and Table S1](https://doi.org/10.5194/acp-22-11529-2022-supplement) ·
[Five reproduced cases](https://github.com/uncscode/particula/blob/main/docs/data/coagulation/van-der-waals-cai.json)

## Resolved stochastic execution

At 298 K, 101325 Pa and a volume of $10^{-13}$ m³, the 150 and 220 nm
oppositely charged spheres give $K=1.7143\times10^{-15}$ m³/s. At one
initial collision time, the continuous-time merger probability is 0.6321.
For this two-slot sampler the finite-step result is $1-(1-1/n)^n$.

| Steps | Finite-step probability | CPU, 1,024 boxes | CUDA, 2,048 boxes |
| ---: | ---: | ---: | ---: |
| 8 | 0.6564 | 0.6514 | 0.6748 |
| 32 | 0.6379 | 0.6494 | 0.6479 |

Each realization conserved both tracer masses and signed charge. Native
CUDA graph replay from the midpoint matched every final box. Direct
preflight matched the CPU pair hazard. The example records preparation and
synchronized resident times separately. Compilation and table cache warmth
affect these timings. [CPU ensemble](https://github.com/uncscode/particula/blob/main/docs/data/coagulation/van-der-waals-cpu-ensemble.json) ·
[Prepared CUDA example](gpu_van_der_waals.py)

## Reproduction

Run from the checkout with its scientific environment active. Use native CUDA
for the dimensional grid, trajectory experiment and prepared example. The
saved example manifests identify the source revision, device, dependencies
and input hashes. The recorded GPU was an NVIDIA GeForce RTX 5060 with
Warp 1.11.1, CUDA toolkit 12.9 and driver 13.4.

```bash
export OPENBLAS_NUM_THREADS=1
d=docs/data/coagulation/van-der-waals
python -m scripts.coagulation_validation.accuracy "$d-reference.json"
python -m scripts.coagulation_validation.refinement "$d-reference.json" "$d-refinement.json"
python -m scripts.coagulation_validation.cuda_accuracy "$d-reference.json" "$d-cuda.json"
python -m scripts.coagulation_validation.langevin "$d-trajectories.json"
python -m scripts.coagulation_validation.published_rates "$d-cai.json"
python -m scripts.coagulation_validation.cpu_ensemble "$d-cpu-ensemble.json"
python scripts/docs_examples.py refresh van-der-waals
python scripts/docs_examples.py refresh gpu-van-der-waals
```
