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
