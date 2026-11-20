# Recovering initial size after closed-box condensation

With known initial vapor, represented counts and transport parameters, twelve
size-selective channels recovered the four initial mode radii in two boxes
to within 0.61 nm. Broad channels admitted markedly different initial
distributions. The [worked CUDA example](infer_initial_radius.py)
contains both experiments. All observations in this study are synthetic.

## What the optimizer differentiates

The concrete `particula.execution.inverse_condensation.terminal_radius` function
takes positive radii in µm and number densities in m⁻³ for each occupied
component. Convert native resolved counts to number densities using each box's
volume. The single vapor concentration is in kg/m³. In each fixed step, particle
mass per physical particle increases by the Fuchs–Sutugin corrected flux at unit
accommodation. Remaining vapor equals initial vapor less the concentration-weighted
mass gains. The code bounds the timestep using the maximum attainable radius so
the vapor stays positive. Its inputs describe isothermal growth of one material
with zero surface vapor. An optional output adjoint returns the derivative of
a weighted sum of final radii with respect to the initial radii.

The independent radius ODE uses DOP853 and Radau. A third calculation integrates
time as a function of radius by quadrature. Ordinary native CPU stepping uses
the mapped vapor in the second gas lane, an inert gas, an inactive particle slot
and a 0.37 m³ box. Radius and vapor agreed with the optimization path at the same
Euler timestep to relative tolerance 10⁻¹¹. The inert lane and inactive slot
retained their initial values. Against the continuous reference the maxima over
initial radii of 0.8, 1.0 and 1.2 µm and both backends were:

| Steps to 1.4 s | Radius error (nm) | Vapor error (kg/m³) | Relative slope error |
| ---: | ---: | ---: | ---: |
| 16 | 1.176 | 2.78 × 10⁻⁸ | 0.00693 |
| 64 | 0.293 | 6.90 × 10⁻⁹ | 0.00171 |
| 256 | 0.074 | 1.73 × 10⁻⁹ | 0.000426 |

The discrete gradient regression perturbs each initial radius by 10⁻³ and
10⁻⁴ µm. It checks both a terminal radius and a nonzero squared-error loss.
It detects the negative cross sensitivity of two modes sharing vapor and the
zero sensitivity between boxes. The discrete comparison uses relative tolerance
10⁻⁴ and absolute tolerance 10⁻⁹ in the respective derivative units.

## Recovery, noise and resolution

On a 41-point grid the scalar reference had positive slopes from 0.748 to
0.857 µm/µm. Bracketing and bounded least squares from three starts recovered
all five held-out truths to within 2 nm. With 10 nm Gaussian observation noise,
the middle three cases had 12–13 nm spread in the inferred initial radius.
In 256 replicates per case, inversion of the nominal 95% observation interval
covered the true radius 93.8–97.3% of the time. Near the two bounds, 35–39% of
fits landed on a bound. Bounds and prescribed initial vapor both affect these
uncertainties.

The distribution example fixes two known counts per box and fits their radii
in ordered intervals, with no smoothness penalty. It uses the sum of squared
channel residuals scaled by 0.01 of box count. Search bounds and the two-mode
representation are prior information. At 32 steps all four optimizers reached
the same noiseless error target. Each counted forward evaluation covers both
boxes. The global search evaluates its trial population in a single batch.

| Optimizer | Forward evaluations | Reverse evaluations | Seconds |
| --- | ---: | ---: | ---: |
| L-BFGS-B with Warp adjoint | 26 | 13 | 0.297 |
| Finite-difference least squares | 30 | 0 | 0.049 |
| Powell | 2672 | 0 | 2.689 |
| Differential evolution | 1800 | 0 | 0.110 |

These are single elapsed times for complete fits in execution order on the RTX
5060, including transfers and first use where applicable. Reference observations
and post-fit sensitivity diagnostics are outside the timing. All fits converged.
Finite differences took the least time for four unknowns on this run.

With one fixed 1% noise realization, widening the channel standard deviation
from 0.04 to 1.5 µm increased the largest radius error from 3.6 to 94.2 nm.
The smallest singular value of the noise-scaled observation Jacobian fell from
about 521 to 5.2 per µm. In the second box, 40% of the inferred population moved
into the truly empty 0.95–1.05 µm band. A converged fit therefore needs a
resolution check before it can support a claim about an empty band.

## Reverse memory, graph replay and stochastic processes

At 128 boxes and 128 steps, saving every time level used 803840 bytes at the
Warp CUDA pool high-water mark. Saving every eighth level and recomputing the
intervening states used 134144 bytes. Both schedules reproduced the radii and
adjoints. The table in the data file also covers two boxes and 32 steps in
fresh processes. Pool allocation includes input, output and adjoint buffers.
It excludes context, driver and Python memory.

A separate native CUDA regression captures eight condensation steps, a
per-box Gaussian observation draw, the loss and its reverse pass. Three
replays advance independent per-box counters. Storing the draw before recording
the loss on the tape lets forward and backward use the same observation. The
test compares each replay with ordinary launches and independently perturbed
initial radii with identical draws. This qualification covers that direct
graph. It does not establish checkpoint continuation or differentiation of a
sampled collision.

**Representation decision.** Use resolved size components for this smooth
condensation inverse. For future coagulation inversions, evaluate a deterministic
size-by-composition distribution against resolved ensemble observables. As a
first check, a constant-kernel discrete Smoluchowski solution predicted the mean
numbers of monomers, dimers, trimers and larger clusters within 1.15 Monte Carlo
standard errors across 256 independently simulated boxes of 256 initial
monomers. This is an exact pair-hazard reference experiment for a size-independent
kernel. Brownian, charged and composition-dependent mixing require their own
validation. No collision gradient enters the reported retrieval. A mixing-state
or hygroscopicity inversion additionally needs species-resolved observations,
a response model and demonstrated sensitivity to material fractions.

## Reproduction

Use the development checkout with the Stage 5 implementation at `9fa5880745`,
its Python environment and an actual CUDA device. The qualification data include
source hashes, measured memory and full refinement rows. The example bundle
records the executable inputs, exact device and software versions.
[Qualification data](https://github.com/uncscode/particula/blob/main/docs/data/numerics/initial-size-inversion.json)

```bash
python -m scripts.inverse_studies.qualification .artifacts/stage5/qualification.json
python scripts/docs_examples.py refresh infer-initial-radius
python -m pytest -q -Werror --no-cov particula/execution/tests/inverse*test.py
```
