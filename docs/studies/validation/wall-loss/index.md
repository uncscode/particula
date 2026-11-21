# Adding settling and diffusion losses predicts too few airborne particles in a spherical chamber

![Identical spherical chambers start with equal numbers of 1 µm particles. After an hour, coupled theory retains about 68% and the additive approximation retains about 58%. Starting from 50 in each chamber, 34 versus 29 dots remain after rounding. The extra row of five makes the difference visible; the dots are a tally.](../../../images/spherical-wall-loss/graphical-abstract.svg)

For 1 µm particles in the reference chamber, the additive approximation leaves **58.2%** airborne after one hour. Coupled spherical deposition theory leaves **68.4%**. Particula’s CUDA wall removal simulation gives **68.55% ± 0.27 percentage points** (95% Monte Carlo interval). This tests agreement with theory under shared diffusion and settling inputs. Independent chamber decay measurements would test the transport assumptions.

Crump and Seinfeld derived how diffusion and settling jointly remove aerosol in a mixed vessel. The reference case uses neutral water-density spheres of fixed size, a chamber radius of 0.30 m, 298.15 K, 101325 Pa and a wall mixing coefficient of 3.6 s⁻¹. A chamber analysis using the additive correction could therefore
assign too much disappearance to wall deposition under these assumptions.

<a id="the-comparison-tests-a-published-coupling-not-measured-chamber-decay"></a>

## Diffusion and settling share a size-dependent wall loss rate

The authors' [Eq. 13 and Eq. 17, PDF page 39](https://thesis.caltech.edu/3500/3/crump-jg_1983.pdf#page=39)
provide the surface integral and its Debye-function form. For settling speed
v [m/s], Brownian diffusion coefficient D [m²/s], sphere radius R [m], and
wall mixing coefficient kₑ [s⁻¹], define x = πv/(2√(kₑD)). The coupled rate is
β = 6√(kₑD)D₁(x)/(πR) + 3v/(4R), where D₁ is the first Debye function.
Our baseline sets D₁=1, adding the zero-settling diffusion rate to the
zero-diffusion settling rate. It loses the coupling through x and overpredicts
loss. This is an illustrative approximation, not the authors' recommended model.
Independent angular quadrature evaluates Eq. 13; Particula supplies D and v.

The 0.30 m radius, water density and kₑ=36 s⁻¹ follow the authors' Fig. 2.
Temperature, pressure, one-hour decay, fixed nonvolatile sizes, and kₑ=0.36
and 3.6 s⁻¹ are explicit analysis choices. No measured points were digitized
and no parameters were fitted. Condensation, coagulation, charge, ventilation
and inertial deposition are absent. The new contribution is the quantified
survival bias and its sampling/refinement evidence, not discovery of coupling.

## CUDA ensembles resolve the ten-point difference

For the central 1 µm, kₑ=3.6 s⁻¹ case, 16 independent CUDA boxes with 8192
unit-count particles each gave **68.546% survival ± 0.266 percentage points**
(95% Student-t interval for the ensemble mean). This interval describes Monte
Carlo sampling only. It does not cover uncertainty in chamber physics or inputs.

[View the survival figure and run the CUDA comparison](spherical_wall_loss_theory.py)

*Original synthetic predictions: solid curves are independent surface quadrature;
dashed curves add the two limits; points are CUDA ensemble means with 95%
Monte Carlo intervals, often smaller than the symbols. Radius is 0.30 m and
particle density 1000 kg/m³. These are not measurements.*

| Wall mixing coefficient [s⁻¹] | Largest survival difference in scan [percentage points] | Diameter at scanned maximum [µm] |
| --- | ---: | ---: |
| 0.36 | 5.588 | 0.659 |
| 3.6 | 10.175 | 0.986 |
| 36.0 | 13.361 | 1.419 |

The maximum changes with mixing because diffusion and settling compete at
different diameters. These are sampled maxima over 0.03–3 µm, not global
optima. The coarse end is particularly subject to the paper's inertia limits.

## Refinement shifts are smaller than the principal approximation error

The campaign completed 12 physical conditions, each with 16 independent boxes,
for a 2048-particle baseline, a halved timestep (30 to 15 s), and an
8192-particle refinement: 576 boxes plus the 4-box pilot. Independent root
seeds are recorded per run; refinement differences include sampling noise.
All 36 ensemble means met the predeclared five-binomial-standard-error plus
0.002 absolute gate against the reference. Maximum mean/reference difference
was 0.418 percentage points. Halving the timestep shifted means by at most
0.504 points; increasing particle number shifted them by at most 0.359 points.
This supports resolution of the central ten-point effect, not a zero timestep
error claim for all cases or resolution of every small effect in the scan.

Doubling the reference diameter grid from 121 to 241 points changed the largest
survival difference by at most 0.009 percentage points. CPU Particula wall-loss
coefficients differed from quadrature by at most 0.1001%, within the preset
0.5% gate. This finite coefficient approximation is separate from Monte Carlo
uncertainty. All retained mass states were finite/nonnegative; remaining mass
fraction and particle survival fraction agreed within 3.34e-16. Wall deposition
removes mass, so total airborne mass is not conserved.

## Measure decay near 1 µm before adopting an additive correction

For chamber studies needing percentage-point survival accuracy, the additive
baseline is inadequate in the tested crossover regime. The next test is
size-selected decay near 0.3, 1 and 3 µm in a sphere with independently
characterized mixing, charge and ventilation, comparing observed rate constants
with both predictions. An independently determined mixing coefficient is
important: fitting it separately to each diameter could conceal the effect.
A useful target would be survival precision below two percentage points, an
experimental design target rather than an established instrument uncertainty.

The main unresolved uncertainty is whether a real chamber meets the homogeneous
mixing, thin boundary layer and negligible inertia assumptions. No experimental
uncertainty interval is available here. A calibrated flow/mixing relation and
inertia check should precede quantitative interpretation of the coarse end.
Particle-resolved sampling does not add missing chamber physics. No defect was
diagnosed and no issue was filed. Documentation publication was approved by the researcher on 2026-11-19.
That approval does not establish experimental validation.

## Reproduction and provenance

The original automated independent analysis ran at revision
`5ec711e600fa21d81a89f19d0bfa5e29ac89c68e` with source Particula 0.3.2,
Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2 and Warp 1.11.1.
All 580 boxes ran on NVIDIA GeForce RTX 5060 (`cuda:0`, 8 GiB), with
119.09 seconds total pilot/campaign wall time including setup and observations.
The initial sandbox GPU probe failed; host CUDA execution succeeded.
Installed distribution metadata reported 0.2.14, but the actual imported
source and execution revision were verified. This is source-version evidence.

[Original ensemble summaries](https://github.com/uncscode/particula/blob/main/docs/data/spherical-wall-loss/summary.csv),
[refinement checks](https://github.com/uncscode/particula/blob/main/docs/data/spherical-wall-loss/analysis.json), and the
[frozen specification](spherical-wall-loss-specification.md)
preserve the reviewed numbers and checks. The
[worked example](spherical_wall_loss_theory.py)
contains the executable reference, pilot, full ensemble and refinements.
Its saved results retain the publication rerun's own version, device, execution
revision and script hashes. The initial study and rerun are distinct executions.

From a checkout with the documented dependencies and CUDA:

```bash
python scripts/docs_examples.py refresh spherical-wall-loss-theory
python scripts/docs_examples.py check --changed --base upstream/main
```

To redraw verified saved numerical data without CUDA or simulation:

```bash
python scripts/docs_examples.py replot spherical-wall-loss-theory
```

The downloadable example also runs directly with Python. No paper text,
measurement dataset or source figure is redistributed. Figure curves are
original calculations credited to the authors' equations. The script assertions
check coefficient accuracy, finite/nonnegative masses, survivor inventory and
Monte Carlo agreement without silently substituting CPU for CUDA.
