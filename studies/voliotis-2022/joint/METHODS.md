# How to reproduce a joint chamber mechanism fit

## Choose which experiments can answer the question

Fit suspended organic aerosol, excluding the inorganic seed, in all four Manchester α-pinene experiments with both an AMS record and an archived precursor fraction. Run 1 began at nominally 309 ppb, runs 4 and 5 at 155 ppb, and run 6 at 103 ppb. Treat run 2 separately: it has a complete AMS record and the same nominal loading as run 1, but its precursor instrument failed. In the five-run fit it receives its own seed, temperature, initial number and AMS target, and borrows run 1's fractional precursor history. Run 3 has no organic-mass trajectory. The remaining 18 experiments concern isoprene, o-cresol or mixtures. The final section uses additional precursor-specific pathways and measured forcing wherever those time series exist. [Paper and Table 1](https://acp.copernicus.org/articles/22/14147/2022/) · [Supplementary experiment inventory](https://acp.copernicus.org/articles/22/14147/2022/acp-22-14147-2022-supplement.pdf) · [Four fully paired runs](../data/README.md) · [Run 2 data](data/README.md)

All uses of aerosol observations in the optimization are in-sample. Each α-pinene five-hour curve contributes the 84 original AMS values through 4.98 h. The model saves a six-minute trajectory that is interpolated to those timestamps. The six-hour AMS records outlast the five-hour gas forcing. The earlier study explores two extrapolations of the missing final gas hour. No fit here uses that final hour. [Forcing and wall-loss treatment](../METHODS.md)

## Construct the candidate chemical mechanisms

Gas chemistry is expressed in C₁₀ equivalents. Oxidation of a condensable-branch precursor consumes one O₂ equivalent per C₁₀H₁₆ and creates an intermediate I of molar mass 168.236 g mol⁻¹. The rate uses the interval-specific loss of measured precursor. The remaining precursor enters a persistent gas reservoir. The fitted molar branch fraction *f* controls how much of the reacted precursor can supply particles. Even with an atmospheric O₂-equivalent balance, this scheme cannot tell which oxidant consumed the α-pinene.

- **One product.** I produces a semivolatile vapor S with mean lag τ. S exchanges reversibly with the seed and suspended organics and can be lost to a vapor-wall reservoir.
- **Two products with aging.** I branches to S and a lower-volatility vapor L. A first-order S → L gas reaction changes the distribution as oxidation proceeds. Both vapors condense, evaporate and compete for the same absorbing mass. The effective saturation concentration of L is that of S divided by the fitted volatility ratio.
- **Competing chemistry.** A second-order channel 2 I → 2 L competes with the first-order routes. This is a balance of low-volatility C₁₀ equivalents. It tests nonlinear production without assigning molecular dimer identities. Concentration-dependent peroxy reactions and measured shifts in particle-phase volatility motivate these alternatives. The fitted coefficient cannot identify a molecular reaction from these data alone. [α-Pinene peroxy chemistry](https://doi.org/10.1073/pnas.1812147115) · [Same-campaign particle volatility](https://acp.copernicus.org/articles/22/13677/2022/)

The NOₓ experiments add a shared condition-response law, *logit(f) = logit(f₃₀) + β ln(NOₓ₀ / 30 ppb)*. One slope and one intercept apply to every run. Initial NOₓ from Table 1 (40, 26, 35, 18 and 43 ppb for runs 1, 4, 5, 6 and 2) is a proxy for conditions, not a time series of NO or RO₂. Conflicting archive values for runs 4 and 6 are evaluated with the fitted law held fixed. A nonzero slope can absorb unmeasured co-variation among the few chamber runs.

One concrete four-run recipe uses the `single_nox` vector. Set the condensable branch at 30 ppb NOₓ to 0.30399 and its log-odds slope to 0.9260. The implied fractions are 0.3631, 0.2767, 0.3350 and 0.2139 for runs 1, 4, 5 and 6. Use nominal precursor doses and each measured fractional decay to drive the intermediate. Set its mean maturation time to 21.34 min, C* to 1.951 µg m⁻³, accommodation to 0.5548, exponential particle loss to 0.2864 h⁻¹ and vapor-wall loss to 4.793 h⁻¹. Run `native.py single_nox --dt 3 --split 18` with the archived fit for full precision. High accommodation and vapor loss in this solution illustrate why identifying product chemistry from aerosol mass alone is difficult. [Native chemical mechanism](results/single_nox_mechanism.json) · [Full parameter vectors](results/physical_parameters.csv)

## Separate chemical choices from chamber assumptions

The default chamber starts with the paper's seed mass and measured count near lights-on, using a single mass-equivalent particle size. Particle temperature is fixed to the initial chamber temperature, with 24 °C assumed for run 6. Pressure is 101325 Pa. Accommodation, vapor deposition, and particle removal are shared across runs. Vapor transfer uses ideal absorbing mass (seed plus all condensed organics), Fuchs–Sutugin transition correction, Kelvin curvature, diffusivity 6 × 10⁻⁶ m² s⁻¹ and organic density 1400 kg m⁻³. Vapor pressure is set from C* at 298.15 K. The seed density is 1770 kg m⁻³ and surface tension 0.03 N m⁻¹. These are study assumptions. Changes in humidity, water content, activity coefficients, size distributions and release from the chamber walls could alter the inferred chemistry.

In the count-conditioned variants the measured number histories replace a fitted exponential particle loss. A single material-removal share controls the fraction of declining number that carries material out of suspension. The end members are mass-preserving consolidation and removal of complete particles. Number is then an input. For the fitted-dose variants, each run has one multiplier within the authors' ±15% injection bounds. All chemical coefficients stay shared. No extra likelihood or penalty favors nominal doses. Because the model source is proportional to the product of dose and condensable branch, a common rescaling of all doses trades against the common branch within its bounds. Relative dose adjustments matter for the inter-run comparison. [Precursor and count provenance](../data/README.md)

## Optimize the worst-reproduced run

For each candidate, compute each run's RMSE divided by that run's measured peak. Minimize the largest of those normalized RMSEs across the included runs. A 309 ppb experiment therefore cannot dominate the fit simply by producing more mass. Record the absolute RMSE, peak error, peak timing and bias separately. The illustrative reproduction target requires both a normalized RMSE at most 10% and an absolute peak error at most 15% in **each** fitted run. It describes a practical tolerance chosen for this study. The paper provides no pointwise AMS covariance for a statistical equivalence test, and these limits do not define a confidence interval.

| Shared parameter | Range |
|---|---|
| Branch at reference conditions | 0.05–0.85 by mole |
| S saturation concentration at 298.15 K | 0.316–316 µg m⁻³, log scale |
| Vapor accommodation | 0.01–0.562, log scale |
| Intermediate lifetime | 2.00–70.8 min, log scale |
| Particle removal | 0.04–0.65 h⁻¹ or measured number with removal share 0–1 |
| Vapor wall loss | 0–5 h⁻¹ |
| Direct fraction of condensable products entering L | 0–1 |
| S/L saturation concentration ratio | 1–316, log scale |
| Gas aging S → L | 0–4 h⁻¹ |
| Nonlinear removal rate at 10 µg m⁻³ intermediate | 0–12 h⁻¹ |
| Initial NOₓ branch slope, when fitted | −3 to +3 |

Each restart begins with 2,048 scrambled Sobol candidates. The best 32 and 96 dispersed candidates initialize differential evolution, with up to 85 generations of 128 candidates. The broad search uses 18 s steps. Rescore its entire final population and the nested starting vectors at 3 s, then refine the best of these at 3 s using 32 batches of 128 bounded perturbations. The independent direct 3 s search for the first three families is archived as a numerical cross-check using the same Sobol design, population and restart seeds, with the broad evolution also run at 3 s. [Direct fine-step comparison](results/direct_fine_audit.csv) The search uses seeds 1941, 3527 and 7819 for each model. Where appropriate, a simpler fitted model seeds the nested mechanism. Repeat searches and parameter populations expose sensitivity to initialization. This finite search produces feasible witnesses and best-found scores within the stated bounds. It cannot prove that a better fit is impossible.

A second search minimizes the maximum of two error ratios across the included runs, normalized trajectory RMSE divided by 0.10 and absolute relative peak error divided by 0.15. A maximum ratio ≤1 supplies a parameter set meeting both criteria. It starts from two separate scrambled Sobol sets of 1,024 candidates (seeds 1410 and 6004), with a 96-member differential-evolution population for up to 60 generations at 18 s. It reranks the final population at 3 s and refines it over 36 batches of 128 candidates. The saved table includes the fitted vector for each restart. [Direct feasibility searches](results/joint_gate_search.csv)

## Execute and inspect the checks

The search kernel runs one chamber and one candidate per GPU thread. It uses Particula's native Warp mass-transport and transition-correction functions. A conservative two-reservoir update advances reversible exchange with the composition denominator fixed over each substep. Gas production follows the measured loss over each interval. Explicit wall ledgers track both vapor and deposited particles. Concentration-dependent chemistry and first-order transfers are split across the substep. The [native replay](native.py) builds mass-balanced Particula ReactionMechanism records, evolves gas with adaptive RK23 and uses the native isothermal condensation solver. To advance two volatile lanes on CUDA, it alternates two native prepared sequences and transfers their complete chamber state between them. In NumPy it advances both exchanges every substep. These native histories provide an independent numerical comparison with the fast optimizer. All the reported costs refer to in-sample curve fit.

These runs used Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2, pandas 2.3.3, Matplotlib 3.10.7 and Warp 1.11.1 on an NVIDIA GeForce RTX 5060. The imported Particula source identifies itself as 0.3.3 at study base commit `e5f2d731a1`. The environment’s installed distribution metadata reports 0.2.14, so the explicit checkout path in the commands is important. Run from the root of the study worktree with its local environment and NVIDIA device. Bundled measurement CSVs are sufficient.

```bash
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
PY=/home/kyle/Code/particula/.venv/bin/python
JOINT=studies/voliotis-2022/joint
"$PY" "$JOINT/optimize.py"
"$PY" "$JOINT/feasibility.py"
"$PY" "$JOINT/native.py" single expanded single_doses expanded_doses expanded_counts five_doses competition_doses single_nox single_nox_five competition_nox_counts
"$PY" "$JOINT/native.py" single_nox five_doses --dt 3 --split 18
"$PY" "$JOINT/summarize.py"
```

The individual family JSONs include search bounds by reference to the script, the winning physical vector and restart records. The exported native mechanism JSON covers the reaction stoichiometry and rate laws. The chamber initialization supplies the condensed material, dose, branch fraction and time-dependent precursor forcing; the mechanism JSON alone does not specify an experiment. [Global search](optimize.py) · [Numerical engine](engine.py) · [Saved physical parameters by model and run](results/physical_parameters.csv) · [Individual run scores](results/scores.csv) · [Native replay checks](results/native_checks.csv)

For the structural NOₓ–dose equivalence calculation, hold each fitted four-run source strength fixed. Its allowable branch is the source strength divided by a dose multiplier between 0.85 and 1.15. On the log-odds scale each run gives two linear inequalities in reference intercept and slope; two linear programs minimize and maximize the slope subject to all eight inequalities and the original parameter bounds. Every point in the region reproduces exactly the same four suspended-mass curves with all remaining model parameters held fixed. [Equivalence calculation](equivalence.py)

## Fit all seven precursor systems with coincident gas and aerosol records

Fifteen experiments provide a suspended organic mass curve and the fractional decay of every injected precursor. They supply 1,258 AMS values through 5 h or the last overlapping sample. Retain the two measured zero isoprene curves and the signed near-zero observations. At each original AMS time, normalize the residual by max(observed peak, 1 µg m⁻³). For a zero run this gives an absolute-error scale of 1 µg m⁻³ chosen for the calculation. It is not an instrument detection limit. The original 24-experiment design and source discrepancies are documented separately. [Campaign data](data/CAMPAIGN_SERIES.md) · [Complete endpoint inventory](data/CAMPAIGN_PEAKS.md)

The cross-precursor calculation is a reduced, analytically integrated box model. It uses nominal doses, 298.15 K and 101325 Pa. Sample each dimensionless precursor record at 20-minute knots, constrain it to decrease from one with isotonic regression, and interpolate on a one-minute grid. Multiplying the lost fraction by the nominal initial mass yields each minute's mass source. Treat the source as constant within the minute. Each precursor produces an intermediate with a first-order maturation rate. Its matured product enters suspended aerosol at a fitted mass yield. A shared first-order rate removes suspended product. This reduction combines oxidation-product formation and net uptake into an effective yield and delay. The detailed Particula gas/particle exchange and chamber wall reservoirs are evaluated in the α-pinene analysis above.

The additive model fits one mass yield per precursor, a common delay and one loss rate. For the mixture model, split each precursor's lost mass between two channels in proportion to its own and all other precursors' shares of total **reacted mass during that minute**. Each channel has its own fitted mass yield, shared across runs. The six yield coefficients are ordered α-pinene, o-cresol and isoprene in the own channel, followed by the same sequence in the other channel. The third model also fits one delay per precursor. All yields lie between 0 and 2 kg of suspended organic product per kg reacted precursor. This empirical range allows for oxygen addition. No reactions between identified molecules are inferred.

The time integrator solves the linear intermediate and aerosol equations exactly for each piecewise-constant source. At fixed maturation and removal rates the squared error in each run is a convex quadratic in the yields. The optimizer minimizes the largest error across runs. A linear program over supporting planes checks the difference between a feasible solution and a global lower bound at those fixed rates (up to 2 × 10⁻⁷ normalized mean-square error). For a common lag we search a 10 × 11 grid over 5 min–4 h lag and 0.05–3 h⁻¹ removal and refine the three best starting points with a bounded Nelder–Mead search. For the three distinct delays, we include the best common-delay solution, two scrambled Sobol sets of 128 points, and local searches from each set's best two points. The kinetic optima are best found within these bounds.

A final comparison holds each model's selected kinetic rates fixed while refitting its six or three chemical yields and one common dose scale per experiment across all its precursors (0.85–1.15). For a fixed tolerance τ and reciprocal dose vᵢ, each run requires ‖Xᵢ·yield − vᵢ·observation‖ ≤ τ·vᵢ after scaling to normalized RMSE. Those constraints are convex. Bisection and tangent planes give numerical lower and upper bounds on the attainable worst error. The bounds refer to this dose treatment at the given kinetic rates. We also reoptimize the kinetic rates jointly, profiling the convex dose/yield problem for each kinetic candidate. That search includes its fixed-rate start, 64 log-scaled Sobol points (seed 8103), and two bounded local searches. The resulting score bounds apply at the selected rates; the outer search has no global optimality certificate. They do not cover independent changes to each component of a mixture. [Global campaign fit](campaign.py) · [Bounded dose calculation](campaign_doses.py) · [Error bounds per run](results/campaign_dose_scores.csv)

From the same worktree and environment, the following commands reproduce the campaign comparison and the exact-equivalence calculation for the four-run NOₓ fit.

```bash
"$PY" "$JOINT/campaign.py"
"$PY" "$JOINT/campaign_doses.py"
"$PY" "$JOINT/campaign_figure.py"
"$PY" "$JOINT/equivalence.py"
"$PY" "$JOINT/verify.py"
```
