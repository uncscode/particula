# How the shared gas and particle pathways were compared

The response is suspended SOA from the AMS and particle number from WCPC or DMPS. All 15 paired experiments use their own measured precursor disappearance, seed and initial number. Every fit retains neutral Brownian coagulation, the same size-dependent rectangular chamber deposition law and the same representative-size approximation. The time horizon is five hours. The modeled emissions below are effective oxygenated-product mass per mass of reacted VOC. Oxygen uptake and unobserved oxidation products prevent treating this product inventory as an elemental carbon balance. [Original study and forcing](REPORT.md) · [Deposition and counts](WALL_LOSS.md) · [Campaign kernel](wall_campaign.py)

## Chemical hypotheses

The reference feeds each parent's effective product mass into a first-order formation delay shared across runs. Each parent has a yield for its own share of instantaneous VOC consumption and a yield for its pooled partners' share. The delayed product becomes suspended SOA. Its irreversible condensed endpoint already represents a low-volatility limit. The following branches change the amount, timing or destination of the inferred product source. They represent testable limiting behavior of a lumped chemical class. No exact molecular formula or saturation concentration was fitted. [Product formation and source weights](campaign.py)

| Candidate | Change relative to reference | Additional fitted parameters |
|---|---|---:|
| Prompt low-volatility α-pinene product | An additional yield from reacted α-pinene enters particles immediately. | 1 |
| Autoxidation and competition | An α-pinene-derived gas intermediate has two competing first-order fates. One produces a nonvolatile condensable end product. The other produces volatile material whose formation rate tracks isoprene disappearance. | 3 |
| Reactive uptake | Reacted α-pinene and o-cresol produce a common vapor. The evolving particle surface removes it irreversibly over the experiment. A vapor loss competes with uptake. | 3 |
| Autoxidation and uptake | Both gas intermediates operate together with independent effective yields. | 6 |
| Partner-specific yields | Each parent has two distinct mixed yields, one for each partner. The mixed source is allocated in proportion to contemporaneous partner mass disappearance. Own-parent yields and delays remain free. | 3 |
| Diversion to volatile products | Some of the existing α-pinene and o-cresol product source is diverted to a volatile reservoir before it can form SOA. The fraction increases with isoprene disappearance. | 1 |
| Diversion and uptake | The volatile branch acts on the original condensable source while a separate oxidation product can undergo reactive uptake. | 4 |
| Diversion and prescribed uptake | The same channels operate with uptake-vapor yield 0.10 g g⁻¹, effective uptake probability 0.003 and vapor loss 1 h⁻¹ fixed as a positive-uptake sensitivity case. | 1 |

Let I be the smoothed isoprene disappearance rate in ppb min⁻¹ including the run's dose factor. In the autoxidation surrogate a gas intermediate converts to condensable product at 1/τ and to volatile product at bI/τ. The input is a fitted yield times the measured α-pinene mass disappearance. In the diversion model the fraction of the pre-existing α-pinene and o-cresol source retained for SOA is 1/(1+bI). This construction leaves the isoprene-free runs unchanged at fixed chemistry and dose. The proxy I supplies a mixture-dependent rate contrast. It does not measure radical concentrations. Because parent consumption is supplied from observations, attributing the same effect again to consumption of OH by isoprene would count that part of the precursor response twice. [Measurements of oxidant and product scavenging](https://www.nature.com/articles/s41586-018-0871-y)

For uptake, k = γ c̄ A/4, with dry geometric area A = 4πr²N in m² m⁻³ and a representative molecular speed c̄ = 200 m s⁻¹. The alternative vapor loss is first order. The effective γ absorbs accommodation, transport and chemical retention. The model assumes irreversible uptake on the five-hour horizon. Seed acidity, product desorption, gas-side diffusion resistance and particle-phase reactions are unresolved. The two gas reservoirs use the analytic solution for piecewise-constant production and competing losses at each substep. Suspended product encounters the same deposition law whichever route forms it. [α-pinene HOM uptake on seed aerosol](https://doi.org/10.1029/2021JD036414)

## Bounds, objectives and numerical checks

The shared reference parameters retain the previous delay and yield bounds. Added product yields span 0–0.25 g g⁻¹, the gas-intermediate lifetime 0.1–300 s, b from 0 to 10 (ppb min⁻¹)⁻¹, γ from 10⁻⁵ to 0.03 and the vapor loss rate 0.05–5 h⁻¹. The three alternate partner yields each span 0–2 g g⁻¹. These are exploratory ranges. A separate diversion sensitivity repeats the search with upper bounds of 30 and 100 on b. The forcing sensitivity changes the knot spacing of the monotone precursor curves from the original 20 min to 10 and 40 min, holding the selected chemical and wall parameters fixed and reselecting doses. Fits used all 1,258 AMS and 682 count observations and profiled one dose multiplier per experiment over 0.85–1.15. For each mechanism we minimized the worst run's mass error at E = 5.1079 s⁻¹, minimized the worst mass-or-count error at the same E, and then minimized the joint error while fitting shared E within 2.8184–16.5514 s⁻¹. Mass RMSE is divided by observed peak mass with a 1 µg m⁻³ floor. Count RMSE is divided by initial count. The original 10% trajectory target applies to mass. [Machine-readable bounds and environment](results/pathways/design.json) · [Prior wall uncertainty](../ASSESSMENT.md)

Two seeded population searches for each comparison used scrambled Sobol initialization, differential evolution and local perturbations. Constrained local refinement used an auxiliary worst-error variable and one inequality per run. The new search also revisited the reference chemistry. The comparison also retains exact nested solutions, including a zero new-product branch and a zero uptake yield wherever those settings are allowed. The reported comparisons select candidates on a 30 s step and a dose grid spaced by 0.005. Final tables replay these vectors at 15 s with the same dose grid, and compare to a 7.5 s replay. The population searches and refinements supply best-found fits. Several local searches reached their iteration limit. Changes across time steps quantify numerical sensitivity, while fits of different structures quantify model sensitivity. The on/off comparisons remove one channel while retaining the other fitted parameters and reselecting doses. The parent-specific switches keep only α-pinene or only o-cresol diversion and retain the other fitted coordinates. These sensitivity runs complement the full refits of nested models. [Shared chemistry searches](pathways.py) · [Partner-specific fits](pathway_partners.py) · [Local refinements](pathway_refine.py) · [Coupled uptake fits](pathway_hybrid.py)

Across 256 independent reservoir tests the maximum discrepancy from a matrix exponential was 1.91 × 10⁻¹¹ µg m⁻³. The augmented kernel reproduces the archived reference within 2 × 10⁻¹² in normalized trajectory error. Disabling the new branches recovers the reference. Turning up the isoprene diversion at fixed parameters left all eight isoprene-free trajectories unchanged. The GPU ledger checks supplied product against suspended, deposited, still-gaseous, delayed and diverted material at each evaluation. Across the 27 selected fits, the maximum absolute product residual at five hours was 7.25 × 10⁻¹² µg m⁻³. Halving the replay step from 15 to 7.5 s changed the reported run-level normalized errors by at most 0.005 percentage point. [Independent checks](results/pathways/checks.json) · [Checks implementation](pathway_checks.py)

## Reproduce the experiments

From the study worktree root, using the local Particula environment and a CUDA device, run the following. The figure reads the saved numerical results.

```bash
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
PY=/home/kyle/Code/particula/.venv/bin/python
J=studies/voliotis-2022/joint
"$PY" "$J/pathways.py"
"$PY" "$J/pathway_partners.py"
"$PY" "$J/pathway_refine.py"
"$PY" "$J/pathway_hybrid.py"
"$PY" "$J/pathway_bounds.py"
"$PY" "$J/pathway_checks.py"
"$PY" "$J/pathway_results.py"
"$PY" "$J/pathway_forcing.py"
"$PY" "$J/pathway_figure.py"
```
