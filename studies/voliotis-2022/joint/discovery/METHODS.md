# Fitting reactions before looking for a chemical analogue

The 20 rounds compare the same 15 Voliotis experiments with complete precursor, suspended organic mass and particle count records. Gas-phase precursor consumption, seed loading and initial particle number are run specific. Every reaction rate, effective yield and formation time is shared across the experiments within a fit. Inputs, wall geometry, Brownian coagulation and the comparison at actual instrument times follow the preceding study. All runs use the local CUDA device. [Original chamber study](https://acp.copernicus.org/articles/22/14147/2022/) · [Previous chemical fits](../../CHEMICAL_PATHWAYS.md) · [Observation inventory](../data/CAMPAIGN_SERIES.md)

## What a fitted reaction means

The network tracks 11 organic mass reservoirs. A, C and I tag effective products of α-pinene, o-cresol and isoprene. Each parent has a delayed gas intermediate G, an oxygenated-vapor reservoir V and a labile suspended product P. Stabilized suspended products S and terminal volatile products F share bulk reservoirs. A channel retains its own product label and gross production total. A tracer tracks how much of its immediate product survives at five hours, including deposition for suspended products, background loss for vapors and consumption by other reactions. Distinct S products have identical density and wall-loss physics. Their inventories can differ even when the simulated mass and count histories agree exactly. The labels describe candidate classes and representative molecules. The chamber instruments did not measure their fitted concentrations.

The reference is the earlier volatile-diversion model with the wider competition bound. It diverts a fraction of the potential α-pinene and o-cresol condensable source according to measured isoprene disappearance. Retained material enters G, then P after its shared parent-specific delay. The new network places the diverted material in the corresponding V reservoir. It also supplies V with 10% of the positive difference between reacted parent mass and modeled gross product supply. This additional source represents a possible supply of unmeasured oxygenated vapors. It was fixed for the primary fits. Background vapor loss transfers V to F at 1 h⁻¹. These choices leave the original mass and count predictions unchanged when the new reaction rates are zero. Final sensitivity runs halve and double the 10% assumption with chemistry fixed and doses reprofiled.

A first-order channel removes mass at k times the source pool concentration, with k in h⁻¹. For the uptake channels, k is the effective rate at 0.001 m² of surface per m³ of air and scales with the evolving geometric particle area. First-order gas oxidation and photochemistry assume an effective oxidant or illumination exposure folded into k. Bimolecular channels use k in m³ of air per µmol per hour and reactant concentrations in µmol per m³ of air. The representative molar masses are 184, 140 and 118 g mol⁻¹ for A, C and I products. Homodimers consume two equivalents per event. A cross-reaction consumes one equivalent of each reactant and transfers their combined effective mass to the product. Even for particle-phase reactions these are rates per air volume. Water content, condensed-phase diffusivity, acidity and actual radical concentrations remain unresolved. None of the fitted constants should be transferred directly to an elementary chemical mechanism.

Each channel transfers available material from its reactants with a nonnegative step. The simulator updates seed size, coagulation and size-dependent deposition as particles grow or shrink. The budget tracks effective organic mass supplied, suspended and gaseous material, deposition and terminal products. Oxygen addition, nuclei and individual carbon atoms are outside this inventory. The study tests source equivalents, phase and kinetic form. Agreement with a literature reaction family establishes plausibility; it does not identify the molecule or establish its rate. [Reaction definitions](reactions.py) · [CUDA integration and product tracing](gpu.py) · [Machine-readable mechanism](../results/discovery/mechanism.json)

## How each round chose, fitted and admitted an equation

1. Refit the incumbent mechanism to the original five-hour records. Screen unused, precursor-consistent candidate equations at a range of nonnegative rates. Rank them by their worst normalized mass error, using mean mass error to break near ties. The search selects from an explicit candidate bank, including competing reactions that never entered the final set. It does not enumerate every molecular graph.
2. Fit the selected equation with the rate free, including zero, and save its equation, score profile and the current best fits. Candidate selection uses the mass objective. A second fit of the same reaction set minimizes the largest mass or number error in any experiment.
3. After the proposal was recorded, a literature reader searched primary experimental work for the process and recorded the named analogue, source, observation and differences from the computed equation. A direct process match requires relevant experimental evidence for a closely corresponding substrate and process. A family match can involve another precursor, partner or phase condition. Neither grade certifies the fitted rate. All 20 selected equations had an analogue at one of these levels.
4. Admit the supported equation, retain the zero-rate alternative, refit existing and new rates and proceed with the resulting mechanism. A retrospective comparison gave the previous network one additional local fit without the new reaction. This separates improvements due to more searching from those obtained by introducing a channel. Per-round scores describe best-found fits under their respective searches.

The first two exploratory screens included transport-only controls. Both selected chemical accretion reactions. From round 3, pure evaporation and condensation were ineligible for a reaction slot. Three extra gas-to-carbonyl or ring-opening candidates were available. The later screens also evaluated candidate additions while adjusting previously fitted parameters using linearized minimax constraints, then scored those suggestions in the nonlinear CUDA solver. This change was made during the study. The saved proposals preserve the actual candidate lists and scores from each round. [Per-round evidence and rates](../results/discovery/REACTION_EVIDENCE.md) · [Screen and fit](fit.py) · [Local rate adjustments](directions.py)

## The score, the control and the remaining uncertainty

The 1,258 AMS and 682 count measurements each enter their experiment’s trajectory RMSE. Mass error is normalized by observed peak with a floor of 1 µg m⁻³. Number error is normalized by initial count. The mass-only fit fixes wall E at 5.1079 s⁻¹. The joint fit allows one shared E within approximately 2.82–16.55 s⁻¹. Both fits allow one dose multiplier per experiment in the original ±15% range. The final discrete comparison profiles 61 doses at increments of 0.005. First-order candidate rates span 0–5 h⁻¹ and bimolecular rates 0–25 m³ µmol⁻¹ h⁻¹. Every objective uses one shared chemical parameter set across all 15 experiments. Mass-only and joint solutions are two separate sets.

After the online sequence, a population of local perturbations and two independent starts revisited the zero- and 20-channel networks. A constrained nonlinear optimizer then varied shared baseline chemistry, all admitted rates and 15 continuous run-specific doses jointly. Each candidate optimum was rescored on the same discrete dose grid. Additional, ordered refits applied that optimizer to each successive reaction prefix, carrying the smaller network forward with the new rate at zero. These retrospective fits show how much of the final change the earlier prefixes can achieve with more optimization. They do not alter the actual decision sequence or the publication matched to an equation. Search iteration limits preclude claims of mathematical optimality.

For final omission experiments the selected rate is set to zero, first holding the other parameters and doses fixed, then giving the other parameters a local refit and reselecting doses. A substitute mechanism from a more extensive search could do better. The rate profiles are conditional sensitivities; neither their widths nor the small between-fit differences are statistical confidence limits. The fine-step, reverse-order, vapor-supply and nominal-dose checks quantify distinct sensitivities. Gross transfer through one channel and the amount of its product remaining in suspension are different observables. [Controls](../results/discovery/ablations.csv) · [Rate profiles](../results/discovery/final_rate_profiles.csv) · [Product inventories](../results/discovery/reaction_flux.csv)

F accumulates transfers to terminal products, including the background vapor loss. Its inventory is cumulative transfer. S tracks stabilized material still suspended after deposition. A matched evaporation substitution redirects the same particle loss to V, allowing subsequent return to particles. Both alternatives retain the original per-run doses. [Physical alternatives](../results/discovery/physical_alternatives.csv)

## Reproduce a result

From the study worktree root use the selected Python environment, Particula source checkout and local CUDA device. Replaying the exported network runs both final parameter sets at a 15 s step against the 15 experiments.

```bash
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
PY=/home/kyle/Code/particula/.venv/bin/python
J=studies/voliotis-2022/joint/discovery
R=studies/voliotis-2022/joint/results/discovery
"$PY" "$J/replay.py" "$R/mechanism.json"
"$PY" "$J/replay.py" "$R/best_seen_mechanism.json"
"$PY" "$J/checks.py"
```

To repeat the selection from the saved literature decisions, specify an empty output directory for the fit checkpoints. The driver checks that each archived decision belongs to the exact newly proposed equation. With a new evidence source, each proposal awaits its own matching decision before the next round. The order among numerically tied candidates can vary across hardware or numerical libraries. Archived per-round states can also be re-evaluated directly with `audit.py`.

```bash
"$PY" "$J/fit.py" --output /tmp/voliotis-reaction-rebuild --evidence-source "$R"
```

For the full retrospective assessment of the archived rounds, run the commands below in order. The random seeds are embedded in the search scripts. The reference environment uses Python 3.12, Particula 0.3.3 and Warp 1.11.1 on a GeForce RTX 5060. [Numerical controls](../results/discovery/numerical_checks.json)

```bash
"$PY" "$J/paired.py"
"$PY" "$J/finish.py" --polish-only
"$PY" "$J/continuous.py"
"$PY" "$J/prefixes.py"
"$PY" "$J/finish.py" --refined
"$PY" "$J/audit.py"
"$PY" "$J/alternatives.py"
"$PY" "$J/summarize.py"
```
