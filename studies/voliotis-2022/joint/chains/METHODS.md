# Fitting explicit dimer and daughter-product life cycles

## Inputs and comparators

We used the same 15 Voliotis chamber experiments, five-hour windows, measured precursor disappearance, 1,258 AMS organic mass points and 682 particle count points as the earlier study. Each run has one adjustable injection factor within ±15%. A common set of chemical rates and yields applies across all 15 runs. The mass-only calibration keeps the wall eddy coefficient at 5.1079 s⁻¹. The joint calibration can adjust that coefficient from 2.818 to 16.5515 s⁻¹. Particle size, deposition and coagulation evolve through the previous Particula chamber calculation. A reaction changes suspended material and size. This network has no particle-number source. Mass error uses RMSE divided by the observed peak with a 1 µg m⁻³ floor. Count RMSE is divided by initial number. Each objective minimizes the largest applicable error across runs. [Data and discrepancies](../data/CAMPAIGN_SERIES.md) · [Inherited calibrations and source assumptions](../discovery/METHODS.md)

The starting chemistry and two rate sets come from the earlier 20-channel mechanism, including its best zero-rate control. With all additional rates set to zero, relabelling particle products reproduces both archived objectives and per-run outputs. We monitored the effective organic mass inventory with wall deposition included, nonnegative product pools, and the residual of total stable aerosol less its explicitly tracked particle components. [Numerical checks](../results/chains/checks.json)

## Five product families can react again

| Family | Material routed from the prior mechanism |
|---|---|
| AA | α-pinene dimer from gas accretion or particle-phase coupling |
| AC | Mixed α-pinene/o-cresol gas accretion product |
| AU | Particle product formed by uptake of α-pinene-derived vapor |
| CC | o-Cresol dimer from gas accretion or particle coupling |
| II | Isoprene dimer from gas accretion or particle coupling |

For each family we track four distinct mass inventories in µg m⁻³ of air. D is the original dimer or uptake product, W its further-oxidized particle product, L a retained cleavage residue and U a released gas daughter. F collects terminal volatile products. Families retain their origin through the W, L and U stages. W also receives reoxidized vapor and residue from that family. Its entire inventory therefore represents oxidized particle material of that origin, and several routes may feed it. Gas- and particle-formed dimers of the same parent share one D inventory, with identical effective downstream reactivity. Each family remains a surrogate for potentially several molecules. Oxygen addition, detailed stoichiometry, particle diffusion and oxidant concentrations are unresolved. Particle products share the original organic density and wall deposition physics. Every U has the same background loss of 1 h⁻¹ to F. We compare alternative branching fractions for steps with simultaneous release of U and formation of L. The default divides transferred *organic mass* equally. These fractions are assumed mass yields. The model conserves organic mass equivalents, excluding oxygen gained or lost from the chemical environment. [Network and all trial definitions](bank.py)

Eight motifs per family give 40 ordered trials. A motif can activate a pair of successive steps together, including an intermediate whose formation alone leaves bulk mass unchanged. Repeated steps reuse the same rate. Across the 40 motifs there are 50 distinct added first-order transfers, with effective rates bounded between zero and 5 h⁻¹. The full network has 70 channels. The 20 additional inventories bring the state to 31 entries, including the aggregate suspended pool used for physical properties. The split-off inventories are counted once in the mass ledger. [Complete equations and bounds](../results/chains/manifest.json)

## Searching the chemistry and controlling for refitting

The motif order was specified before inspecting the 40 scores. Candidate rates and all previously admitted rates can equal zero. We keep all candidates available in later rounds so a later reaction can activate a previously invisible intermediate. At each round we fit the mass and joint objectives independently. We give the newly opened reaction rates coordinated starts and allow weak or dormant upstream formation to turn on. We approximate the response to shared chemistry, wall mixing where applicable and the 15 doses by finite differences. A bounded linear minimax solve proposes parameter changes, and we compare them with the actual nonlinear CUDA solver. Rates, yields, per-run dose choices and all incumbent scores are saved for each round.

Each added motif competes against a refit of the immediately preceding network from the same incumbent. A parallel control spends all 40 rounds optimizing only the original 20 reactions. The previous solution and both controls remain eligible at every round. The two-step fit gives null and expanded models the same number of local tangent refinements, with extra kinetic starts for the added channels. This is an ordered, bounded search among explicit mechanisms. The result depends on topology, initial conditions and optimizer effort. [Search implementation](search.py) · [All 40 paired outcomes](../results/chains/rounds.csv)

After selection we independently optimize four networks: the original 20; those 20 plus the α-pinene dimer candidates; the union of steps from the literature-family-supported motifs; and all candidates. We project retained round fits into each network and profile each starting point. Two bounded SLSQP starts then vary shared chemistry, 15 continuous doses, and the minimax error bound, for up to 45 iterations per start. Because several starts reached their iteration cap, we gave each network and objective one further 60-iteration start, including the improved narrower solutions when they were nested. We retain both optimizer checkpoints. [Initial two starts](../results/chains/comparisons.json) · [Further matched optimization](../results/chains/continued.json)

The final comparisons again profile the same 61-point dose grid from 0.85 to 1.15. Broader networks can retain a narrower network as an exact zero-rate control. Time-step sensitivity was checked at 15, 7.5, 3.75 and 1.875 s; forward and reversed chemical process order were compared at 15, 3.75 and 1.875 s. Rates remained fixed. Omission and branching perturbations freeze chemical rates. The branching and order tests reprofile doses. Single-family omissions, dimer perturbations and the two-rate interchange hold doses fixed. These diagnostics measure sensitivity around the fitted solutions. [Optimization controls](compare.py) · [Branching and step sensitivity](../results/chains/sensitivity.csv) · [Dimer perturbations](../results/chains/counterfactuals.csv)

For a resolution-checked export we select the calibration with the smallest maximum score over forward and reversed order at 1.875 s, separately for each objective. A narrower calibration can run in the full bank by setting other rates to zero. This choice only compares the four calibrated candidates. [Resolution-checked mechanism](../results/chains/resolution_checked_mechanism.json)

The evidence table was developed independently of the scores. A complete modeled sequence may have a family analogue, partial evidence or unresolved chemistry. All 40 hypotheses were evaluated numerically, and a separate fit restricts the extra channels to steps used by the 15 motifs with family-level support. This separates empirical fitting scope from the evidence for transferring a molecular assignment or rate. [Sources and evidence grades](EVIDENCE.md)

## Reproduce the saved result

From the study worktree root, set the source checkout on the Python import path and use the selected Python environment with Warp and an accessible CUDA device. The development checkout reports Particula 0.3.3; the run used Python 3.12 and Warp 1.11.1. The mechanism replay reconstructs all channel specifications, loads each parameter vector, and checks both scores against the saved fit. The original campaign files are required.

```bash
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
PY=/home/kyle/Code/particula/.venv/bin/python
J=studies/voliotis-2022/joint/chains
"$PY" "$J/replay_chain.py"
```

For the full calculation, run the following sequence. The motif profiles hold the entering chemistry and doses fixed and set all steps in a proposed motif to the same nonzero rate, providing a direct check of the response to each individual proposal. [Rate probes for all 40 motifs](../results/chains/motif_profiles.csv)

The search resumes completed iteration files in the output directory. To rebuild all 40 from scratch, use a fresh working copy without the archived iteration files in its results/chains directory.

```bash
"$PY" "$J/chain_checks.py"
"$PY" "$J/search.py"
"$PY" "$J/profiles.py"
"$PY" "$J/compare.py"
"$PY" "$J/continue_fit.py"
"$PY" "$J/results.py"
"$PY" "$J/select_replay.py"
"$PY" "$J/plot.py"
```
