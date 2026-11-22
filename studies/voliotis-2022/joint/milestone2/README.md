# Manchester network discovery with joint mass and particle counts

This study asks whether adding a reaction to the best archived joint mechanism improves the five-hour histories of all 15 experiments more than refitting its parent. Chemistry is shared. Each run has one precursor dose multiplier between 0.85 and 1.15. The inputs, fixed temperature, Brownian coagulation, deposition, source competition and original dimensionless RMSE definitions follow the [archived chamber methods](../chains/METHODS.md). Read the [literature report](../../../../docs/studies/literature/evaporation-and-uptake/manchester-network-growth/index.md) for the comparison with Manchester observations.

## Reproduce the fitted networks

Use the checkout's Particula Python environment with NumPy, SciPy, pandas, Matplotlib and Warp. The campaign kernels and the independent particle replay require CUDA. The forward-mode derivative implementation also requires a working CUDA Toolkit compiler on PATH. From the repository root, use the commands below. Change the interpreter path when the environment lives outside a linked worktree.

```bash
PY="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
STUDY=studies/voliotis-2022/joint/milestone2
IMAGES=docs/images/manchester-soa

"$PY" "$STUDY/baseline_replay.py"
"$PY" "$STUDY/growth.py"
"$PY" "$STUDY/activation.py"
"$PY" "$STUDY/catalog.py"
"$PY" "$STUDY/ad_shortlist.py" --calls 100 --extra 22,32,64,65
"$PY" "$STUDY/ad_fine.py" --sub 16 --calls 180 --extra 76 --extra 80
"$PY" "$STUDY/ad_fine.py" --sub 16 --calls 300 --extra 76 --continue-pair
```

The [AD methods and parameter map](ad_methods.md) describe each fitted coordinate and the exact selection controls. The search enumerates accretion candidates in a three-parent basis and tests their transfers in the original 31-pool model. A single candidate bank fixes channel ordering. For bimolecular events, each reactant contributes its own effective molar mass. For split daughters, half the **transferred organic mass** goes to each branch. The replay conserves effective organic mass after allowing for deposition and terminal products. Only the accretion proposals carry the additional explicit parent-equivalent balance. More detailed oxygen, nitrogen or molecular budgets would require a different set of species and measurements.

A 15 s search profiles 61 dose scales independently for each run and takes the worst resulting mass or count RMSE. The parent and child have identical budgets within each matched fit. The finite-difference line search uses the minimax linearization of all 30 errors and tests feasible proposals back on the nonlinear CUDA model. The extra coordinate is fixed to zero in the parent. The local directional probes use the current best doses, and each line-search comparison reprofiles the full dose grid. `per_start_calls` counts independent parameter/dose combinations per run. A batched CUDA launch can cover many such combinations. For dormant intermediate tests, the parent and child receive the same warm upstream rates wherever both contain the upstream transfer.

Forward-mode AD propagates precursor dose and fitted rate sensitivities through the bulk campaign model. It includes changing radius, coagulation, size-dependent wall deposition, source partitioning, and clock-time interpolation. The derivative of each run's RMSE enters a bounded minimax fit. Separate runs check positive-coordinate centered finite differences, one-sided rates at zero, a simultaneous direction, primal parity with Warp, and the reversed order of reaction updates. A finite calibration budget can leave a profile open. The returned rates are the best fitted values encountered.

## Compare starts, time steps and competing reactions

For the marginal first-round vapor-return addition, retain the initial decision, challenge it by deleting the new channel and optimizing the projected parent, and repeat the matched comparison at finer resolution. The original round-two screen began from the first qualifying 39-channel fit. Save that provisional seed locally before the finer adjudication below. Search scripts write restart files and trial histories into this directory for local reuse. Git ignores those files. Start a fresh search in a clean checkout.

```bash
"$PY" "$STUDY/ad_refine.py" ad_cont_76_01.json
"$PY" "$STUDY/ad_decisions.py"
"$PY" "$STUDY/selection.py" --round 1
PYTHONPATH="$STUDY:$PYTHONPATH" "$PY" -c 'import json; from growth import HERE, growth_seed; growth_seed(2, json.loads((HERE/"lineage.json").read_text())[0])'
"$PY" "$STUDY/ad_ablate.py"
"$PY" "$STUDY/ad_adjudicate.py" --calls 240
"$PY" "$STUDY/ad_decisions.py"
"$PY" "$STUDY/ad_ablate.py" --adjudicated
"$PY" "$STUDY/ad_decisions.py"
"$PY" "$STUDY/selection.py" --round 1

"$PY" "$STUDY/growth.py" --round 2 --sub 16 --seconds 4800
"$PY" "$STUDY/activation.py" --round 2 --sub 16
"$PY" "$STUDY/ad_shortlist.py" --round 2 --checks-only
"$PY" "$STUDY/ad_shortlist.py" --round 2 --checks-only --check-sub 16
"$PY" "$STUDY/ad_fine.py" --round 2 --sub 16 --calls 100 \
  --extra 71 --extra 100 --extra 38 --extra 66,67 --extra 82 \
  --extra 88 --extra 81 --extra 79 --extra 91
"$PY" "$STUDY/ad_decisions.py" --round 2
"$PY" "$STUDY/ad_adjudicate.py" --round 2 --extra 100 --calls 100
"$PY" "$STUDY/ad_decisions.py" --round 2
"$PY" "$STUDY/ad_ablate.py" --round 2 --adjudicated
"$PY" "$STUDY/ad_decisions.py" --round 2
"$PY" "$STUDY/selection.py" --round 2

"$PY" "$STUDY/growth.py" --round 3 --shortlist 3 --cycles 3 --sub 16 --seconds 1200
"$PY" "$STUDY/activation.py" --round 3 --sub 16 --cycles 3 --finalists 1
"$PY" "$STUDY/ad_shortlist.py" --round 3 --checks-only
"$PY" "$STUDY/ad_shortlist.py" --round 3 --checks-only --check-sub 32
"$PY" "$STUDY/ad_fine.py" --round 3 --sub 32 --verify-sub 64 --calls 40 \
  --extra 66,67 --extra 38 --extra 82 --extra 79
"$PY" "$STUDY/ad_decisions.py" --round 3
"$PY" "$STUDY/selection.py" --round 3
```

## Replot the published results

From the repository root, use the environment and paths above. These commands use the final data directly and run without CUDA or a search checkpoint.

```bash
"$PY" "$STUDY/verify.py"
"$PY" "$STUDY/figures.py"
"$PY" "$STUDY/size_plot.py" "$STUDY/size_results" "$IMAGES/milestone2-size.png"
"$PY" "$STUDY/../campaign_performance/study.py"
```

## Freeze and evaluate

```bash
"$PY" "$STUDY/evidence.py"
"$PY" "$STUDY/size_replay.py" "$STUDY/size_inputs.json"
"$PY" "$STUDY/size_plot.py" "$STUDY/size_results" "$IMAGES/milestone2-size.png"
"$PY" "$STUDY/size_resolution.py"
"$PY" "$STUDY/ad_trajectories.py" --round 3 --sub 64
"$PY" "$STUDY/final_omission.py" --calls 100
"$PY" "$STUDY/peak_timing.py"
"$PY" "$STUDY/flux_response.py"
"$PY" "$STUDY/figures.py"
"$PY" "$STUDY/search_summary.py"
"$PY" "$STUDY/verify.py"
"$PY" -m pytest -q "$STUDY/growth_controls_test.py"
```

The frozen comparison keeps doses fixed when changing integration step or channel order. The archive reproduction separately repeats the published practice of profiling doses at each numerical setting. Within a matched method, a new topology must improve on the parent at the tested settings by more than the numerical shift and the larger restart spread of the two competing networks. The final decision retains the original fit and eligible same-topology refits and ranks them by the worse of the two order scores at the finest completed qualification step. The global comparison also tests the child against the strongest simpler fit available from the other methods. Particle sizes are evaluated with the chosen rates and doses frozen. Aggregate error and peak timing use the same validation grid, with full per-run results also saved for the finer 0.46875 s audit.

| Files | Contents and units |
|---|---|
| `baselines*.{json,csv}` | All three archived joint fits, timings, differences from saved per-run results, and paired resolution/order checks. |
| `search_summary.json` | Distinct proposed and refitted motifs, matched improvements in percentage points, and retained channel counts at each stage. |
| `selection.json`, `size_inputs.json` | Final selection scores, plus the two frozen calibrations with exact reaction order, parameter values, and per-run doses. Size results record the input checksum. |
| `scorecard.csv`, `instrument_fits.csv.gz`, `trajectories.csv.gz` | Run-level scores; measurements and predictions at instrument sampling times; five-minute chamber mass (µg m⁻³), number (cm⁻³), deposited organics and corrected yield. Peak-to-last changes use sampled mass within the five-hour window. |
| `peak_timing.csv` | Organic mass changes over the measured peak-to-last interval and over each model’s own peak-to-last interval. |
| `channel_flux.csv.gz`, `diagnostics.json`, `final_omission.json` | Gross mass transferred through each channel in each run, closure, influential channel chosen by single-channel omission, and matched refit after removal. An appreciable channel moves at least 1 µg m⁻³ in one or more runs. |
| `ad_observation_jacobian.csv.gz`, `ad_observation_jacobian.json` | Per-observation dose and additional-rate derivatives, exact frozen inputs, and independent finite-difference checks. |
| `size_results/` | Every digitized spectrum cell, its quality flag, GMD and mode measurements, per-run scores, particle resolution and seed sensitivity. Size RMSE is in particles cm⁻³ per decade of diameter. |
| [Chemical evidence](ad_evidence.md) | Which product and volatility observations could discriminate an effective transfer, and which assignments remain hypothetical. |

The wall-corrected yield column is a diagnostic separate from the fit to suspended organics. The particle-size replays are comparisons with reserved measurement types from the same experiments. The reported restart ranges and step sensitivities describe numerical or optimizer behavior. No sampling-error confidence interval follows from those ranges.
