# Reproduce the CLOUD run 1833 follow-up experiments

Three figures trace how the study tested explanations for the residual in the main growth comparison. The first changes the liquid water content and the diameter seen by the instrument. The second exchanges the timing and total number of particles entering the model. The third asks whether a dark-period fit can identify the wall deposition that matters for the illuminated rise. The scripts and design files below also cover the intermediate experiments that connect those tests. [Main analysis and shared inputs](README.md)

## Redraw the three published figures on CPU

Start in `studies/cloud-1833` in the publication checkout, in a Python environment with NumPy, pandas and the Particula plotting extra. Each command reads final numeric results from this directory. The PNGs use hours since the start of run 1833. To convert the archived `archive_h` column to this clock, add 6.5 h.

```bash
out="$(mktemp -d)"
python activity/plot.py "$out/water-and-sizing.png"
python entry-timing/plot.py entry-timing "$out/particle-entry.png"
python recovery-four/plot.py "$out/wall-loss.png"
```

| Published panel | Final data and calculation |
|---|---|
| Liquid water, counts and size-resolved errors | [Equilibrated water uptake](activity/activity.csv) and the three fine-grid attractive-capture [spectra for the figure](activity/plot-spectra.npz). The [renderer](activity/plot.py) integrates the measured and predicted distributions with fractional bin overlap at 9.7 nm and masks the missing small channels. [Neutralized sulfate scores](activity/scores.csv) · [Acidic sensitivity](activity/acidic-scores.csv) · [Laboratory comparison](activity/reference/comparison.json) |
| Entry time versus total entry | [Four count histories and the observed count](entry-timing/tail.csv) come from the [flux exchange calculation](entry-timing/entry_timing.py). Threshold delays and shape scores are in [comparisons.csv](entry-timing/comparisons.csv). The [renderer](entry-timing/plot.py) selects the rising portion of the time series. |
| Dark wall fit and illuminated transfer | [Synthetic classification rates](recovery-four/classification-summary.csv) are recovery frequencies with imposed wall laws. [Held transfer scores](recovery-four/04-held-transfer.csv) compare fitted predictions with observations. The [renderer](recovery-four/plot.py) uses the [two final count trajectories](recovery-four/plot-series.csv) for the illuminated comparison. |

## Run the simulations and fits

Use the frozen engine checkout and Python environment described in the [primary replay instructions](README.md#rebuild-the-frozen-cuda-results). The original study used a GeForce RTX 5060. The recovery and ensemble workflows also use CPU workers. The entry, activity and inverse-model runs require an NVIDIA GPU, CUDA nvcc and a compatible driver. The signed-particle and molecular runs additionally compile C++17 code with OpenMP. Set `CLOUD_CUDA_ARCH` for your GPU when it differs from the default `sm_120`. The independent thermodynamic extraction uses Poppler.

The commands below start in the publication's `studies/cloud-1833` directory. Set `ENGINE` to the absolute path of the frozen Particula checkout, and activate the Python environment first. Individual studies read the saved final outputs of their predecessors, so a reader can replay one hypothesis directly. Replaying a later study against modified upstream physics requires replacing its corresponding frozen input explicitly in a separate working copy.

```bash
export PARTICULA_ROOT="$ENGINE"
export PY="$(command -v python)"
out="$(mktemp -d)"
export MPLCONFIGDIR="$out/matplotlib"

bash activity/source/replay.sh "$ENGINE" "$out/activity" "$PY"
bash inverse-model/source/replay.sh "$ENGINE" "$out/inverse-model" "$PY"
bash entry-timing/replay.sh "$ENGINE" "$out/entry-timing" "$PY"
bash particle-resolved/replay.sh "$ENGINE" "$out/particle-resolved" "$PY"
bash charge-clearing/replay.sh "$ENGINE" "$out/charge-clearing" "$PY"
bash charge-evaporation/replay.sh "$ENGINE" "$out/charge-evaporation" "$PY"
bash ammonia-sequence/replay.sh "$ENGINE" "$out/ammonia-sequence" "$PY"
bash five-nm-flux/replay.sh "$ENGINE" "$out/five-nm-flux" "$PY"
"$PY" five-nm-flux/analysis.py "$out/five-nm-flux"

bash dark-twenty/run.sh "$out/dark-twenty" dark-twenty/designs.json
"$PY" dark-twenty/analyze.py "$out/dark-twenty"
bash eddy-twenty/run.sh "$out/eddy-twenty" eddy-twenty/designs.json
"$PY" eddy-twenty/analyze.py "$out/eddy-twenty"
bash recovery-four/replay.sh "$out/recovery-four"
```

| Experiment | What the implementation varies or tests |
|---|---|
| [Activity and sizing](activity/source/physics.py) | Evaluates salted liquid composition and Kelvin curvature on the size grid. Dry-solute coordinates transport the particles, wet diameter controls chamber interactions and retained water controls the reported diameter. The [activity handoff](activity/freeze.json) records the sizing scenarios taken into fitting. |
| [Inverse model](inverse-model/source/forward.py) | Estimates bounded parameters by directional automatic differentiation through the chamber solver. The [fitting procedure](inverse-model/source/fit.py), [transfer](inverse-model/source/transfer.py) and [synthetic recovery](inverse-model/source/recovery.py) show training, scoring and mechanism confusion. Downstream experiments use the [selected parameters](inverse-model/frozen-final.csv) and [final gas and entry histories](inverse-model/final-gas.npz). |
| [Entry timing](entry-timing/entry_timing.py) and [individual particles](particle-resolved/campaign.py) | Exchange timing and integrated entry above 2.9 nm, then follow individual birth cohorts in independent boxes. Sampling uncertainty is estimated across boxes. [Cohort fate by stage](particle-resolved/cohorts.csv) · [Final cases](particle-resolved/cases.csv) |
| [Charge clearing](charge-clearing/clearing.py) and [cluster evaporation](charge-evaporation/campaign.py) | Compare charge-dependent clearing, ion attachment and signed molecular evaporation with the inherited chamber and entry settings. [Clearing scores](charge-clearing/clearing.csv) · [Evaporation diagnostics](charge-evaporation/diagnostics.csv). The [aligned AIS data](charge-evaporation/ion-reference.csv) distinguish each polarity. The [rate table](charge-evaporation/evaporation-reference.csv) lists acid-loss channels by molecular composition and charge. |
| [Ammonia sequence](ammonia-sequence/designs.json) | Runs eleven successive designs spanning exchange, molecular thermodynamics and signed survival. The [stage-transfer calculation](ammonia-sequence/transfer.py) scores frozen choices against signed observations. [Final case comparison](ammonia-sequence/comparison.csv) |
| [5 nm flux](five-nm-flux/boundary.py) | Compares an inlet estimated from neighboring observed bins with modeled first passage through 5 nm. The [counted crossings and parity check](five-nm-flux/observe.py) replay the selected molecular case. [Resolution checks](five-nm-flux/refinement.csv) compare the five-nanometer boundary calculations. The subsequent wall experiments use the [frozen boundary](five-nm-flux/boundary.csv). |
| [Twenty dark-period designs](dark-twenty/designs.json) and [twenty eddy-parameter designs](eddy-twenty/designs.json) | Each design evaluates a specific perturbation and can inherit the settings chosen in an earlier design. Run the complete design list in order. The [wall treatment](eddy-twenty/treatments.py) and [reference check](eddy-twenty/wall_check.py) specify the size dependence and chamber geometry. [Dark-cycle results](dark-twenty/cycle-results.csv) · [Eddy-cycle selections](eddy-twenty/execution-summary.csv). |
| [Wall recovery and transfer](recovery-four/campaign.py) | Generates the dark-period response bank and refined grid, runs [synthetic classification](recovery-four/recovery.py), then transfers frozen dark choices to the illuminated record. The replay compares classification and transfer tables with their published values and checks particle number closure. |

## Check the independent observations

For the ammonium sulfate comparison, save the E-AIM HTML as `observations-source.html` and the original Cheng article PDF as `cheng2015.pdf` in one directory. The extractors check the source bytes before using individual measurements and the vector geometry of Figure 2a. After the water-uptake replay, run the following with `REFERENCE_INPUTS` set to that directory. The generated residuals compare like liquid compositions. Mapping those measurements to a CLOUD particle diameter additionally assumes the model's surface tension, density and Kelvin correction. [E-AIM ammonium sulfate tables](https://www.aim.env.uea.ac.uk/aim/accent3/nh42so4-all.html#section2) · [Cheng et al., 2015](https://doi.org/10.1038/ncomms6923)

```bash
bash activity/source/reference-replay.sh "$out/activity" "$PY" "$REFERENCE_INPUTS"
```

The signed-ion extractor reads the Donahue Figure 3 records on the Figure 6 clock. It retains native sample identifiers so a repeated measurement can be counted once. To recover the dry molecular energies and evaporation table, supply the original Myllys ZIP to the two source extractors. The extraction records distinguish charge, sulfate equivalents and ammonia equivalents. Molecular energies and derived loss rates carry the uncertainty of the source calculations. [Donahue and CLOUD collaboration figure data, CC BY 4.0](https://doi.org/10.5281/zenodo.19695085) · [Myllys et al. supplementary data](https://doi.org/10.5194/acp-19-9753-2019-supplement) · [Ion extraction](charge-evaporation/ion_reference.py) · [Evaporation extraction](charge-evaporation/evaporation_reference.py) · [Energy extraction](ammonia-sequence/recover_energies.py)

With `MYLLYS_ZIP` set to the downloaded supplement and `DONAHUE_ZIP` to the downloaded chamber figure archive, regenerate the derived tables in the output directory.

```bash
"$PY" charge-evaporation/ion_reference.py --archive "$DONAHUE_ZIP" --output "$out/ion-reference.csv"
"$PY" charge-evaporation/evaporation_reference.py "$MYLLYS_ZIP" --output "$out/evaporation-reference.csv"
"$PY" ammonia-sequence/recover_energies.py "$MYLLYS_ZIP" "$out/energies.csv"
```
