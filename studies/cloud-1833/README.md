# Reproduce the CLOUD run 1833 growth comparison

The [main analysis](../../docs/studies/literature/cloud-1833/index.md) asks how sulfuric acid capture affects the predicted size distribution when the models receive the same particle supply at 2.9 nm. The growth and wall-loss multipliers were chosen using the first lamp. The three frozen growth laws then received the same calibrated acid, estimated inlet and initial spectrum within each segment. Donahue et al.’s curves are their optimized outputs from the experiment. [Donahue et al., 2026](https://doi.org/10.5194/amt-19-3363-2026) · [Follow-up experiment sources](FOLLOWUPS.md)

## From each claim to its figure and numbers

| Figure or quantity | Calculation and retained result |
|---|---|
| TOC, including the 34-minute delay | [TOC generator](source/cloud_growth_toc.py) reads the 288-cell [stage summaries](full-transfer/stage-scores.csv) and the first sampled crossings in the [matched count histories](full-transfer/matched-series.csv.gz). |
| Stage 12 spectrum and nine-stage shape mismatch | [Capture and transport](source/study.py), [CUDA chamber kernel](source/chamber.cu), [original equal-entry controls](source/mechanism.py) and [frozen replay](full-transfer/simulate_frozen.py). The [mechanism table](mechanism.csv) gives the stage median of individual peak diameters. The plotted curves are the time-mean spectra. |
| Full size and time residuals | [Distribution matching](full-transfer/compare.py) maps the measured and published physical-size spectra to common bins and times. [Plotting code](full-transfer/plot.py) draws the matched ratios. [Cutoff sensitivity](full-transfer/threshold-check.csv) reports the effect of several masks separately from the full-spectrum scores. |
| Measured and modeled larger-particle counts | Integrate over the overlap of each log-diameter bin with 9.7 nm through the measured upper edge, approximately 64.7 nm. The [matched series](full-transfer/matched-series.csv.gz) stores both clocks and the integrated counts. The final mixed-light window and the stage-19 missing-inlet sensitivity have separate rows in the [stage scores](full-transfer/stage-scores.csv). |
| Model and numerical sensitivity | [Frozen design](full-transfer/design.md), [selected calibration settings](campaign.json), [paired stage resampling](full-transfer/stage-resampling.json), [run record and number budgets](full-transfer/execution.json) and [clean replay evidence](full-transfer/verification.json). |

### Data conventions

The two-minute size density is in particles/cm³ per decade of physical diameter. `spectra.npz` retains the measured series and grid. `edges` contains log₁₀(diameter/nm). `forcing.csv` contains the calibrated acid concentration in molecules/cm³, lamp histories and the small-instrument outage mask. `lamp_budget.csv` adds the coupled gas reconstruction used by the follow-up studies. `mechanism.npz` retains the three fine-grid common-flux, measured-acid reference spectra used for parity checks. The row order is ordinary vapor transfer, finite hard-sphere capture, then attractive capture. Local replays reconstruct the additional experimental branches.

The two predictions archives in `full-transfer/` contain final spectra at 192 and 288 cells. Each stores four segments with axes (growth law, time sample, measured size channel). `published-spectra.npz` contains the aligned Figure 9a and Figure 10b densities, with missing values outside each fit. `stage=999` in the stage table pools stages 09–17 for the excimer/LED segments and the scored samples of the other named segments. The shape mismatch is one-half the sum of the absolute differences in normalized per-bin particle fractions. The log-density error factor is computed from the absolute log₁₀ density errors weighted by bin width and time sample, using the stated offset. Cleaning periods and initialization samples are identified separately. The paper clock is the archive time plus 6.5 h.

Measured data and archived fitted model series are derived from the [Donahue and CLOUD collaboration figure archive](https://doi.org/10.5281/zenodo.19695085) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Transformations include median time matching, the paper's calibrated sulfuric acid scaling, linear-density recovery from plotted logs, overlap integration and interpolation. The zero floor in the archived log spectrum cannot distinguish all unresolved values below one particle/cm³ per decade. The full model comparison uses that convention consistently; the 50 particles/cm³ per decade display cutoff is separate.

## Regenerate the figures from the final results

Run from a checkout of this PR, using its usual Particula development environment with NumPy, pandas and Matplotlib. The values, scores and final spectra needed to plot are included. The figure destination defaults to `docs/images/cloud-1833/`. These commands direct an independent replot into a new output directory.

```bash
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
cd studies/cloud-1833
export CLOUD_FIGURES="$PWD/replay-output/figures"
python full-transfer/plot.py
python source/cloud_growth_toc.py
```

## Rebuild the frozen CUDA results

The original execution used Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2, Warp 1.11.1, CUDA 13.4 and a GeForce RTX 5060 with compute capability 12.0. The frozen comparison used the [Particula execution revision](https://github.com/Gorkowski/particula/commit/af0dc67bf7faa3013292659ee15872483a0a6d9a). The first growth-law calibration used [4403136e17](https://github.com/uncscode/particula/commit/4403136e17df574a63ba4d2096281f16dc19f6ad). The same later revision enables the optional AIOMFAC follow-ups. The scripts use the existing Dunne study's spectrum reader and acid collision helper from the indicated execution checkout.

From the publication repository root, create a separate checkout of the execution revision. Keep the publication scripts and tables in the original checkout.

```bash
git fetch https://github.com/Gorkowski/particula.git af0dc67bf7faa3013292659ee15872483a0a6d9a
git worktree add --detach ../particula-cloud1833-engine af0dc67bf7faa3013292659ee15872483a0a6d9a
export ENGINE="$(cd ../particula-cloud1833-engine && pwd)"
export PY="$(command -v python)"
cd studies/cloud-1833
export CLOUD_RESULTS="$PWD/replay-output/frozen"
bash full-transfer/replay.sh "$ENGINE" "$PY" "$CLOUD_RESULTS"
```

The runner compiles the retained CUDA kernel with nvcc, evaluates a short transport reference on CPU and GPU, and executes three laws across four segments at each of the two resolutions. It writes both prediction archives and the number-budget record to the supplied output directory. Set `CLOUD_CUDA_ARCH` to the local CUDA architecture if it differs from sm_120. Distinct device and compiler configurations require their own parity assessment.

## Rematch Donahue's archived fit and rescore the simulation

Download the original 131 MiB figure archive once. The comparison validates its SHA-256, extracts the optimized Figure 9a and 10b particle spectra and verifies the independently exported Figure 10a observations against the Figure 6a measurements. `compare.py` writes summary tables, aligned published spectra and paired stage-resampling intervals to `CLOUD_RESULTS`. Set that directory to the completed frozen replay above. For a score-only run on CPU, copy the two published `predictions-*.npz` files into a fresh `CLOUD_RESULTS` directory first.

```bash
mkdir -p inputs
curl --fail --location 'https://zenodo.org/api/records/19695085/files/archive.zip/content' \
  --output inputs/donahue2026-figure-data.zip
sha256sum inputs/donahue2026-figure-data.zip
export CLOUD_INPUTS="$PWD/inputs"
export CLOUD_FIGURES="$PWD/replay-output/figures"
"$PY" full-transfer/compare.py
"$PY" full-transfer/plot.py
"$PY" full-transfer/threshold_check.py
PYTHONPATH="$ENGINE" "$PY" source/cloud_growth_toc.py
"$PY" full-transfer/verify.py "$CLOUD_RESULTS"
```

Expected SHA-256: `5f6f48d653a989a57b633472908ebbddffad10a014db73032dc9f940912d4c85`. The threshold table uses the same two-minute windows, observed diameters and measured-density cutoff as the report.

## Rebuild the original inputs and parameter selection

The initial chamber models and fitting script are in [source/study.py](source/study.py). [Entry and refit controls](source/controls.py), the [lamp chemistry reconstruction](source/gas_replay.py), [equal-flux controls](source/mechanism.py) and [dark decay treatment](source/dark_decay.py) run in sequence. [The runner](source/reproduce.sh) also regenerates the initial diagnostic figures and their timing checks. From this study directory, pass the engine checkout and the downloaded archive directory, with a separate destination for regenerated calibration and trial outputs.

```bash
bash source/reproduce.sh "$ENGINE" "$CLOUD_INPUTS" "$PWD/replay-output/calibration" "$PY"
```

For the full archive-to-figure calculation, point `CLOUD_PACKET` at that rebuilt calibration directory before running the frozen and comparison commands. Leaving it unset uses the compact publication inputs. [Water uptake, entry timing, cluster survival and wall-loss replays](FOLLOWUPS.md)
