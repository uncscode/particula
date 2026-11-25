# Lund freezing and entrainment

The [study](../../docs/studies/literature/lund-freezing/index.md) compares the isothermal assays of Jakobsson et al. with forecasts fitted to short holds. It then follows incoming and resident carriers through prescribed clouds. [Source-to-figure guide](../../docs/studies/literature/lund-freezing/reproducibility.md) · [Published numbers and case configurations](../../docs/data/lund-freezing/)

## Redraw the figures

From the repository root, use a Particula development environment with pandas, NumPy, SciPy, Matplotlib and Warp installed. The figure renderer imports the model definitions and runs on the CPU. These commands use Bash. Choose a fresh temporary directory for generated files.

```bash
repo="$PWD"
src="$repo/studies/lund-freezing"
data="$repo/docs/data/lund-freezing"
work=$(mktemp -d)
export LUND_PYTHON="$(command -v python)"
PYTHONPATH="$repo" "$LUND_PYTHON" "$src/replot.py" --output "$work/figures"
```

The eleven PNG figures use the Particula plotting styles. The 86 KiB figure archive holds CSV data for each plotted panel. The supporting archive contains plate and ensemble statistics used in the text, including sensitivity to filter volume, exposure, moisture, grid spacing and incoming grain type. The three-row baseline table is also provided as plain CSV. The movie uses the particle and field history from one track run. It can be rendered after the reference eddy replay below.

## Recover the author inputs and repeat the assay

The event archive is maintained at Lund University. The workbooks are Tables 1, 3 and 4 of Jakobsson et al. (2022). The downloader compares all four files with the study’s SHA-256 digests before using them. The Arctic sounding equations and mineral fits are implemented in the study source. Their original references are linked in the [cloud sensitivity analysis](../../docs/studies/literature/lund-freezing/cloud-sensitivity.md). [Original paper](https://doi.org/10.5194/acp-22-6717-2022) · [Event data](https://web.nateko.lu.se/vp/inp_raw_data.tar)

Run the following from the same shell. A separate checkout supplies the exact Particula library used for the scientific runs. The model and fitting scripts are taken from this publication. Copy them into the scratch directory so that the generated results stay together.

```bash
model="$work/particula-model"
run="$work/run"
git worktree add --detach "$model" db804546ce7af9443f3aafd7e4b3a58b187253ac
mkdir -p "$run"
cp -R "$src/." "$run/"
"$LUND_PYTHON" "$run/fetch_inputs.py" "$run"
export PYTHONPATH="$model" OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
"$LUND_PYTHON" "$run/study.py" "$run" "$run"
```

The CPU assay should return 12.09 observed and 15.93 fixed-rate predicted late freezes per 100 hold-entry liquid wells. The newly fitted well forecasts also provide the filter-normalized counts used by the cloud driver.

## Replay the CUDA experiments

The recorded setup used Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2, Matplotlib 3.10.7, Warp 1.11.1 and an NVIDIA GeForce RTX 5060 (8 GiB). Each simulated cloud is a 1,600 × 600 × 1 m slice with prescribed circulation. The overview example reads the saved comparisons and runs on CPU. The commands below require a CUDA device.

For a short replay, run the eleven selected cases with their original seeds and compare the total, resident and arriving primary ice, suspended ice and both water paths against the published endpoints. The time allowance is in wall seconds. A slow device may need a larger allowance from the outset.

```bash
bash "$run/cloud-sources/run.sh" "$model" "$run/selected" \
  --cases "$run/cloud-sources/replay-cases.json" --seconds 900
"$LUND_PYTHON" "$run/cloud-sources/check_selected.py" \
  "$run/selected/runs.jsonl" "$data/selected-endpoints.csv"
```

For the full study, the first command repeats the assay and pulse analyses into an empty directory. The second copies the resulting numerical tables alongside the source for the subsequent stages. The reference-eddy driver executes the paired and tagged cases. The campaign driver then reads the exact 4,350 completed settings and seeds. Its wall-time allowance below provides headroom for slower hardware. The published two-hour experiment completed 4,348 six-hour runs and two half-hour pilots.

```bash
bash "$run/replay.sh" "$model" "$run/assay-results"
cp "$run/assay-results/"*.csv* "$run/"
bash "$run/cloud-eddy/replay.sh" "$model" "$run/cloud-eddy/results"
bash "$run/cloud-sources/run.sh" "$model" "$run/cloud-sources/results" \
  --cases "$data/executed-cases.json.gz" --seconds 86400
bash "$run/cloud-sources/replot.sh" "$model" "$run/cloud-sources/results"
"$LUND_PYTHON" "$run/export_tables.py" "$run" "$work/regenerated-data"
"$LUND_PYTHON" "$run/replot.py" --data "$work/regenerated-data" --output "$work/regenerated-figures"
```

For the perspective animation use Pillow and FFmpeg with the completed reference-eddy results. The script reads the saved freezing events, carrier trajectories and liquid-water fields and writes `cloud-eddy/results/cloud-dynamics.gif`. The one-metre depth shown in perspective is the thickness of the modeled slice.

```bash
"$LUND_PYTHON" "$run/cloud-eddy/animate_eddy.py"
```

The archived timing and maximum relative water-budget residual appear in [accounting.json](../../docs/data/lund-freezing/accounting.json). The 95% intervals in the study quantify paired particle realizations within the specified model. They exclude uncertainty in the imposed circulation, incoming aerosol activity and atmospheric sampling.
