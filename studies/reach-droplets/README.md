# Reproducing the REACh droplet study

The [study](../../docs/studies/literature/reach-droplets/index.md) compares one expansion with fans and one without. The independent input records are digitizations of Figures 6 and 7 of Sagan et al. and Figure S2 of the supplement. Diameter is wet diameter in µm. Time is in seconds from recording start. The source uses `forced` and `unforced` for fans on and off. [Article](https://doi.org/10.1063/5.0344133) · [Supplement](https://doi.org/10.60893/figshare.jcp.32982017)

## Replot the published figures

From the repository root, use Python 3.12 with NumPy, pandas, Matplotlib and Pillow. The CPU example integrates the measured and simulated spectra during the fan-off burst. It also runs from the standalone download on the study page.

```sh
python docs/studies/literature/reach-droplets/burst.py
python studies/reach-droplets/figures.py
python -m scripts.render_literature_tocs reach-droplets
```

| Published output | Values in [figure-data.zip](../../website/public/study-data/reach-droplets/figure-data.zip) | Scientific source |
|---|---|---|
| TOC and Figure 1, concentrations and sizes | `time-series`, `spectra`, `pooled-observed`, `tail-uncertainty` | `model/simulate.py` runs independent well-mixed water-droplet populations. `model/replay_fits.py histories` fits time-dependent controls and replays them in the spatial chamber. |
| Figures 2a,b and Table 1, measured and modeled size histories | `spectra`, `scores`, `time-series` | `model/compare_spectra.py` averages like times and sizes and checks the archived concentrations. The maps use log₁₀((model + 0.005)/(measured + 0.005)). |
| Figure 3, highest-ranked family representatives | `best-four`, `ranked-configurations`, `spectra` | `model/replay_fits.py ranked` runs the four selected configurations. Rank by the square root of the mean of the four mean squared density errors (two fan settings × two windows). The selection used these same two records. |
| Figure 4, matched camera A and proposed upper view | `matched-tails`, `matched-spectra`, `matched-summary`, `matched-transfer` | `model/experiment.py` samples all views in the same chamber. `model/match.py` matches peak time and increment at A. `model/transfer.py` replays the interventions at other resolutions and flows. |
| Figures S1–S2, source and camera-position sensitivity | `source-spectra`, `source-tails`, `view-positions`, `placement-summary`, `placement-floor` | `model/placement.py` runs the seven flow, time-step and grid settings and compares vertical and lateral displacements. Divide changes in bin inventory by bin width for density. |

The column key and units are inside the ZIP. The [observations archive](../../website/public/study-data/reach-droplets/digitizations.zip) contains the extracted numbers before model fitting. Its six CSVs are sufficient for the simulations. The two archives have SHA-256 digests `ef4785135ed68f1e51e6a71fecf7a891820dd36c09a18811d15dbde4af66575c` (figures) and `e30b0ed8861572c2e3f909ece46dbedb04386d59b64647d05b9a437b87c62d01` (observations).

## Repeat the microphysics and fits

Use Particula from this checkout, SciPy, pandas, Warp, CUDA-enabled PyTorch, a C++17 compiler named `g++` and an NVIDIA device. The original simulations used Particula revision `bf91687e521d24689486e721d6073250e6ce85f6` and an RTX 5060. The replay evaluates finite-vapor condensation, gravitational settling, physical collision kernels and size-dependent wall deposition. The multivolume fit evolves the distribution in bins and uses reverse-mode differentiation. The particle check uses six independently sampled populations, conservative stochastic mergers and the same optical footprint.

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export REACH_OUTPUT="$PWD/reach-output"
model=studies/reach-droplets/model
python "$model/simulate.py" --cases full --particles 768 --replicas 32 --split 2
for fan in forced unforced; do
    python "$model/replay_fits.py" histories "$fan"
    python "$model/replay_fits.py" ranked "$fan"
    for series in Hologram 'Forward physics' 'Fit before 5 s' 'Blocked fit' 'Spatial update'; do
        python "$model/compare_spectra.py" "$fan" "$series"
    done
done
```

Use a fresh output directory for a new model version. Completed fits can be reused in the same calculation. The optional `model/early_background.py` repeats the early joint calibration on a CPU and writes its fit to the output directory. The scientific replay reads the frozen [early fit](early-background.json). Its entries are the temperature offset in K, temperature contrast, log exchange scale, log circulation scale and log outer-to-central initial concentration ratio. The early fit used six diameter bands before 5 s, 128 seeded Sobol starting points and five gradient refinements.

For the camera experiment, start with a 1,536-volume unperturbed chamber and target a common 0.080 droplets/cm³ extra peak at the original camera. The perturbation changes the number arriving from above or the probability of local collisions. All views passively sample the resulting evolution. The placement calculation includes separate local-cooling and import-from-below scenarios.

```sh
python "$model/experiment.py" 1536 jets 2 0.05 control
python "$model/match.py"
python "$model/transfer.py"
python "$model/placement.py"
```

## Repeat the figure digitization

Obtain the [article PDF used for extraction](https://repository.library.noaa.gov/view/noaa/76087/noaa_76087_DS1.pdf) and the [supplemental PDF](https://doi.org/10.60893/figshare.jcp.32982017). The digitizers check SHA-256 because the pixel coordinates depend on the PDF version. Install Poppler's `pdftocairo` and `pdftoppm`. The last step uses the Figure 6 time stamps from the preceding command.

```sh
digitize=studies/reach-droplets/digitize
python "$digitize/holograms.py" --source article.pdf --output extracted
python "$digitize/gas.py" --source article.pdf --output extracted/figure6.csv
python "$digitize/radial_temperature.py" --source supplement.pdf --output extracted
```

The article is licensed CC BY 4.0. The publisher lists CC BY-NC 4.0 for the supplement. The archived figure transcriptions retain attribution and the original sources. [Licenses, sampling windows and digitization uncertainty](../../docs/studies/literature/reach-droplets/data.md)
