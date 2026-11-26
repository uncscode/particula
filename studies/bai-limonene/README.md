# From irradiated limonene films to cloud reflection

The model links Bai et al.’s laboratory observations to a controlled atmospheric comparison. Each illuminated aerosol population has its own dark control with the same initial particles, evaporation time, thermodynamics and updraft. The cloud and radiative transfer calculations use the same incident sunlight in each pair.

## Replot the published figures

The [figure scripts](plots/) read compact numerical summaries. Replotting does not require the original PDFs, CUDA or a new model run. The scalar results in the paper use the refined mesh. The July turnover check uses a fixed droplet observation window, while the monthly number and median use each particle’s κ-Köhler activation threshold.

## Replay the model

Run the commands below from the repository root. The reference study used Particula revision `3697f46ace75199c1043df9f4fd876c45ba6da5d`, Python 3.12, NumPy 2.3, SciPy 1.16, pandas 2.3, Warp 1.11 and an NVIDIA RTX 5060. Use the repository’s Python environment with the `extra` dependencies, Linux, Poppler’s `pdftocairo`, `curl`, an OpenMP C++ compiler and a CUDA-capable driver. The optional reduced-grid device comparison also uses `nvcc`. A new dedicated run directory holds downloads, generated tables and temporary optical columns. Allow several GiB for the full sweep.

```sh
python -m pip install -e '.[extra]'
run_dir=$(mktemp -d)
python studies/bai-limonene/replay.py "$run_dir" inputs smoke
python studies/bai-limonene/replay.py "$run_dir" july-week
```

The `inputs` stage extracts Bai’s film trajectories, solar photon flux and action spectra from the paper and supplement. It ranks Pallas background spectra from the versioned PaCE DMPS and size-resolved CCN files, combines 120 Finnish station days with 2022 NASA POWER irradiance, and computes size-dependent UV absorption using CUDA Mie theory. The [download manifest](sources.json) gives the original URLs and SHA-256 values. The solar spectrum uses Bai’s lamp anchors and Bohn and Lohse’s Jülich UV measurement. The water-sphere optics use ASTM G173 irradiance and Hale and Querry refractive indices from pinned source revisions. FMI WFS responses can change their XML envelope. The driver verifies a digest of the numerical station records for all 120 days. It also checks the hourly data payload when NASA updates its JSON metadata.

Bai et al. supply the source film measurements and photochemical spectrum. The film plot re-expresses and redraws individual Figure S15 trials. The article identifies CC BY 4.0 and the supplement record identifies CC BY-NC 4.0. Backman et al. supply the PaCE particle-size and activation records (CC BY 4.0). The Finnish Meteorological Institute supplies the surface irradiance observations (CC BY 4.0). [Bai article](https://doi.org/10.1021/acs.est.5c16671) · [Supplement and reuse terms](https://doi.org/10.1021/acs.est.5c16671.s001) · [PaCE v4](https://doi.org/10.5281/zenodo.23309964) · [FMI open-data licence](https://en.ilmatieteenlaitos.fi/open-data-licence)

The `smoke` stage checks a 1,728-size dark parcel and one sun-aged July parcel at the same 0.25 m/s ascent. The `july-week` stage advances the clean, slow July pair through independent particle aging, cloud growth and 168 hours of angular shortwave transfer. The mean light-minus-dark changes should be −6.34227 droplets/cm³, +1.24733 µm and −1.18001 W/m². These are July values. The three sampled summer weeks include June and August.

```sh
python studies/bai-limonene/replay.py "$run_dir" cloud radiation
python studies/bai-limonene/plots/prepare.py --run-root "$run_dir" --output-dir "$run_dir/plot-data"
```

The `cloud` stage uses 1,728 physical sizes in the independent OpenMP parcel solver. It follows 72 hours of aerosol history centered on each sample, saves 101 heights through a 100 m ascent, and evaluates two updrafts and two background spectra. Sampling is every three hours in July and six hours in other weeks, including both endpoints. One refined July screen assigns the same fast photochemical fraction to independent or less volatile material; another allows first-generation products to react again. The `radiation` stage uses CUDA for sphere calculations, then combines layer optical properties with direct and diffuse incident light in an adding–doubling solver. Fluxes above the cloud and reflecting surface include nights. An RTX 5060 ran the original GPU refinement and 9,460 cases were independently compared on CPU. The published high-resolution endpoints use the host solver, which accommodates the 1,728-size mesh.

For the original film transfer comparisons, use `python studies/bai-limonene/replay.py "$run_dir" film-parcels` once per clean output directory. It recovers the 1 m/s fresh, water-affinity-only, uniform-shrinkage and size-dependent residue cases on CUDA. To compare the reduced 162-size, 0.05 s parcel on both devices, use `python studies/bai-limonene/replay.py "$run_dir" cuda-parity`. The 1,728-size sweep and the 162-size device test answer different numerical questions. Additional suspension and field-observation experiments are described in the [checks guide](checks/README.md).

For the July exposure and illumination sensitivities, run `python studies/bai-limonene/replay.py "$run_dir" solar-controls` after `inputs`. This fetches the [laboratory ozone cross sections and CERES clear-sky irradiance](solar-control-sources.json). The [control driver](model/solar_controls.py) pairs 6, 24 and 72 hours of aging with separate dark controls on a common 432-size grid. It also compares two station-UV-index corrections on the refined mesh and changes the incident radiation over the same fast parcel. The resulting `seasonal/july-solar-controls.csv` records the changed observable and units for each test.

## Source to figure

| Published quantity | Calculation | Generated file below the run directory |
|---|---|---|
| Film mass and water uptake | [Vector extraction and water-volume conversion](model/film/extract.py), Bai Fig. 3 and SI Fig. S15 | `film/replicates.csv` |
| Film transfer at fixed bulk residue | [Parcel setup](model/study.py), [conserving size redistribution](model/transfer.py) | `results/screen.csv`, `results/structure.csv` |
| Optical action on aged aerosol | [Solar rates](model/sunlight/solar/solar.py), [UV absorption](model/sunlight/optics/mie_dose.py), [15-class aging and vapor exchange](model/seasonal/transport.cpp) | `sunlight/solar/rates.csv`, `sunlight/optics/dose.csv` |
| Monthly and diurnal droplet number and median | [Forcing reconstruction](model/seasonal/forcing/build.py), [selected Pallas backgrounds](model/sunlight/inflow/build_inflows.py), [parcel controller and activation](model/seasonal/experiment.py), [refined solver](model/seasonal/parcel_cpu.cpp), [aggregation](model/seasonal/resolved_summary.py) | `seasonal/analysis/resolved-{monthly,diurnal,seasonal}-cloud.csv` |
| Number gained or lost in a wet size bin | [Absolute paired counts in 0.25 µm bins](model/seasonal/size_spectra.py) | `seasonal/analysis/seasonal-fine-bin-differences.csv.gz` |
| July growth and selection | [Match originally emitted cohorts across both cloud states](model/seasonal/matched_cohorts.py) | `seasonal/sampling/cohort-months.csv` |
| Reflected sunlight and graphical abstract | [Water-sphere scattering](model/seasonal/radiation/sphere_table.py), [layer optics](model/seasonal/radiation/optical.py), [transfer solver](model/seasonal/radiation/transfer.cpp), [incident and surface conditions](model/seasonal/radiation/seasonal_flux.py) | `seasonal/radiation/resolved-monthly.csv`, `resolved-diurnal.csv.gz`, `resolved-seasonal.csv` |

Derived counts, diameters and fluxes are model results. The measured records constrain the incident light and starting particle distributions. Composition, source loading, transport age, liquid-cloud occurrence and ascent are prescribed scenarios.
