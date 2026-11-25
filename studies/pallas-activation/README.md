# Reproduce the Pallas salt-allocation fits

The fit uses the measured fraction of summit aerosol that activated within each dry mobility channel. The same CCN counter alternated hourly between total and size-resolved sampling, allowing a count comparison in different minutes. On 27 dates, 1,823 valid fractions overlap a dry size spectrum and a regional carbon sample. The 27 daily chemical-population fits and their empirical κ(D) comparators use the same observations. Leave-one-setting-out fits and the separately sampled bulk minutes test how well each fit transfers.

## Inputs and environment

Use Python 3.12 and a compatible NVIDIA GPU. The reviewed calculation used an RTX 5060, Particula source revision `3562abffafb4291383aadf84732a9d298a57ec4a`, the repository's `particula-aiomfac` 0.1.4, Warp 1.11.1, NumPy 2.3.4, pandas 2.3.3 and SciPy 1.16.2. From the repository root in an activated environment, install the checkout and the local AIOMFAC package. Choose a working directory outside the checkout for the observations and generated files.

```bash
python -m pip install -e . -e './packages/particula-aiomfac[cuda]' \
  'warp-lang==1.11.1' 'numpy==2.3.4' 'pandas==2.3.3' 'scipy==1.16.2'
export PALLAS_RUN="$(mktemp -d)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd studies/pallas-activation
python observations.py --download
```

Backman, Luoma, Servomaa, Vakkari and Brus (2026) supply the five daily size-selected CCN scans and author-fitted thresholds, the minute-resolution total-counter data and the dry DMPS spectra. The downloader checks the MD5 of each PaCE 2022 v4 ZIP. For an offline replay place the three named ZIPs in `PALLAS_RUN` and run `python observations.py`. The selection requires observations on both sides of 50% activation at all five settings, positive fitted diameters and positive fit errors. It yields 29 dates. The earliest 19 define the original early cohort, and the later ten supply the bulk evaluation. [Size-resolved and bulk observations, CC BY 4.0](https://doi.org/10.5281/zenodo.23309964) · [Instrument methods](https://doi.org/10.5194/essd-2025-284)

The included [Matorova chemistry extract](matorova-chemistry.csv) contains daily organic and elemental carbon and available particulate ions at station FI0036R, about 6 km from the summit. Two of the 29 dates lack organic carbon. Missing sodium or chloride allows zero assignable paired NaCl in the nominal scenario. Ammonium and sulfate limit their salt stoichiometrically. The carbon and salt measurements constrain their ratio within the population weighted by summit dry volume through 300 nm. Each compound's density and carbon fraction convert dry volume to carbon mass. The table derives from M. Aurela, ACTRIS, EMEP and GAW-WDCA (OC/EC), and U. Makkonen and EMEP (inorganic ions), 2026, hosted by EBAS at NILU under CC BY 4.0. [Carbon data](https://doi.org/10.48597/CAMS-V8ZB) · [Ion data](https://doi.org/10.48597/J4SV-VVBN)

To check the extract against its original revisions, install HDF5's `h5dump`, retrieve the two files below and run the extractor. It verifies their SHA-256 hashes and accepts only samples whose quality flags are all zero. [Carbon NetCDF](https://thredds.nilu.no/thredds/fileServer/ebas/FI0036R.20211231223000.20260213130000.online_thermal-optical_analysis..pm25.4y.23h.FI01L_SC-OCEC_RT3183.FI01L_thermal-optical_analysis_EUSAAR_2.lev2.nc) · [Ion NetCDF](https://thredds.nilu.no/thredds/fileServer/ebas/FI0036R.20130102095828.20260611090843.filter_3pack...13y.1d.FI01L_fp3s36.FI01L_IC_c.lev2.nc)

```bash
python chemistry_extract.py /path/to/carbon.nc /path/to/ions.nc "$PALLAS_RUN/chemistry.csv"
cmp matorova-chemistry.csv "$PALLAS_RUN/chemistry.csv"
```

## Replay from measurements to the figures

```bash
python tables.py
python study.py --pilot
python study.py
python boundaries.py
python midpoints.py
python between.py
python summarize.py
python figures/export.py "$PALLAS_RUN" --out "$PALLAS_RUN/figure-data"
```

| Stage | Scientific calculation | Generated evidence |
| --- | --- | --- |
| [Thermodynamics](tables.py) | Compare CPU and CUDA critical diameters for a six-composition pilot. Tabulate the Köhler maximum over 385 water volumes using AIOMFAC aqueous water activity, a separate pure-squalane reservoir and water surface tension at 298.15 K. Double the water-grid resolution for one salt blend. | `thresholds.npz`, `thermodynamics.json` |
| [Daily calibration](study.py) and [loss derivatives](calibration.py) | Minimize mean squared activated-fraction error, weighting the five supersaturations equally within each date. Particle-class fractions sum to one at each of five log-diameter knots. The primary model has 15 compositions, with 12 model sensitivities. Warp differentiates the loss on CUDA and SciPy constrains the fit. An independent linear bound checks the numerical optimum. Fit a two-parameter κ(D) power law using the same log-diameter response width (0.2572664) inherited from the earlier 17-date glutaric-acid model. | `fits.csv`, `comparator.csv`, `curves.csv`, `shares.csv.gz`, `samples.csv`, `run.json` |
| [Five midpoints versus full curves](midpoints.py) | At each of five archived D₅₀ values per date, fit a 50% activated fraction using the same population, uptake physics and regional chemistry. Refit with the 120 nm salt fraction bounded above by the 50 nm fraction. Compare the within-objective penalty with the same constraint applied to full curves. The central D₅₀ values are inputs and their fit errors in nm are not propagated. | `midpoint-check.csv`, midpoint-constraint figure and TOC counts |
| [Daily feasible ranges](study.py) and [boundary checks](boundaries.py) | Allow daily RMSE to rise by 0.01 in fraction units. Vary salt enrichment and organic ratio separately. Hold daily fitted inventories fixed when optimizing the mass-weighted three-group Riemer–West index. Refit the no-enrichment and externally mixed boundaries. | `profiles.csv`, `boundary-checks.csv` |
| [Omission and bulk replay](study.py) | Refit each model after omitting one setting, then predict its observations. Match a bulk-counter minute midpoint to the first subsequent DMPS scan end within 390 s. Convert spectral shape to number using measured ambient CN and integrate the response with 12-point quadrature in each log-diameter bin. The ten later dates yield 4,110 minutes at four settings. | `omitted.csv`, `comparator-omitted.csv`, `bulk.csv`, `comparator-bulk.csv` |
| [Between-channel redistribution](between.py) | Choose the median-error day, insert two DMPS channel centers between observed activation sizes, and redistribute two particle classes while preserving their integrated histograms. Independently check activation at each observed size and predict experimental setpoints at the inserted sizes. | `between.json`, `between-curves.csv`, `new-size-test.csv` |
| [Headline calculation](summarize.py) and [figure export](figures/export.py) | Recompute the primary daily RMSE from individual measured fractions. Use 20,000 paired date resamples for the difference from empirical κ(D). Export only the plotted quantities. | `headline.json`, `figure-data/{daily,activation,between-sizes,new-size-test,midpoint-check}.csv` |

The input-to-output replay reproduced daily mean RMSEs of 8.52295 and 9.68232 percentage points, with mean salt volume of 9.77865% at 50 nm and 49.59081% at 120 nm. Twelve daily solutions require enrichment within the chosen one-point tolerance. For September 16, the exact redistribution changes the salt allocation at 112.20 and 130.50 nm while preserving the measured-size responses. Decimal round-off in fitted class abundances can vary across compatible optimizers and CUDA drivers.

## Redraw the published figures on CPU

From this directory, run the following commands with the checkout installed. The five small [plot tables](../../docs/data/pallas-activation) ship with the manuscript. To check a new calibration, compare its exported tables with those files at the displayed precision.

```bash
python figures/replot.py
python figures/toc.py
python ../../docs/studies/literature/pallas-activation/daily_salt.py
```

The [manuscript plotting script](figures/replot.py) uses `midpoint-check.csv` for the paired penalties in Figure 1. It combines the 16 September scan in `activation.csv` with daily salt profiles from `daily.csv` for Figure 1. The same daily table supplies the mixing-index ranges in Figure 3. Figure 4 plots the two salt allocations in `between-sizes.csv` and the manuscript quotes the activation predictions from `new-size-test.csv`. The [TOC source](figures/toc.py) pairs the counts of dates requiring enrichment with the mean full-curve salt fractions. All scientific plots use `particula.default`, and the TOC uses `particula.toc`.

## Check the chemical surrogates and the eight-ion regional budget

The chemistry comparison adds three independent calculations. Reuse the source files and environment above.

```bash
python glycerol.py
python phase_checks.py
python chemistry_extract.py /path/to/carbon.nc /path/to/ions.nc \
  "$PALLAS_RUN/component-volumes.csv" --eight-ions
```

- [Glycerol sensitivity](glycerol.py) replaces all sorbitol with glycerol, builds new CUDA threshold tables and refits all 27 daily populations. The files `alternate.csv` and `alternate-bulk.csv` give daily salt allocation and the ten-date count errors.
- [Phase and activity checks](phase_checks.py) computes the aqueous activity at three measured binary sorbitol concentrations and checks equality of activities across the two binary squalane–water phases. It reports the multicomponent liquid–liquid solver's convergence status at two growth states alongside the pure-oil reservoir estimate in `physical-checks.json`. [Ninni et al., Table 1](https://doi.org/10.1021/je990303c)
- [Eight-ion extraction](chemistry_extract.py) adds quality-screened calcium, magnesium and potassium. The available-ion volume divides the sum of quantified ion masses by 1.77 g cm⁻³. The central organic volume uses 2.1 g of organic matter per gram of organic carbon and a density of 1.20 g cm⁻³. `component-volumes.csv` records the number of quantified ions per date. Its sodium-plus-chloride fraction requires all eight ions. The bulk fractions describe the regional samples collected with different ion and carbon inlets.
