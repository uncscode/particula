# Reproduce the Mt. Soledad droplet and water-uptake comparisons

The [article](../../docs/studies/literature/particle-water-uptake/mt-soledad/index.md) compares two distinct field records. On seven reserved dates at 0.8% supersaturation, the denuded path counted 0.776 times the bypass response and the mean diameter of detected droplets changed by −0.013 µm. On eight reserved filter dates at 85% relative humidity, assigning 45% of particles to a nearly dry class reproduced the mean 60 nm growth factor but overpredicted the number fraction with growth factor ≤1.10. The two populations, instruments and date selections differ. [Numerical evidence](evidence/README.md) · [Plot tables and renderers](figures/README.md)

## From measurements to each figure

| Manuscript result | Observation and scientific calculation | Saved public output |
|---|---|---|
| Coastal graphical abstract and Figure 1 | [Match scans and one-second logger signals](source/mechanisms/droplets/extract.py) using [raw voltage and fitted transit delays](source/extension/acquisition.py). [Fit optical growth and thinning](source/mechanisms/droplets/models.py) on 0.4% scans, propagate the measured bypass wet-size spectrum at 0.8%, and retain counts below the detector boundary as missing from the optical spectrum. | [Channel masses](figures/optical-spectrum.csv), [date-weighted count and diameter contrasts](figures/optical-summary.csv), [graphical abstract values](figures/toc-reference.csv) |
| Integrated κ baseline | [Extract mobility spectra](source/extract.py), [calculate activation](source/activation.py) and [fit and transfer κ](source/study.py) using the native CUDA operator. Original scan pairs and integrated spectra have a separate 21-date transfer. The common CPC measures the reference concentration, so the ratio cannot establish physical transmission through the denuder. | [Integrated counts](evidence/integrated-counts.csv), [large-seed counts by date](evidence/large-seed-date-means.csv) |
| Figure 2A, measured and predicted growth | [Normalize the published growth PDFs](source/field-mixtures/observations/observations.py), [check salt standards](source/field-mixtures/observations/calibration.py) and [align the instrument clock](source/field-mixtures/observations/clock_profile.py). [Weight filter inlet flow](source/field-mixtures/comparison/filter_flow.py), reconstruct [six-hour PM₁](source/organic-films/bulk/composition.py) and [fit functional groups to admissible molecular recipes](source/field-mixtures/chem/recipe.py) independently of water uptake. [Compute water activity and Köhler thresholds](source/field-mixtures/chem/thermo.py) from one liquid or separate water pools. | [Eight-filter comparisons](figures/growth-by-filter.csv), [paired model errors](evidence/growth-contrasts.csv) |
| Figure 2B and C, shape of the growth distribution | [Fit a near-dry number fraction](source/field-mixtures/observations/partial_uptake.py) to the training 60 nm means, fix it on other dates and sizes and integrate the observed densities up to GF 1.10. The independent salt measurements set the residual width. Resample training and held filters separately to obtain paired bias intervals. Each accepted 12-minute scan has equal weight within its six-hour cell. | [Filter values](figures/growth-by-filter.csv), [uncertainty in distribution tails](figures/growth-intervals.csv) |
| Bypass midpoint and width | [Fit the original diameter mapping](source/mechanisms/mixing/mixing.py) at 0.4%, [transfer through the measured response](source/field-mixtures/comparison/response.py), [check inlet hydration](source/field-mixtures/chem/hydration.py), and [score matched windows and dates](source/field-mixtures/comparison/field_score.py). The uniform and composition-segregated populations have the same dry constituent inventory. | [Matched activation](evidence/activation.csv) |
| June 10 upper-bound film | [Bound the C16 inventory](source/field-mixtures/chem/c16.py), [assemble the dry particle and admissible-film cases](source/field-mixtures/films/prepare.py) and [propagate organic surface coverage and water transfer](source/organic-films/kinetic/film.py) through the [outlet](source/field-mixtures/films/outlet.py). The small-particle filter did not coincide with the large-particle optical episodes. | [Conditional film comparison](evidence/june-film.csv) |

## Data and environment

Run from the repository root. The calculation used Python 3.12.12, Particula 0.3.5 at `52e98ee06d4249b2a0c454229554f3047ae66a7b`, Particula-AIOMFAC 0.1.4 (upstream AIOMFAC-web `24041a1b0db090de1285008814a5285b4b9b746b`), NVIDIA Warp 1.11.1, NumPy 2.3.4, SciPy 1.16.2, pandas 2.3.3, Matplotlib 3.10.7, Pillow and the system `h5dump` executable. The original GPU was an NVIDIA GeForce RTX 5060. The replay selects Particula and the bundled AIOMFAC source from the checkout. If the software and this publication are in separate checkouts, set `PARTICULA_ROOT` to the recorded numerical revision. Use the interpreter from an environment containing those dependencies.

Three compact inputs support an offline check. [Selected optical spectra](source/mechanisms/droplets/large.npz) contain the number of paired cycles, UTC six-hour block, nominal supersaturation, and branch means for CPC, CCN, overflow and 20 optical channels. [Whole-scan spectra](source/mechanisms/droplets/processed.npz) are a separate processed control. [Fitted recipes](source/field-mixtures/chem/recipes.csv) contain window and filter IDs, chemical masses in µg/m³, measured temperatures and settings, and low/high-solubility assignments. The field solver uses a compact [elemental closure input](source/field-mixtures/chem/xrf.csv) and [previously fitted response mapping](source/mechanisms/mixing/final/fit.csv) as well. The full stages regenerate these from source. No raw instrument ZIPs are needed for the offline check.

| Public data source | Local names used by the replay |
|---|---|
| Ravichandran et al., CCN and mobility scans, April–June 2023. The two archive products contain processed daily netCDF and original one-second logger messages. | `ccn-processed.zip`, `ccn-raw.zip`. [Collection](https://doi.org/10.6075/J0KH0NHG) |
| Berta et al., Mt. Soledad aerosol mass spectrometer, corrected ambient components, particle size bands and author κ product. | `ams.nc`, `ams-kappa.nc`. [Collection](https://doi.org/10.6075/J0WM1DSC) |
| Mt. Soledad ultrafine particle measurements. Use the 30, 60 and 90 nm Gysel-inverted HTDMA PDFs and the published summary graphic. | `growth.zip`, `htdma-summary.png`. [Collection](https://doi.org/10.6075/J08G8KWG) |
| Revised dry-size-segregated filter functional groups, organic nitrate, XRF elemental filter inventory, and valve/flow minute series. | `ftir.csv`, `organic-nitrate.csv`, `xrf.nc`, `filter-flow.zip`. [Collection](https://doi.org/10.6075/J06W9B8D) |
| Liggio et al., calibrated non-cloud inlet gas channels, conditional donor for gas removal controls. | `gas.zip`. [Collection](https://doi.org/10.6075/J0RV0P3H) |

The [download manifest](source/products.json) identifies the exact release object URL and SHA-256 for each source except the full 1.92 GB filter-flow ZIP. Its [minute-series extractor](source/field-mixtures/comparison/extract_flow_member.py) verifies the member CRC and SHA-256. If the original processed minute series is already available, place it in the inputs directory as `minute-flow.csv`; the replay verifies its SHA-256 and skips downloading the larger ZIP. Keep the listed archives outside the repository. Consult the source collection for its data creators, citations and license terms. The tables here aggregate and transform those observations.

## Commands

The first command copies the published code and compact inputs to a separate working directory, refits the optical models on 26 training windows, scores 16 test windows and evaluates six native AIOMFAC pilot predictions. It runs on CPU, without downloads. It checks the 0.776 measured ratio, the uniform-delay predicted −1.541 µm shift and the June 10 low-solubility 60 nm growth factor of 1.49885. The independent chemistry pilot uses 97 water-volume nodes; the manuscript uses 257, checked at 513.

```sh
export SOLEDAD_RUN="$PWD/../soledad-replay"
python studies/mt-soledad/source/replay.py check "$SOLEDAD_RUN"
```

For the complete rebuild, allow several GB of working storage. The science stages write intermediate and final outputs under `$SOLEDAD_RUN/model`. Stage names are sequential below. `fit-original` and `field` require CUDA. The raw data and filter flow need only be retrieved once.

```sh
python studies/mt-soledad/source/fetch.py "$SOLEDAD_RUN/inputs"
python studies/mt-soledad/source/replay.py extract "$SOLEDAD_RUN"
python studies/mt-soledad/source/replay.py fit-original "$SOLEDAD_RUN"
python studies/mt-soledad/source/replay.py field "$SOLEDAD_RUN"
python studies/mt-soledad/source/replay.py film "$SOLEDAD_RUN"
python studies/mt-soledad/evidence/select_tables.py "$SOLEDAD_RUN/model"
PYTHONPATH=. python studies/mt-soledad/figures/plot_figures.py --refresh --study-root "$SOLEDAD_RUN/model"
PYTHONPATH=. python -m scripts.render_literature_tocs mt-soledad
```

A full replay from the archived instrument products regenerated all thirteen manuscript evidence tables and all five figure and graphical-abstract CSVs byte for byte. It included the 8,638-condition original CUDA activation fit, the field chemistry runs and the June film calculation. The final three commands refresh the public evidence tables and artwork in the checkout. To redraw the figures from the saved CSVs without running the models, follow the [figure instructions](figures/README.md). Model evaluation on a withheld supersaturation or filter date tests transfer within this campaign. The 64 chemistry draws describe uncertainty in the chemical allocations and properties; the date or filter intervals describe sampling uncertainty. The raw optical channels do not supply the size of an undetected droplet. The PM₁ group and ion measurements do not identify molecular composition at each mobility-selected size.

The field experiment and interpretation of trace-gas removal are due to Ravichandran and colleagues. The chemical scenarios, training/held division and response-model transfers in this directory are the Particula analysis of their published data and the complementary Mt. Soledad archives. [Primary paper](https://doi.org/10.1126/sciadv.adx0960)
