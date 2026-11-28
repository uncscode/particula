# Reproducing the SABRE surface-area and N₂O₅ uptake comparison

The quickest way to examine the result is to redraw the figures from the archived plot values. To test a different physical assumption, recover the aircraft files and run the population, hydration and chemistry models together. Both routes use a CPU. The full replay starts with the final instrument files for the 2023 deployment, campaign figures preserved as numeric coordinates, and one published laboratory table. [Main study](index.md) · [Optical tests](optics.md) · [Chemistry tests](chemistry.md)

## Redraw the published figures

Begin in the root of the Particula checkout containing this study. Use a Python 3.12 or later environment with Particula, NumPy, SciPy, pandas, Matplotlib and Pillow. To run both the rendering and the full numerical calculation, install the local packages and plotting dependencies into that environment.

```bash
python -m pip install -e . -e packages/particula-aiomfac matplotlib pandas Pillow
python studies/sabre-small-mode/render_figures.py
```

The renderer reads the downloadable figure archive, including the original digitized observations. It writes the optical and chemical figures and the square graphical abstract to `docs/images/sabre-small-mode/`. The ZIP contains three CSV files of digitized measurements and `figure-values.json`, which stores the plotted model curves, flight medians, sensitivity points and annotations at up to eight significant digits. The main-page figures and tables also use the 81 paired cold-acid predictions and their humidities, three median area corrections, the two tracer-group contrast tests and two quartile predictions. The two laboratory uptake values at 230 K are transcribed from the [Fried et al. abstract](https://ntrs.nasa.gov/citations/19950051590). The replay reads the digitized values directly from this archive. The calculation scripts and the three flight-file manifests are under `studies/sabre-small-mode/` in the repository. [Download figure values and digitizations (ZIP)](../../../../website/public/study-data/sabre-small-mode/figure-data.zip)

## Recover the original observations

NOAA distributes the aircraft measurements as ZIP downloads after registration. The frozen manifests give the flight date, product ID, original filename, revision, file size and SHA-256 digest. Their URLs identify the archive landing page. Register there, choose each flight and its listed products, select **Download Selected**, and extract the original `.ict` files directly into `studies/sabre-small-mode/raw/`, preserving their names and bytes. The headers identify the instrument investigators and the product-specific terms. NOAA asks users to contact instrument investigators before use and to offer coauthorship as appropriate for publication. [NOAA archive and data-use policy](https://csl.noaa.gov/projects/sabre/data.html) · [2023 WB-57 download form](https://csl.noaa.gov/groups/csl7/measurements/2023sabre/wb57/DataDownload/)

The numerical replay needs 81 original aircraft files. To print the required filenames before downloading, run the following from the repository root. The last entry is the independent laboratory reference PDF.

```bash
python studies/sabre-small-mode/replay.py --list
```

| Source | Revision and coverage | Contribution |
| --- | --- | --- |
| NOAA Aerosol Microphysical Properties suite (AMP), Spectrometers for Optical Aerosol Properties (SOAP) | R3 UCATS-time spectra and R0 extinction from ten paired flight dates in March 2023 | Dry size distribution and independent 532 nm optical check |
| UCATS gas chromatographs, meteorological measurement system (MMS), UAS ozone photometer | R0 UCATS and 1 Hz MMS on those dates, R0 ozone on nine | Air-mass tracer, position, temperature and pressure, plus an ozone sensitivity |
| Diode Laser Hygrometer (DLH) and NOy laser-induced fluorescence | R0, eight selected flights | Ambient water and available nitrogen-oxide measurements |
| Particle Analysis by Laser Mass Spectrometry (PALMS) | R1 class fractions and R0 chemical mass, eight selected flights | Organic fraction, sulfate neutralization and independent mass-concentration check |

The 49 files in the optical and meteorological set span March 8, 11, 13, 14, 15, 17, 20, 23, 26 and 27. March 14 contributes zero valid optical windows. The geographic, tracer and altitude cuts select windows on eight flights, which supply 32 additional humidity and composition files. The replay’s file list selects the required revisions. [Instrument descriptions and campaign participants](https://csl.noaa.gov/projects/sabre/2023/payload/)

The instrument credits in the flight headers are Charles Brock, Adam Ahern, Ming Lyu, Anthony Fodel and Samuel Taylor for AMP, and Adam Ahern for SOAP. Eric Hintsa, Fred Moore, Geoff Dutton, David Nance and Brad Hall supplied the UCATS product. T. Paul Bui provided MMS, Troy Thornberry and Ru-Shan Gao provided UAS ozone, and Glenn Diskin provided DLH. Andrew Rollins and Eleanor Waxman supplied NOy-LIF. Gregory Schill, Michael Lawler, Maya Abou-Ghanem and Daniel Murphy supplied the two PALMS products. The N₂O₅ uptake estimates come from Decker and colleagues’ preliminary analysis. [NOAA SABRE payload](https://csl.noaa.gov/projects/sabre/2023/payload/) · [Decker et al., air-mass comparison](https://insidecires.colorado.edu/rendezvous/uploads/Rendezvous_2024_7167_1715358233.pdf)

For the laboratory comparison, download the original Gaston, Thornton and Ng article into the same directory. The replay uses Poppler’s `pdftotext` to recover Table 1 from printed page 5697 and checks the PDF digest before extraction. [Gaston et al. (2014), Table 1](https://doi.org/10.5194/acp-14-5693-2014)

```bash
mkdir -p studies/sabre-small-mode/raw
curl -fLsS https://acp.copernicus.org/articles/14/5693/2014/acp-14-5693-2014.pdf -o studies/sabre-small-mode/raw/Gaston-2014.pdf
python studies/sabre-small-mode/replay.py --check-inputs
```

## Repeat the calculations

Use the same checkout and environment as the figure renderer, with Poppler’s `pdftotext` available on the executable path. The reported calculations used Python 3.12.12, Particula 0.3.6, particula-aiomfac 0.1.4, Warp 1.11.1, NumPy 2.3.4 and SciPy 1.16.2. The entry point uses the checkout’s in-tree Particula and AIOMFAC implementations.

```bash
python studies/sabre-small-mode/replay.py
python studies/sabre-small-mode/render_figures.py
```

The first command checks original file sizes and hashes, aligns the flight records, fits the digitized number spectra, compares them with extinction, and computes the wet areas. It then fits the field and laboratory uptake probabilities, runs the reaction-channel and thermodynamic checks, calculates the contrasts across tracer groups, and compiles the resulting plotted values. Intermediate tables and the joined size distributions are regenerated locally. To inspect the unrounded fit results, use `chemistry/field-fits.csv`, `integrated-comparison.csv`, `integrated-contrasts.csv` and `tracer-contrast.csv` inside the study directory after the run. In the last table, select `case=aiomfac`, `basis=wet` and `subset=all_groups` to recover the headline comparison.

A full CPU replay from the 81 verified instrument files and the extracted laboratory table reproduced the archived graph values byte for byte. The integrated comparison, tracer-group contrast, nonideal hydration table, field and laboratory fits, and native chemistry replay also matched the research outputs byte for byte.

The join matches final AMP and UCATS samples by midpoint. It requires at least 57 valid one-second SOAP and navigation records inside each nominal 75-second window. DLH humidity uses the same coverage rule. Three-minute PALMS records are weighted by their overlap with the window, with at least 56.25 seconds of coverage for each reported composition variable. The replay starts from 1,600 matched optical windows. The study selection leaves 180 windows on eight flights. Of those, 133 enter the eight-group chemical comparison by tracer and flight date. The published chemistry marks have no individual retrieval timestamps. [Sampling and optical comparison](optics.md) · [Chemical comparison](chemistry.md)

## Tracing the digitized values back to their figures

| Retained table | Location in the primary source | Meaning of a row |
| --- | --- | --- |
| `jensen_spectra.csv` | [Jensen and colleagues, slide 14](https://csl.noaa.gov/events/sabre2024/presentations/day3/Jensen.pdf) | Vertex of a smooth number-size distribution grouped by N₂O. The two high-tracer groups used in the fits have 49 vertices each. |
| `surface_markers.csv` | [Lyu, Ahern and Brock in Thornberry, slide 10](https://csl.noaa.gov/events/sabre2024/presentations/day1/Thornberry.pdf) | Diameter of a surface-mode marker in a 10 ppbv N₂O group. |
| `chemistry/evidence/decker-tracer-marks.csv` | [Decker and colleagues, 2024 SABRE poster, *Air Mass Relation* panel](https://insidecires.colorado.edu/rendezvous/uploads/Rendezvous_2024_7167_1715358233.pdf) | Long central mark read from the preliminary chemical retrieval in a 5 ppbv tracer bin. |

The three extraction scripts retain their pixel or vector-coordinate calibrations and the expected SHA-256 digest for each original PDF. To audit the extractions, download the three linked originals into the private raw folder and run the commands below from the study directory. The checks use Poppler, NumPy, pandas and Pillow. The retained coordinates allow subsequent comparisons without re-reading the graphics.

```bash
cd studies/sabre-small-mode
curl -fLsS https://csl.noaa.gov/events/sabre2024/presentations/day3/Jensen.pdf -o raw/Jensen.pdf
curl -fLsS https://csl.noaa.gov/events/sabre2024/presentations/day1/Thornberry.pdf -o raw/Thornberry.pdf
curl -fLsS https://insidecires.colorado.edu/rendezvous/uploads/Rendezvous_2024_7167_1715358233.pdf -o raw/Decker-2024-poster.pdf
python extract_jensen.py raw/Jensen.pdf
python extract_surface_markers.py raw/Thornberry.pdf surface_markers.csv
python chemistry/evidence/extract_decker.py
```

The authors’ reported modal ranges, campaign presentations and archived flights establish the benchmark for this study. The original paper’s full-text PDF, supplement, exact data selection and archived chemistry–climate model output were unavailable for the reconstruction. The replay reproduces the Particula comparison documented here. [Lyu et al., *Science* (2026)](https://doi.org/10.1126/science.adw8939) · [Article abstract](https://pubmed.ncbi.nlm.nih.gov/42024743/)
