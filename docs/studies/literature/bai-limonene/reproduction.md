# Reproduce the limonene aerosol and cloud study

The film readings and field records come from published measurements. The droplets and radiative fluxes in this study come from paired simulations. Each monthly result averages seven days of modeled clouds under the station's illumination. An independent way to check the central result is to average the June, July and August rows of the monthly figure table for each regime. Each month contributes 168 hours, including darkness. The clean 0.25 m/s mean should give 6.45 fewer droplets/cm³, a 1.27 µm larger wet median and 1.45 W/m² less upward shortwave than the equally old dark control. [Monthly CSV](../../../../website/public/study-data/bai-limonene/monthly.csv) · [Runnable energy calculation](reflected_energy.py)

## From observations to cloud and radiation

| Stage | Observation or model input | Reproducible comparison |
| --- | --- | --- |
| Film and suspended aerosol | Bai's film mass and hygroscopicity, Wilson's airborne evaporation classes and Liu's fresh activation measurements | Reconstruct the film's relative water uptake. Compare shrinking particles with particles given the new water affinity at their original size. |
| Prior solar exposure | Finnish Meteorological Institute records from Sodankylä, satellite-derived UV bands and reconstructed action spectra | Integrate photochemistry and evaporation during the 72 hours preceding each parcel. Compare light-aged aerosol to the same initial aerosol aged in the dark. |
| Competition for cloud water | Pallas background size spectra, an assigned limonene-derived population and ascent speed | Solve activation and condensational growth through 100 m of ascent. Follow the same seed cohorts across paired clouds. |
| Energy above the cloud | Wet droplet distributions through the column, water optical constants, incident direct and diffuse sunlight and the surface boundary | Integrate Mie scattering and absorption across wavelengths and heights. Compare reflected energy for identically illuminated clouds over black and locally reflecting surfaces. |

The original calculations used Python 3.12 and an NVIDIA GeForce RTX 5060. The seasonal campaign contained 53,740 retained parcel-case entries, including resolution, chemistry and environmental checks. The every-hour grid used 162 weighted sizes. The reported seasonal comparison used 1,728 sizes, sampling cloud states every three hours in July and six hours in the other months, under hourly radiation. An independent CPU implementation checked 9,460 matched cases on the hourly grid. Eleven June cases missed the initial pointwise tolerances. With a 0.0125 s time step, all eleven agreed to within 0.000001 cm⁻³ and 0.000001 µm. [Which uncertainties could change the cloud result?](uncertainty.md)

The source guide identifies the acquisition, chemistry, parcel, radiation and aggregation steps. Use its retrieval commands to rebuild local inputs from the providers. The repository carries the calculation source, the figures and small tables of plotted quantities. Per-hour parcel states, compiled kernels, wavelength lookup caches and full meteorological downloads are generated in the replay workspace. [Simulation and retrieval guide](https://github.com/uncscode/particula/blob/main/studies/bai-limonene/README.md) · [Airborne and instrument checks](https://github.com/uncscode/particula/blob/main/studies/bai-limonene/checks/README.md)

For the independent suspended-particle measurements, the following command downloads two checksum-verified PDFs, reads the wet–wet limonene row of Wilson’s Table S1 and digitizes the eight limonene markers in Liu’s Figure 1b. It uses the Poppler commands `pdftotext` and `pdftocairo`. The 106.6 nm marker should give 0.357% measured critical supersaturation and 0.481% for the fresh film’s water affinity.

```bash
python studies/bai-limonene/suspended_measurements.py ./suspended-observations
```

For a matched simulation, run the following commands from the repository root on a CUDA-capable host. Set `work` to a new directory outside your checkout. The first command prepares the original observations and tests one clean July parcel. The second exercises a week of cloud and spectral radiative transfer. The source guide gives the full annual command and requirements, including the native compilers.

```bash
work=../bai-study-run
python studies/bai-limonene/replay.py "$work" inputs smoke
python studies/bai-limonene/replay.py "$work" july-week
```

Bai et al.'s supporting information supplies the digitized series in the film figure under CC BY-NC 4.0. PaCE v4 and the Finnish Meteorological Institute supply the particle and radiation observations under CC BY 4.0. The source guide identifies the specific records and how the inputs were transformed. [Bai supporting information](https://doi.org/10.1021/acs.est.5c16671.s001) · [PaCE v4](https://doi.org/10.5281/zenodo.23309964) · [FMI terms](https://en.ilmatieteenlaitos.fi/open-data-licence)

## Redraw a figure from the published values

From a Particula checkout with the plotting dependencies installed, run the plotting script to regenerate the study figures from their compact inputs. This step runs on CPU. To change an environmental or chemical assumption, use the simulation replay before regenerating the plot inputs.

```bash
python studies/bai-limonene/plots/render.py
python -m scripts.render_literature_tocs bai-limonene
```

| Figure | Downloadable plot values |
| --- | --- |
| Summer cloud graphical abstract | [Paired number, diameter and reflected sunlight](../../../../website/public/study-data/bai-limonene/summer-pair.csv) |
| One week per month | [Monthly changes in the four cloud regimes](../../../../website/public/study-data/bai-limonene/monthly.csv) |
| Which droplet sizes gain or lose number? | [Quarter-micrometre size bins, compressed CSV](../../../../website/public/study-data/bai-limonene/size-spectra.csv.gz) |
| Growth of the July droplets | [Counts and median contributions by cohort](../../../../website/public/study-data/bai-limonene/july-cohorts.csv) |
| Seasonal diurnal profiles | [Local-solar-hour means](../../../../website/public/study-data/bai-limonene/diurnal.csv) |
| Measured film properties and inferred water | [Three limonene trials](../../../../website/public/study-data/bai-limonene/film-water.csv) |

The 84 sampled days and the controls support the [main study's conditional comparison](index.md). Extending it to a different place or year requires local exposure histories and evidence about the particles and clouds that actually occur there.
