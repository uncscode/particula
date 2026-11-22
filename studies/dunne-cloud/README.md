# From CLOUD chamber measurements to the Particula study figures

The study has two entry points. To redraw the published figures, use the [eight figure CSVs and replot script](../../website/public/study-data/dunne-cloud/README.md). To see how the study obtained those values, start with the table below. Each row follows measurements through a model or fit into a plotted quantity. [Study report](../../docs/studies/literature/dunne-cloud/index.md)

| Figure | How the plotted values were obtained |
| --- | --- |
| Formation overview and withheld conditions | Dunne’s corrected 1.7 nm rates enter the [published law](analysis/reference.py) and [nominal multistart fits](analysis/fit.py). The [joint refits](monte-carlo/formation.py) withhold complete condition groups, perturb the calibrations and refit both pathway models. The [summary](monte-carlo/formation_summary.py) takes percentiles of the predicted rates and holdout errors. The [fixed-share calculation](monte-carlo/formation_profile.py) supplies the separate training-error comparison in the opening paragraph. |
| Cluster growth and neutral dimers | [Cluster states and evaporation energies](analysis/chemistry.py) and the [dimer balances](analysis/dimers.py) define the molecular models. The [cluster calculation](monte-carlo/cluster.py) follows a tagged acid–ammonia pair and compares modeled dimer concentrations with three digitized CLOUD measurements and a flow-tube measurement. |
| Cold survival, flux and zero crossings | The [CUDA particle histories](analysis/particles.py) supply the first-passage calculation. The [transport ensemble](monte-carlo/transport.py) varies gas conditions, growth and wall loss. The [finite-volume simulation](monte-carlo/transport_population.py) counts threshold crossings in hours 12–18. The [transport summary](monte-carlo/transport_results.py) pairs survival with fitted source rates and calculates the fraction of replays with zero crossings. |
| Warm growth controls, trajectories and paired errors | [Run 1833 extraction](companion/extract_run1833.py) and the [measurement and boundary calculation](simba/replay.py) convert the author spectra into measured counts and a 2.9 nm entry density. The [SiMBA network generator](simba/cloud_networks.py) and [candidate search](simba/search.py) test sulfur reactions. The [dark-period wall calibration](simba/wall-sulfur/balance.py) and [sulfur forcing](simba/wall-sulfur/sulfur_inputs.py) feed the [warm ensemble](monte-carlo/warm.py). Its [C++ solver](monte-carlo/warm_solver.cpp) advances the size distribution. The [warm summary](monte-carlo/warm_summary.py) calculates per-draw peaks, pointwise bands and paired count errors. |
| Proposed acid scan and published-law comparison | The [paired rate refits](monte-carlo/formation.py) predict the high-to-low acid rate ratio at 223 K within each draw. The [archive calculation](monte-carlo/archive.py) evaluates the fixed published coefficients on different subsets. The [summary](monte-carlo/formation_summary.py) takes the plotted percentiles and supplies the nominal references. |

The [CSV exporter](analysis/export_plot_summaries.py) selects the conditions, model families, units, reference points and percentile columns for each figure. It uses the high-to-low acid ratio within each draw, the peak count within each warm trajectory, and the error difference within each paired simulation. Taking ratios, peaks or differences after averaging would give different results. Each exported CSV shares its filename stem with its figure.

## Inputs and provenance

Run these commands from this folder in a checkout of Particula. The downloader reads the [Dunne manifest](sources.csv) and the required records from the [companion manifest](companion/sources.csv). It verifies each file before use. For an existing copy, pass `--cache PATH` pointing to a directory with the same `raw/` and `companion/` layout. The original files remain available from the authors. [Dunne measurements](https://zenodo.org/records/13981565) · [Kürten SANTIAGO data](https://doi.org/10.5281/zenodo.2634985) · [Donahue run 1833 data](https://doi.org/10.5281/zenodo.19695085)

```bash
python fetch_inputs.py
bash run.sh inputs
```

The input stage verifies the two archive copies of Dunne’s table, extracts the diagnostic figures, generates the neutral-rate crosswalk and reads the companion size spectra. Expected landmarks are 377 Dunne records, 82 neutral records in the 33–43% humidity window and 355 markers recovered from Fig. S3. The [PDF extractor](rate_sources/extract_reference_figures.py) records its axis calibration and color selection. The [three dimer points](rate_sources/kurten_fig9_248K_dimers.csv) include the source pixel centers and approximate concentrations digitized from the filled circles in the final panel of Fig. 9. The [rate coefficient table](rate_sources/coefficients.csv) identifies the printed and higher-precision parameter sets by column. [Dunne supplement](https://authors.library.caltech.edu/records/gtyc9-ykr88) · [Kürten Fig. 9](https://acp.copernicus.org/articles/15/10701/2015/) · [ClimaParams coefficient revision](https://github.com/CliMA/ClimaParams.jl/blob/3c1f8be835c43e467ac2d963bd48cc4150a3df01/src/parameters.toml)

## Running the simulations

The joint ensembles used Particula revision `d84209ef685eee9c7d1b3fc9ed813469fa6137a3`, Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2, Matplotlib 3.10.7 and Warp 1.11.1. Install the checkout and its scientific dependencies in the chosen environment. The scripts also use pandas and Pillow, the Poppler commands `pdftocairo` and `pdfimages`, and `g++` with C++17 support. The cold first-passage and SiMBA CUDA replays used an NVIDIA RTX 5060. The original nominal fits used Particula revision `77b383cdb32d45f78567e1388becbf39f70d23b7` and CPU PyTorch 2.10.0 in float64, with NumPy 2.4.3 and SciPy 1.17.1 in the fitting environment. Set `PYTHON` and `FIT_PYTHON` to absolute interpreter paths to use separate environments. Both default to the active Python interpreter. `WORKERS` sets the worker count for the fit and ensemble commands that accept it.

After acquiring the inputs, `bash run.sh all` runs the stages in order. To run selected calculations, use the same stage names as below. Run prerequisites first. The [stage commands](run.sh) show each script invocation, including the second optimization pass and the seeded Monte Carlo designs.

```bash
bash run.sh inputs fits nominal
bash run.sh sulfur
bash run.sh formation clusters transport warm
bash run.sh plots
```

- **Inputs, fits and nominal calculations.** Generate the measured inputs and fit starts in `results/`. The nominal runs also evaluate cluster chemistry, cold particle transport and warm growth controls.
- **Sulfur.** Screen the balanced networks, fit on the early warm interval and generate the size-dependent wall calibration and sulfur-input tables used by the warm ensemble. [Search protocol](simba/protocol.md)
- **Formation and clusters.** Four independently scrambled blocks supply 128 cold draws. The fit stages use the saved nominal solutions as starts. The cluster stage compares the same observations across both collision kernels. [Draw generation](monte-carlo/design.py)
- **Transport and warm growth.** Run the cold analytic transport control before the stochastic CUDA replays. The warm stage uses 512 draws across 16 model variants, then evaluates the second light pulse and summarizes the saved trajectories. Generated numerical histories remain local.
- **Plots.** Recalculate the [headline comparisons](monte-carlo/review_comparisons.py), export the eight tables into `results/plot-data/` and redraw the figures in `results/figures/`. Compare those tables with the [published plot data](../../website/public/study-data/dunne-cloud/README.md).
