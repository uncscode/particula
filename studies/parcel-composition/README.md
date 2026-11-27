# Regenerating a particle-resolved aerosol

The study compares activation, chemistry, cloud processing, nucleation and optics after replacing full particle populations with sampleable checkpoint rows. The published figures cover the initial exchange through the held-out restart and instrument-channel tests.

## Redraw the figures

From the checkout root with the Particula development Python, run:

```bash
PYTHONPATH=. MPLBACKEND=Agg python studies/parcel-composition/figure_data.py --output .artifacts/parcel-figures
```

This regenerates the graphical abstract and 17 quantitative plots from the CSV tables in [figure-data.zip](../../website/public/study-data/boreal/figure-data.zip), the [failure matrix](boreal/redraw-failure-matrix.csv) and one row in the cloudy checkpoint. The ZIP contains ordinary CSVs named for the experiment. They record plotted measurements and small subsets used to calculate ranges, quantiles and curves. Columns use the same names and units as the corresponding study runner. The two downloadable 16-row checkpoints are worked replay examples for the cloud and burst studies.

## Regenerate numerical experiments

Use the checkout's Particula environment. Run the following deterministic pilot studies from the repository root:

```bash
PYTHONPATH=. python studies/parcel-composition/exchange.py
PYTHONPATH=. python studies/parcel-composition/population_metrics.py
```

For the four-species experiment, follow the [aging runner](aging/README.md). For the six-hour boreal challenge, perturbations, coating attribution, redraws, stochastic repeats and adaptive bins, follow the [boreal runner instructions](boreal/README.md) in order. The full-resolution calculations require the pinned optical solver, a C++ compiler and local space for the intermediate reference states. They produce the detailed score tables, draw-by-draw results and additional checkpoint variants described in each chapter. After reproducing all campaigns, pass `--pack` to the figure command to refresh the published CSV bundle.
