# Mt. Soledad plot data

From the repository root, reproduce the [article](../../../docs/studies/literature/particle-water-uptake/mt-soledad/index.md) figures with the study Python environment.

```sh
PYTHONPATH=. python studies/mt-soledad/figures/plot_figures.py
PYTHONPATH=. python -m scripts.render_literature_tocs mt-soledad
```

The renderer writes three SVGs to `docs/images/mt-soledad/`. The graphical abstract takes its count ratio and the two observed wet diameters from [toc-reference.csv](toc-reference.csv). Its [editable template](../../../docs/studies/literature/particle-water-uptake/mt-soledad/toc.svg.in) scales the marker diameters to those field values.

| Plot data | Origin and definition |
|---|---|
| [optical-spectrum.csv](optical-spectrum.csv) | Selected 120–250 nm mobility seeds in `mechanisms/droplets/profiles.csv`. Measured bypass, denuded and modeled response per optical bin, normalized by the bypass total after correcting each branch for its own CPC signal. The two fitted models are the size-independent fractional count loss (`scalar`) and common growth delay (`delay`). |
| [optical-summary.csv](optical-summary.csv) | `mechanisms/droplets/scores.csv`, 0.5 µm optical floor. Count ratios and denuded-minus-bypass shifts in detected wet diameter, averaged over dates. Intervals are 95% date bootstrap estimates conditional on the fitted model parameters. |
| [growth-by-filter.csv](growth-by-filter.csv) | `field-mixtures/comparison/growth-by-date.csv.gz` and `field-mixtures/observations/partial-cases.csv.gz`. Eight held PM₁ filters, lower-solubility sodium-balanced recipe. Each filter contributes one value per dry size and model. Growth factor (GF) means wet/dry diameter. Weak growth is GF ≤ 1.10. |
| [growth-intervals.csv](growth-intervals.csv) | `field-mixtures/observations/partial-held-intervals.csv`. Joint 95% bootstrap intervals on fitted-minus-measured mean growth and weak-growth probability, resampling both training and held filters. |
| [toc-reference.csv](toc-reference.csv) | The common measured 0.8% reference in `field-mixtures/films/final/held-summary.csv`. Inputs are the bypass and denuded mean wet diameter (µm) and the denuded-to-bypass optical count ratio. |

After rebuilding the [scientific stage outputs](../README.md), re-extract the compact tables and redraw the figures with the following commands. Supply the stage output root if it differs from `studies/mt-soledad/source`.

```sh
PYTHONPATH=. python studies/mt-soledad/figures/plot_figures.py --refresh --study-root studies/mt-soledad/source
PYTHONPATH=. python -m scripts.render_literature_tocs mt-soledad
```

The public [worked example](../../../docs/studies/literature/particle-water-uptake/mt-soledad/droplet_response.py) also reads the two optical CSVs. To refresh its stored text, figure and data bundle, run `python scripts/docs_examples.py refresh mt-soledad-droplet-response`. For a plotting-only change, use `python scripts/docs_examples.py replot mt-soledad-droplet-response`.
