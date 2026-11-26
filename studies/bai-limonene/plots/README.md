# Replot the Bai limonene study

The [compact plot tables](../../../website/public/study-data/bai-limonene/) are sufficient to render all six figures on CPU. From the repository root with Particula, Matplotlib, pandas and NumPy installed, run

```bash
python studies/bai-limonene/plots/render.py
```

The default destination is [docs/images/bai-limonene](../../../docs/images/bai-limonene/). Pass `--data DIRECTORY` or `--output DIRECTORY` to use a different copy. The publication charts use `particula.default`. The independent gallery entry point, `python -m scripts.render_literature_tocs bai-limonene`, exports the 3-inch square as a self-contained SVG using `particula.toc`.

| Figure | Plot table | Source and observable |
| --- | --- | --- |
| One week per month under northern Finnish sunlight | `monthly.csv` | Match the 48 monthly, ranked cloud and radiation records by month, background (`profile`) and updraft (`w`, m/s). The columns give sun-aged minus dark-aged cloud-top number (cm⁻³), wet median diameter (µm) and above-cloud reflected shortwave over the local surface (W/m²). Flux includes dark hours. |
| Seasonal daily cycles | `diurnal.csv` | Same three changes across 24 local apparent solar hour bins in each of four seasons. Each season averages three seven-day windows. Hour 0 spans 00–01 and is drawn at its midpoint. |
| Which droplet sizes gain or lose number? | `size-spectra.csv.gz` | Sun and dark concentrations and their difference per 0.25 µm bin within the common 3.5–40 µm wet-diameter window. Limits focus on the part of each spectrum with appreciable counts. All bins remain in the table. Each panel has its own axes. |
| What changes when July aerosol sees sunlight? | `july-cohorts.csv` | The July cohort comparison supplies unsigned lost tagged seeds and gained background particles (cm⁻³). Plot losses with a minus sign. The median contributions (µm) are growth of shared cohorts and the sum of the signed entry and exit contributions. |
| Limonene film water | `film-water.csv` | Digitized dry mass fraction, effective hygroscopicity and plotted hygroscopicity half-bars for S4, S5 and S7 at constant RH, and reconstructed water relative to fresh material. The dotted line locates 30% remaining mass and the horizontal line marks fresh water uptake. |
| Summer cloud graphical abstract | `summer-pair.csv` | The matched JJA clean, 0.25 m/s comparison. The sun-aged reflected value is the dark flux plus the signed flux change above the cloud and local surface. Diameter is number-median wet diameter and determines the circle diameter. |

The [simulation companion](../) describes the source records, laboratory inputs and parcel/optical calculations. After a full scientific replay, rebuild the plot tables and images into separate directories for comparison:

```bash
python studies/bai-limonene/plots/prepare.py --run-root /path/to/replay --output-dir /tmp/bai-plot-tables
python studies/bai-limonene/plots/render.py --data /tmp/bai-plot-tables --output /tmp/bai-figures
```

`prepare.py` selects the ranked season, month and hour summaries from `seasonal/analysis/` and joins the corresponding radiative cases from `seasonal/radiation/`. It selects July from `seasonal/sampling/cohort-months.csv`, the finite-bin spectra from `seasonal/analysis/seasonal-fine-bin-differences.csv.gz`, and the three constant-RH limonene reconstructions from `film/replicates.csv`.
