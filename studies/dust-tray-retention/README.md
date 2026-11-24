# Reproduce the dust-tray study

[Read the literature analysis](../../docs/studies/literature/dust-tray-retention/index.md) for the three-hour median forecasts, the wind-holdout comparison and the optical uncertainty. This directory contains the source data, simulation scripts and saved results.

The packet includes the original 49,294-byte workbook. To replace a missing copy, download [Supporting_data_v2.xlsx](https://zenodo.org/api/records/22962214/files/Supporting_data_v2.xlsx/content) from [Jia et al., Zenodo v2](https://zenodo.org/records/22962214). The analysis checks MD5 07464b6f980a24a0b19598d503b7cd99. The workbook's license is CC BY 4.0.

Run the following from this directory in the publication checkout. For an exact physics replay, use a separate checkout of Particula at revision `3435f653ae2606c4eeef23bdf55b617d2ac58586`. Install it in a Python 3.12 environment with NumPy 2.3.4, SciPy 1.16.2 and Matplotlib. The blocked fits also use PyTorch 2.10.0 on CPU. Set JIA_PY to the interpreter for that environment. The saved results suffice for the two synthesis figures and the graphical abstract.

```bash
JIA_PY=python
"$JIA_PY" analysis_figures.py
"$JIA_PY" analysis_toc.py
"$JIA_PY" study.py --still
"$JIA_PY" study.py
"$JIA_PY" monotone_check.py
"$JIA_PY" replot.py
```

- [study.py](study.py) reads the sheet XML and cached values. The four blocks of the `particle distribution` sheet give channel diameters, early and late volume percentages in D:AR. The `original data` sheet supplies endpoint masses in grams, A:D in rows 4–243. Five rows are allocated to each threshold and endpoint. Empty cells are omitted. The `exponential model` sheet supplies mean mass in grams (F) at all five nonzero times. The 10–20 µm series differences the <20 and <10 means. The first and last values agree with the replicate-based means within rounding.
- [monotone_check.py](monotone_check.py) computes the smallest total variation distance that any nonincreasing weights can achieve against each observed endpoint. Every such weighting is a nonnegative sum of step functions that retain successive prefixes of the ordered early size channels. Normalization gives a mixture of those normalized prefixes, so a linear program over mixture weights covers the entire family. A zero early channel stays zero. This retrospective fit is a diagnostic limit. Its inputs are [channels.csv](channels.csv).
- [replot.py](replot.py) regenerates both figures from the saved CSVs without rerunning the model. The study uses exact elapsed times of 1800 and 10800 s and has no integration step or random seed. [Environment](run.json)

## Wind, walls, collisions, humidity, charge and detection

With the same interpreter and checkout as above, run these in this directory. Restricting linear algebra to one thread keeps the many small matrix products inexpensive. The follow-up reads the original workbook and the saved spectra.

```bash
OPENBLAS_NUM_THREADS=1 "$JIA_PY" transport.py
"$JIA_PY" measurement_check.py
"$JIA_PY" transport_plot.py
```

[Wind and microphysics results](transport.md) · [Measurement sensitivity](uncertainty.md) · [Paired simulation outputs](transport-results.csv)

For the subsequent [frictional charging tests](tribocharging.md), run `OPENBLAS_NUM_THREADS=1 "$JIA_PY" tribocharging.py` from the same directory. [Results](tribo-cases.csv) · [Inferred-source charge by size](tribo-source-charge.csv)

## Settling, impact rebound and release from the tray

Run these commands in the same directory and environment. The simulation reads the workbook, the existing channel and threshold tables, and model functions from the previous two studies.

```bash
OPENBLAS_NUM_THREADS=1 "$JIA_PY" rebound.py
"$JIA_PY" rebound_plot.py
```

The first command saves five retrieval times for each fixed-surface configuration, selected three-hour channel inventories, and separate tables for abrupt and gradual changes in adhesion, the time profile, measurement intervals and delivery bounds. The exact-cutoff band gain interpolates cumulative mass at 10 and 20 µm on log diameter. The adjacent channel-label band gain sums bins with reported diameters in (10, 20] µm. The plotting command uses the exact-cutoff column. [Study and methods](rebound.md) · [Simulation](rebound.py) · [Figure source](rebound_plot.py) · [Channel inventories](rebound-channels.csv)

## Automatic differentiation and blocked refits

Use PyTorch 2.10.0 in the study environment and run these commands after the base study has produced its CSVs.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for batch in core stress extra combined measurement charge
do
    "$JIA_PY" global_fit.py --set "$batch" --workers 5
done
"$JIA_PY" global_summary.py
```

The six JSON files preserve each fold’s training objective, independent start results, fitted parameters, held-out metrics and channel mass at each wind and retrieval. The summary regenerates the figure and score tables, reruns frozen physical parameters on a one-minute grid, verifies that permuting withheld responses leaves training unchanged at 2 m/s, and checks the speed-weighted capture integral against numerical quadrature. Seeds 173 and 547 control the Sobol screen and Latin-hypercube starts. The supplementary external deposition inputs are the central values and reported spreads in Zhang et al.’s Appendix B, with Goossens’ wind-to-friction-velocity pairs. [Study and sources](global-fit.md) · [Design](global-fit-spec.md) · [All-wind vectors with units and bounds](global-shared-sets.csv) · [All fitted vectors](global-parameters.csv) · [Held-out errors](global-scores.csv)

## Optical sizing and uncertainty in the recovered spectra

Run these commands from the packet after the blocked refits above, using the same study environment.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$JIA_PY" optical_test.py
"$JIA_PY" optical_limits.py
"$JIA_PY" optical_summary.py
```

The first command builds the transit and scattering responses, fits the eight endpoint spectra and evaluates synthetic ring noise. The second finds the smallest ring discrepancy that admits a competing interpretation and repeats the integration at finer angular and within-channel diameter resolution. The third passes each early spectrum through the frozen wind-holdout predictor, compares its late prediction with the corresponding reconstruction, checks the 6 m/s tray assignments and regenerates the figure. [Design](optical-spec.md) · [Optical study](optical-study.md) · [Numerical checks](optical-checks.json)
