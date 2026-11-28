# Reproduce the six Jeju chamber spectra

This directory contains the 18-cell chamber simulation, optical response, fit objective, fitted parameters and plotting code used for the six-run comparison. The replay reads the Jeju author data, reconstructs the pre-cloud airborne mixture for run 5, advances aerosol, droplets, ice and vapor from the fill, and samples the modeled chamber on the original 70 Welas bins. Choose the full-record fit to reconstruct an experiment and the prefix fit to test its final ten measured minutes under recorded forcing. [Jeju paper](https://doi.org/10.1029/2026GL123395) · [Park et al., versioned observations](https://doi.org/10.6084/m9.figshare.31744108.v1)

## Environment and input files

The study used Particula source revision `2db6a77323a7ac119f5acce0c180fe6b9e7b555e`, Python 3.12.12, Warp 1.11.1, NumPy, pandas, SciPy and Matplotlib. The current replay imports the spectrum functions from the checkout containing these scripts. The original device was an NVIDIA GeForce RTX 5060. The same study kernels accept `--device cpu`. The raw file cache may initially be empty. On first use the reader downloads the optical, imager count, thermometry and vapor files from Figshare and verifies each file against the checksum for version 1. A completed cache can be used offline. [Versioned file URLs and MD5 values](source-files.json)

Run 5 uses the supplier's lower and upper cumulative dry-volume limits at ten diameters. The table, source URL, treatment of the unresolved fine fraction and conversion to airborne number are in the dust model. An optional `--a2-table FILE` reads a three-column CSV with diameter in µm, lower volume percent and upper volume percent. It replaces the published powder specification for a new sensitivity experiment. [Powder Technology A2 limits](https://www.powdertechnologyinc.com/product/iso-12103-1-a2-fine-test-dust/) · [Airborne loading and dry-to-optical mapping](dust.py)

## Regenerate all main figures

Start at the root of a checkout with the study code. Set `PYTHONPATH` to that checkout. Set the local root to a directory with scratch space for input files, transport tables, compiled kernels and calculated trajectories.

```bash
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1
study="$PWD/studies/jeju-chamber"
local_root="${TMPDIR:-/tmp}/jeju-reproduction"
python "$study/replay.py" --raw "$local_root/raw" --cache "$local_root/cache" \
    --out "$local_root/results" --dt 1 2
python "$study/spectra.py" --input "$local_root/results" --out "$local_root/figures"
python "$study/replay.py" --raw "$local_root/raw" --cache "$local_root/cache" \
    --out "$local_root/challenge" --dt 2 1 --run5-challenge
```

The default replay regenerates the six individual fits and the pooled fit at each resolution. At 1 s the local model uses 80 background cohorts, 168 ice sections and 48 salt cohorts. At 2 s these counts are 40, 84 and 24. The collision table has 36 size nodes per material. Each positive input file checksum is tied to the source record. The row and column ordering, observation units and selection masks are preserved throughout. Each fit carries the physical state across the expansion without a later optical reset.

| Result | Source and local output |
| --- | --- |
| Size–time spectra and signed independent-fit residuals | [Chamber solver](replay.py) writes `observations.npz`, `pooled-full-{1,2}s.npz` and `R{1…6}-full-{1,2}s.npz`. [Renderer](spectra.py) writes `six-spectra.png` on a common concentration scale. |
| Whole-record mean and the three chosen minute cuts | The same renderer writes `mean-distributions.png`, `minute-distributions.png` and `window-counts.csv` from the same observations and 1 s simulations. |
| Graphical abstract | Run 1 record means for the measured spectrum and the shared six-run fit use the same valid minutes. For the small figure, sum counts in groups of four adjacent Welas bins (two in the final group), then divide by each group’s logarithmic width. This gives 18 bands across the full range and preserves the integrated concentration of each curve. The card displays the 2–40 µm portion on a logarithmic axis. Rebuild the paths with `python studies/jeju-chamber/toc_spectra.py "$local_root/results"`, then render the editable vector with `python -m scripts.render_literature_tocs jeju-chamber` from the repository root. |
| Dependence on detection floor and occupied-bin selection | The renderer rescores the same physics at both resolutions, writing `score-sensitivity.csv`. These sensitivity scores use fixed trajectories. |
| Run-5 starting material × vertical vapor contrast | [Perturbation driver](sensitivity.py) changes source allocation and the imposed vertical vapor contrast under the selected run-5 full fit. It writes `sensitivity.csv` with all eight scenarios, the measured and simulated concentrations in three bands, vapor RMSE and budget closure. |
| Withheld-window reconstructions | Use `--case pooled-prefix R1-prefix R2-prefix R3-prefix R4-prefix R5-prefix R6-prefix` and read the final ten valid times for each run. A case such as `R5-aerosol_prefix` instead varies only the aerosol-related and applicable initial-condition parameters. |

The `y` arrays give chamber-condition number per optical bin in cm⁻³. Divide by the width in log₁₀ diameter for number per decade. The boolean `valid` array identifies usable Welas minutes. The final ten valid minutes begin at each run's `cut`. Imager counts and projected lengths remain separate from the water-equivalent optical diameter. The full-record curves use observations from the entire corresponding fit window.

## Repeat the parameter search

The archived search used 65% optical-bin error, 20% band-number error, 10% vapor error and 5% imager length and habit evidence, renormalized over available evidence. The full record and prefix use separate candidate rankings. Each candidate evaluation integrates forward from the fill. The three fixed start vectors, parameter bounds and units accompany the code. Two promising separated starting points in the Sobol screen each generate three rounds of paired perturbations. Automatic differentiation of Gaussian weights over the real solver evaluations proposes additional parameter steps. The leading distinct candidates are reranked at 2 s, then the winners are replayed at 1 s with doubled cohort resolution. [Screen and search](refit.py) · [Gradient and finite-difference check](gradient.py) · [Score definition](scoring.py) · [Physical bounds and starting fits](search-design.json) · [Selected parameter sets](settings.json)

```bash
python "$study/refit.py" pilot --raw "$local_root/raw" --cache "$local_root/cache" --out "$local_root/search"
python "$study/refit.py" screen --raw "$local_root/raw" --cache "$local_root/cache" --out "$local_root/search"
for fit in pooled R1 R2 R3 R4 R5 R6; do
    for window in full prefix; do
        python "$study/refit.py" optimize --run "$fit" --window "$window" \
            --raw "$local_root/raw" --cache "$local_root/cache" --out "$local_root/search"
    done
done
```

The portable search evaluates candidates sequentially and reuses the six-run screening library. The published curves replay the selected physical settings directly. For a selected case, the first four fitted coordinates determine circulation speed, diffusivity, film heat transfer and the wall temperature contrast. The microphysics array uses the original parameter ordering in `microphysics.py` and the additional indices listed in `refit.py`. Unmeasured run-specific vapor in run 6 is represented by its fitted initial condition. Image evidence enters the score through the two selected plate observations and the archived CPI length fractions.

## Model components

[Observations and forcing](observations.py) align the original time stamps. [Thermal forcing](thermal.py), [18-cell circulation](grid.py), [conservative exchange](flow.py) and [measured cold-probe constraint](measured_forcing.py) produce the local histories. [Particula coefficient tables](native.py) depend on measured temperature and pressure. [Microphysics](microphysics.py) and [collision pairs](pairs.py) implement the study's particle, vapor and boundary updates. [Solver and optical sampling](solver.py) apply mixing between updates and the published study's port weighting and optical smoothing. [Refined numerical transport](precise.py) carries concentration-weighted volume through cell exchange.
