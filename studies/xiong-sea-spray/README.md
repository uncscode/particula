# Reproduce the Xiong sea-spray study

This companion follows the comparison from Xiong's measured particle growth and chromatograms to the sulfate carried by 200 and 300 nm particles. It also tests how sodium octyl sulfate changes the inferred reaction rate and whether a trial activity model reproduces the appearance of two liquids. All model stages write source copies, calculations and logs to a private directory chosen by the reader. The public figure samples can be used to redraw the article directly. [Figure data](../../docs/data/xiong-sea-spray/) · [Xiong et al.](https://doi.org/10.5194/acp-26-12183-2026) · [Rosati et al.](https://doi.org/10.1038/s41598-021-89346-6)

## Set up the computation

Run the commands from the root of this Particula checkout, with Python 3.12, Poppler utilities on the executable path and a CUDA-capable NVIDIA GPU. The study used Particula 0.3.6 at revision `2db6a77323a7ac119f5acce0c180fe6b9e7b555e` and its in-tree particula-aiomfac 0.1.4 package. The model runner loads both from the repository checkout. The original runs used a GeForce RTX 5060 and an AMD Ryzen AI 9 HX 370. Warp computes the reaction and phase checks on CUDA. JAX fits kinetic parameters in double precision on the CPU.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e . -e packages/particula-aiomfac \
    'warp-lang==1.11.1' 'jax==0.5.3' 'jaxlib==0.5.3' \
    'ml-dtypes==0.5.3' 'opt-einsum==3.4.0' \
    'numpy==2.3.4' 'scipy==1.16.2' 'pandas==2.3.3' 'matplotlib==3.10.7' pillow
study=studies/xiong-sea-spray
work=$(mktemp -d)
python "$study/data/prepare.py" --output "$work/inputs"
python "$study/run.py" --inputs "$work/inputs" --output "$work/run" --stage all
```

The input command restores the graphical readings and retrieves checksummed publisher tables into a reusable cache. Use `--cache PATH` to choose a cache and `--verify-sources` to repeat the figure extraction. `--digitizations-only` restores the graphical readings offline. The paper's tables are required for the complete simulation. [Input preparation](data/prepare.py) · [Publisher locations and checksums](data/publishers.json)

Choose a new destination for each run. The runner copies each required observation by name and records its SHA-256 digest in `run-manifest.json`. Each completed subprocess adds its command, duration, exit status and log path. The manifest records completion after the last stage succeeds. [Run entry point](run.py)

## Choose the scientific comparison

| Stage | Measurement or assumption varied | Calculation and main output in the run directory |
| --- | --- | --- |
| `reference` | Reproduce the pure-salt volume-mixing retrieval from optical growth and test the allowed endmember ranges. | [Inversion](models/inverse.py), [common-endpoint fit](models/joint.py) and [population calculation](models/population.py) produce `population/rates.csv`, `joint/ratios.csv` and `population/at-sizes.csv`. |
| `chemistry` | Compare reaction mechanisms under the same 200 ppm SO₂ exposure and apply them to the Rosati spectra. Test the independent 20 ppm experiment at 100 nm. | [Network selection](models/aiomfac/pathways.py), [kinetic fits](models/aiomfac/fit.py), [sulfate budget](models/aiomfac/evaluate.py) and [cross-dose fits](models/aiomfac/crossdose.py) produce `aiomfac/{primary,retention,eq8}/`, `aiomfac/results/budget.csv` and `aiomfac/crossdose/series.csv`. |
| `coating` | Use rate constants through 33% organic volume to predict the 50% and 70% cases. Compare area loss, shell resistance and dependence on composition. | Runs chemistry first. The [coating calculation](models/aiomfac/coating/predict.py) produces `aiomfac/coating/rate-curves.csv`, `growth-transfer.csv` and `sulfate-transfer.csv` in the same folder. |
| `phase` | Fit water activities of aqueous sodium octyl sulfate and drying onsets in mixtures with ammonium sulfate. Test transfer to NaCl at the microscope's water-vapor pressure. | The [trial interaction model](models/aiomfac/coating/parameter-fit/trial.py), [grid refinement](models/aiomfac/coating/parameter-fit/refine.py), [parameter selection](models/aiomfac/coating/parameter-fit/transfer.py) and [microscope path](models/aiomfac/coating/parameter-fit/chamber_path.py) produce `aiomfac/coating/parameter-fit/final-calibration.csv`, `predictions.csv` and `chamber-path.csv` in that folder. |

Use `--stage chemistry` for just the primary reaction, chromatogram and dose analyses. Add `--controls` for the additional stage settings, including the pilot, time refinement, optical-response variations, fixed backgrounds, size-law challenges and rate profiles. `--stage all` includes all four scientific comparisons and those controls. Multiple named stages share their prerequisites in one directory. [Exact fit settings](models/aiomfac/stages.json)

## Read the chemical and phase results

The reaction calculation starts with one mole of initial sodium. The [water solver](models/aiomfac/thermo/solve.py) finds water amounts that give the specified activity at 298.15 K for 112 fractions of converted Na, five neutral-sulfate product shares and six readout humidities. [Halite equilibrium](models/aiomfac/thermo/halite.py) permits residual solid NaCl. [Independent reference and endpoint checks](models/aiomfac/thermo/reference.py) and [direct off-grid water solves](models/aiomfac/thermo/verify.py) test the resulting 3,360 liquid states. A wet diameter also requires a volume approximation. The main kinetic fits calibrate pure-salt optical endpoints to the observations and vary the low-humidity phase response.

Look first at `aiomfac/primary/inventories.csv` and its counterpart under `retention/`. At 100 nm and 30 minutes, the parallel uptake and neutral-sulfate fit stores 0.045 mol S(IV) per mole of initial sodium. The retained-chloride fit stores zero. Their reserved growth-factor errors are nearly equal, 0.0266 and 0.0267. These predictions explain why measuring sulfur oxidation state would separate the pathways more directly than another late growth curve. The independent [native integration](models/aiomfac/native_check.py) checks the fitted inventory trajectories on CPU and CUDA.

The phase calculation searches for an electroneutral trial liquid whose formation lowers the Gibbs energy of the homogeneous mixture. [Continuous boundary refinement](models/aiomfac/coating/parameter-fit/boundary.py) checks the first instability, and the [finite-difference audit](models/aiomfac/coating/parameter-fit/audit.py) checks chemical potentials and mole balances. `predictions.csv` holds isothermal predictions at 275.15 K. `chamber-path.csv` adjusts temperature to the 610 Pa microscope path. At 50% organic volume, compare the measured 84.0 ± 3.6% RH with about 94.7% RH on that path. A correct prediction of separation at 70% can coexist with a large error at 50%. The fitted octyl sulfate coefficients belong to the study model. Surface coverage and resistance to gas transport require additional measurements.

## Regenerate the figures

Copy the values needed for the figures from a full private run into a new directory and compare them with the compact tables used by the paper.

```sh
python "$study/plot_data.py" "$work/run" --output "$work/figure-data"
```

The [figure renderer](plot_figures.py) and [graphical abstract](toc.py) read the repository's compact tables and write to `docs/images/xiong-sea-spray/`. From the checkout root, run `python "$study/plot_figures.py"` and `python "$study/toc.py"` to regenerate the published assets.

## Replay checks

Clean replays on the RTX 5060 covered the reference inversion, all reaction controls, coating calculations and phase search. All 52 reference and phase tables matched the study calculations byte for byte. All 95 chemistry and coating tables agreed within absolute and relative tolerances of 5 × 10⁻⁸ after omitting runtime columns. Regenerating the five public figure CSVs from these replays reproduced their exact bytes. The 3,360 water solves missed the target water activity by at most 4.2 × 10⁻¹¹. In 18 independent CUDA trajectory checks across six networks and three particle sizes, the largest difference from the fitted integration was 0.000174 mol per mole of original sodium.

The same replay recovered a 300-to-200 nm sulfate mass ratio of 1.79–2.22 at three hours across the 13 fitted treatments and three sea-spray spectra. With the high-load coating data excluded from fitting, the hydrated-shell model predicted 0.346 and 0.166 h⁻¹ for 50% and 70% organic volume, respectively. These checks reproduce the scientific calculations under the specified laboratory forcing. Applying the rates to a different gas exposure or attributing them to a measured shell would require additional evidence.
