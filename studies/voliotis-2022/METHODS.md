# Reproducing the Manchester α-pinene chamber comparison

## Measurements and scoring

The eight EUROCHAMP records, source hashes, archive discrepancies and instrument units are documented in [the measurement packet](data/README.md). The AMS observable is suspended organic material in µg m⁻³ and excludes ammonium sulfate. The archived peaks and declines agree with the raw-mass panel of Figure 3a. The authors' particle-wall-corrected curves and grouped mass yields are separate observables. The analysis retains the reported AMS values, including the small lights-on baselines. The simulation begins with inorganic seed and zero organic mass. [Voliotis et al., Figures 2 and 3 and Tables 1–3](https://acp.copernicus.org/articles/22/14147/2022/)

The primary score is RMSE over the 84 AMS timestamps from lights-on to 4.98 hours, at 216 s spacing. The model predicts at 360 s spacing through 5 h. Linear interpolation aligns the model to the instrument for RMSE, mean bias and mean absolute error. Each peak and its time use that series' saved samples. Rounded peak times therefore have different resolution. The linked CSVs retain all six hours of AMS and about five hours of fractional precursor. Forcing and score intervals follow elapsed time since lights-on.

Fractional α-pinene is sampled from the archive on a 20 min grid, matching the nominal GC-MS acquisition period. The lights-on residual is 1 by normalization. Isotonic regression on values bounded to [0, 1] removes small apparent increases. The residual fraction is interpolated linearly to each model forcing boundary, and an effective rate −ln(F₂/F₁)/Δt reproduces its change over the interval. This rate includes both OH- and O₃-initiated chemistry and can also include physical precursor loss. Nominal ppb converts to mol m⁻³ using 101325 Pa and the initial recorded temperature. Run 6 uses 24 °C from the companion study because its temperature series is missing. Modeling temperature is constant within each run. The initial number concentrations in runs 4 and 5 came from different instruments. [Manchester gas and particle records](data/README.md) · [Companion study](https://acp.copernicus.org/articles/22/13677/2022/)

Only the run 4 AMS score selected the primary fit and the 21 parameter sets within 2 µg m⁻³ of the best RMSE. This fixed tolerance describes the sensitivity of the prediction to fit quality. No sampling-error distribution was assigned. Run 5 organic measurements were excluded from parameter selection. Runs 1 and 6 serve as additional transfers to higher and lower precursor loadings. Their gas histories and initial seed measurements were supplied. In a separate benchmark, one constant coefficient was fitted by least squares through the origin to run 4 suspended mass versus cumulative precursor disappearance and applied unchanged to all runs.

## Reduced chemistry and reversible condensation

The native Particula gas mechanism splits the initial C₁₀H₁₆ inventory between two otherwise identical precursor lanes. A fraction *f* follows a pathway with a net uptake of one O₂ equivalent. It first produces a C₁₀H₁₆O₂ intermediate, then forms a vapor of the same molar mass after a first-order lag. The complementary fraction becomes an unresolved gas reservoir carrying its original C₁₀-equivalent mass. These formulas close the surrogate mass and carbon ledgers. They represent aggregate pathways. The model does not resolve individual products, fragmentation, organic nitrates, RO₂ chemistry, photolysis, or explicit OH and O₃ histories. The potential product **mass** per reacted precursor mass equals *f* × 168.236/136.238 before partitioning and losses. [Native gas mechanism](https://github.com/uncscode/particula/blob/1e7c93928b4a461c2bab63aca80b618b9404da9e/docs/guide/processes/gas-chemistry.md) · [Chamber implementation](chamber.py)

Particula advances gas chemistry using RK23 and then advances reversible isothermal condensation on the same shared vapor inventory. Product vapor has a first-order irreversible wall reservoir. The primary model attributes all measured precursor disappearance to oxidation. A sensitivity subtracts the independent dark precursor loss rate of 2.24 × 10⁻⁵ s⁻¹ up to the measured disappearance in each interval. The separate ozone budget constrains the interpretation of this assumption. It integrates the original gas observations at native resolution for runs 1, 5 and 6. The run 4 ozone file has an unresolved clock. [Shao et al., chamber losses](https://amt.copernicus.org/articles/15/539/2022/) · [Ozone calculation and rate uncertainty](data/README.md)

Each box starts with the Table 1 seed mass and the particle count nearest lights-on. One representative dry sphere per box carries the full concentration weight. A convergence/sensitivity test uses 5 or 11 equal-number lognormal size classes (assumed geometric standard deviation 1.5) rescaled to preserve initial seed mass and number. Ammonium sulfate and organic densities are 1770 and 1400 kg m⁻³. Product diffusivity is 6 × 10⁻⁶ m² s⁻¹ and surface tension is 0.03 N m⁻¹. Both are assumptions. Vapor pressure is fixed from the chosen saturation concentration C* at 298.15 K. Native ideal-molar activity uses equal effective reference molar masses for fully accessible seed and product, yielding surface activity proportional to organic mass divided by organic plus seed mass. Changing the seed reference mass makes 10% of the seed available to the organic phase as a sensitivity. Salt-water thermodynamics, RH-dependent phase separation, polydisperse wall deposition and seed hydration are unresolved. The 10 and 100 µg m⁻³ volatility tests overlap classes discussed in the same-campaign FIGAERO measurements. The 1 µg m⁻³ case tests a lower-volatility effective product. Relative ion signal does not directly specify a model branching fraction. [Volatility companion study, Figure 1](https://acp.copernicus.org/articles/22/13677/2022/)

For the primary simulations a midpoint removal reduces particle number and all suspended particle materials proportionally. Lost organic mass is retained in the wall ledger. Carbon closure sums precursor, intermediate, product vapor, unresolved product, wall reservoirs, suspended organic material and deposited organic material on a C₁₀-molar basis. The reactive mass check includes stoichiometric oxygen uptake from an explicit surplus O₂-equivalent reservoir. The rate-driver lanes are numerical coefficients and excluded from the mass inventories. The calculation does not interpret the artificial oxygen inventory as atmospheric oxygen concentration.

## Parameter grids and conditional tests

| Parameter | Primary calibration | Count-conditioned calibration |
|---|---|---|
| Condensable molar branch | 0.10 to 0.70 in 0.05 steps | 0.10 to 0.70 in 0.10 steps |
| C* at 298.15 K (µg m⁻³) | 1, 10, 100 | Same |
| Vapor accommodation | 0.03, 0.3 | Same |
| Product lag (min) | 5, 25 | Same |
| Product vapor wall loss (h⁻¹) | 0, 0.6, 1.8 | 0, 0.6, 1.8, 4.5 |
| Particle removal | 0.12, 0.24, 0.42 h⁻¹ | Measured number histories with wall-loss shares 0, 0.5, 1 |
| Number of run 4 fits | 1,404 | 1,008 |

The second calibration uses the within-instrument number trajectory from each run. The measured number is normalized to the nearest lights-on observation, interpolated to six-minute intervals and constrained to decrease monotonically. For a retained number fraction *g* during an interval, suspended particle material is multiplied by *g* raised to the wall-loss share, while surviving mean particle mass is multiplied by that material-retention factor divided by *g*. A zero share conserves material during number loss and approximates consolidation. A share of one removes each particle with its material. This calculation brackets idealized fates for the observed count decline. It does not separate count losses by mechanism or simulate a coagulation kernel. Measured numbers provide an independent diagnostic for the primary fit and an explicit input for the second fit. The frozen mass-only and count-conditioned optima are specific to their respective grids.

Injection sensitivity uses the authors' ±15% nominal uncertainty. Multiplying the calibrating injection by a factor *q* and dividing its fitted branch by *q* preserves all condensable precursor in run 4. Applying that rescaled branch to the opposite run 5 injection extreme tests the largest differential error permitted by independent bounds. The extremes are scenarios. A common systematic injection error would largely cancel. For hours five to six the two explicit scenarios stop further precursor oxidation or continue the effective rate of the last 20 min. Both allow existing intermediates to age, reversible exchange, and ongoing wall losses. The range between the two trajectories has no guaranteed coverage of the missing forcing.

## Execution and verification

The calibration ran on the local NVIDIA GeForce RTX 5060 using Particula source version 0.3.3, Python 3.12.12 and Warp 1.11.1. The primary study uses native CUDA for every chemistry and condensation substep. Python handles experimental forcing, bookkeeping, midpoint number changes and interpolation. No CPU fallback is used. The reference NumPy check explicitly selects CPU for the same two chamber histories. Run metadata records the source path and underlying Particula revision. The editable environment's packaging metadata reports 0.2.14, while the loaded source reports 0.3.3. [Run environment](results/runtime_evaluate.json)

From the study worktree root, use the Particula virtual environment with the CUDA driver accessible. The example below uses the existing local environment. The bundled CSVs are sufficient for replay.

```bash
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
PY=/home/kyle/Code/particula/.venv/bin/python
STUDY=studies/voliotis-2022
"$PY" "$STUDY/study.py" train
"$PY" "$STUDY/study.py" evaluate
"$PY" "$STUDY/study.py" envelope
"$PY" "$STUDY/extensions.py"
"$PY" "$STUDY/counts.py"
"$PY" "$STUDY/verify.py"
"$PY" "$STUDY/figures.py"
```

Use only the final plotting command to regenerate figures from saved outputs. The reference outer step is 12 s and number removal/forcing boundaries are 360 s apart. The refined cases use 6 or 3 s substeps with 180 s boundaries. The 12-to-6 s comparison changes both resolutions, while the 6-to-3 s comparison isolates the substep. Negative or nonfinite state, relative C₁₀ or reactive mass errors above 2 × 10⁻⁷ and precursor-forcing errors above 2 × 10⁻⁵ terminate a run. [Numerical checks](results/numerical_checks.json) · [Refinements at the edges of the transfer ensemble](results/transfer_refinement.csv) · [All reference, resolution and physical controls](results/comparisons.csv)
