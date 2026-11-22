# Replay the post-flame sulfur and potassium reactor study

The frozen forward model calculates 36 hot-quartz SO₂ outlet concentrations and 49 cooled-reactor aerosol readings. It runs the four published selections with the same chemistry, thermal histories, bulk nucleation, finite cluster ladders and instrument windows used in their comparison. All fits share coefficients across feeds. The two combined fits and the two chemistry controls use their own selected coefficients. [Measured readings and working bands](inputs/observations.csv) · [Parameter vectors](inputs/parameters.json)

From the repository root, use Python 3.12+, NumPy, SciPy and NVIDIA Warp. The verified environment used Warp 1.11.1. A CPU run compiles the kernels on first use and writes 340 predictions and their budget checks.

```sh
python scripts/straw_combustion/replay.py --device cpu --output straw-results
```

Use `--device cuda:0` for a CUDA device. Use `--fits split_combined` for the training-only selection. The default mesh uses two equal-flux annuli, a 0.0001875 s maximum aerosol step, four subcritical states per core and 256 dry reactor steps. `--annuli` and `--dt` support fixed-coefficient resolution checks. The default mesh verifies every computed value against its archived reference to within 0.000002 of its working band. On a different mesh, diagnostics record its change from the published grid.

The same host function supports a percent-script or notebook from a checkout on the Python search path.

```python
from scripts.straw_combustion.replay import run_replay

rows, diagnostics = run_replay(device="cpu", fits=("split_combined",))
```

`predictions.csv` gives the fit, original row number, dataset, case, target, dry reactor temperature (K) when applicable, measured value, working tolerance, original training mask, computed prediction, signed residual in tolerance units, and archived reference prediction. The latter is read after simulation. `diagnostics.json` contains the maximum K/S inventory error (ppm), minimum state value, maximum difference from the reference and each subset's cost and coverage. Cost is half the sum of the family means of squared standardized residuals within each subset. The original training and withheld partitions contain 56 and 29 readings. In the all-data fits both partitions were available during selection.

| Frozen selection | All 85 in band | Original training in band | Original withheld in band |
|---|---:|---:|---:|
| All-data combined | 49 | 34 | 15 |
| Training-only combined | 44 | 31 | 13 |
| All-data chemistry control | 47 | 33 | 14 |
| Training-only chemistry control | 44 | 31 | 13 |

The split combined selection lands on the chemistry control with inactive auxiliary cluster formation. In the all-data combined selection, the chosen path mask enables binary and third-species pathways. The sulfate birth, KCl birth, vapor deposition, wall reservoirs and gas reaction network remain present in the chemistry controls. The archive reproduces these four forward predictions. The distinct 525-channel neutral KCl model, signed charged-cluster study and optimization histories have different state spaces.

The impactor conversion solves ρ₁d₁²Cc(d₁) = ρ₂d₂²Cc(d₂) for compact spheres using the room-temperature mean free path λ = 66 nm and Kn = 2λ/d. With Cc = 1 + Kn [1.257 + 0.4 exp(−1.1/Kn)], a 100 nm particle has Cc = 2.8887. The archived predictions include this correction. All 58 previously selected or retained candidate vectors were rescored on the original calibration masks. A paired detector-only sensitivity test froze each of the four published reactor models and fitted the shared geometric standard deviation (1.10–2.80) and relative mode diameter (0.28–3.60) by bounded least squares. It used the same family-weighted residuals as the reactor fit and excluded held-out rows from the training-only objective.

## Input map and provenance

| Input | Role in the calculation |
|---|---|
| [observations.csv](inputs/observations.csv) | Original measurement order, units by target, working tolerance, fit mask and residual family |
| [dry_feeds.csv](inputs/dry_feeds.csv) | Dry 1.05 atm quartz-reactor inlet SO₂ and SO₃ in ppm, O₂ in percent, furnace temperature in kelvin and residence time in seconds |
| [aerosol_feeds.csv](inputs/aerosol_feeds.csv) | Twelve entries in simulation order, with KCl formula equivalents and SO₂ in ppm, O₂ in percent, and the flag for the separate sulfur-free concentration series |
| [temperature_wall.csv](inputs/temperature_wall.csv) and [temperature_center.csv](inputs/temperature_center.csv) | Wall measurement and calculated centerline temperature along the tube, in °C against axial distance in cm |
| [dry_equilibrium.csv](inputs/dry_equilibrium.csv) | Calculated high-temperature equilibrium constants for the seven reversible reactions at the 13 measured temperatures |
| [salt_vapor.csv](inputs/salt_vapor.csv) | Calculated log₁₀ pure-salt monomer and dimer partial pressures in bar, interpolated against reciprocal temperature |
| [parameters.json](inputs/parameters.json) and [parameter_map.csv](inputs/parameter_map.csv) | Frozen joint parameter vectors and zero-based positions, including the tested forms and source links for the 60 perturbations |
| [reference_predictions.csv](inputs/reference_predictions.csv) | Archived output per measurement and fit for the default mesh |

Rows 0–35 use the dry gas conditions and predict SO₂ in ppm. Rows 36–72 use the six Table 3 feed mixtures. The seven possible targets, in detector order, are number in millions per cm³, mobility number geometric mean diameter in nm, impactor mass in mg/m³, inferred mobility mass in mg/m³, impactor mass geometric mean diameter in nm, mobility mass geometric mean diameter in nm, and sulfate mass percent. Each available case/target has its own row. Rows 73–84 use the Figure 14 sulfur-free concentration series at 104–364 ppm KCl and predict number and inferred mobility mass. The reactor is supplied with 3.5 L/min at 293.15 K and 1 atm. The measured wall and modeled center temperature profiles are interpolated over their common axial interval. Working bands use reported count scatter where available and study-assigned comparison tolerances for other readings. Several readings are moments of the same particle-size scan. [Zeuthen thesis, Figure 7, Table 3, Figure 14 and Appendix A](https://backend.orbit.dtu.dk/ws/files/4716854/JZ-Thesis%2009-07-07%20-%20final.pdf) · [Dry SO₃ decomposition experiment](https://doi.org/10.1021/jp0557215)

The first seven x coordinates feed gas and phase kernels. Coordinates 7–8 control the geometric standard deviation and relative diameter in the detector model. Coordinates 9–28 implement the established gas and particle hypotheses, and 29–68 supply the follow-on dry and wet network. The dry calculation uses coordinates 29–40. All u values apply to the common three-core cluster operator and its attachment, binding and ion source. The last position is reserved. The numerical values are model-derived, including study fits of equilibrium constants. The forward operator draws on measured dry sulfur rates, wet potassium reaction pathways and pure KCl vapor/dimer thermodynamics. [Dry sulfur oxidation](https://doi.org/10.1021/jp0557215) · [Chanpirak et al. wet network](https://doi.org/10.1016/j.fuel.2024.130974) · [JANAF condensed KCl](https://janaf.nist.gov/tables/Cl-038.html) · [KCl gas](https://janaf.nist.gov/tables/Cl-039.html) · [KCl dimer](https://janaf.nist.gov/tables/Cl-091.html)

The dry equilibrium columns K1–K7 follow, in order, SO₃ ⇌ SO₂ + O, SO₃ + O ⇌ SO₂ + O₂, 2 O ⇌ O₂, 2 OH ⇌ O + H₂O, SO₃ + OH ⇌ SO₂ + HO₂, O + HO₂ ⇌ OH + O₂, and OH + HO₂ ⇌ O₂ + H₂O. K1 is in mol/m³ and K3 in m³/mol, with the remaining values dimensionless. These are reduced values evaluated at the experimental temperatures under a one-bar thermochemical standard state. Salt vapor pressure inputs are differences of JANAF formation constants. Auxiliary mixed-core stabilization, detector response and radial temperatures include modeled quantities. An agreement between a recalculation and the archive checks the implementation. Transfer to measured feeds tests the joint physical assumptions.
