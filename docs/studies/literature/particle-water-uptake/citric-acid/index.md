# Citric acid droplet hygroscopicity

## Comparison specification

Use all 12 measured points for aqueous citric acid at 293.15 K in Marsh et al. (2017), Figure S9.1 and Table S9.2. The authors report water activity, positive and negative activity errors, solute mass fraction (MFS) and MFS error. The plotted UNIFAC curve is a model prediction. Retain the measured points and their published error columns. [Primary paper](https://doi.org/10.5194/acp-17-5583-2017) · [Supplement, p. 11](https://acp.copernicus.org/articles/17/5583/2017/acp-17-5583-2017-supplement.pdf) · [Data archive](https://doi.org/10.5523/bris.3eoilb28zwoqr2jifjxj7cntex)

This case was selected because it has a traceable composition observable with errors on both axes and a separate published BAT reference for the same organic. The alternative water/glycerol evaporation study reports a time-dependent radius that also depends on thermal and transport assumptions. [Time-resolved evaporation study](https://doi.org/10.1080/02786826.2011.652750)

Freeze the primary model inputs at the values used for the published citric acid BAT curve (Fig. 2b, water/organic molar mass ratio 0.0938, O:C 1.16667, organic density 1469.184958 kg/m³, hydroxyl-equivalent fit). Compare activity evaluated at the reported MFS against the measured water activity. Use signed mean error, RMS error, maximum absolute error, and the count of points consistent with the reported two-axis uncertainty envelope. Compute that envelope from the extrema of BAT activity across the reported MFS interval and the asymmetric experimental activity limits. Fit no parameters. Report a sensitivity using the experimental melt-density estimate (1550 kg/m³, Table S9.1) separately. The original activity fit used 298.15 K and lacks an explicit temperature correction.

As a kinetic cross-check, hold dry citric acid mass fixed and let water exchange with a prescribed vapor reservoir until the flux vanishes. For bulk activity comparisons take the flat-surface limit. Use distinct initial water contents and refine solver and final-time settings until the recovered activity varies by less than 0.0001. Report water exchange with the reservoir and any remaining mismatch to the standalone root. These data quantify hygroscopicity, so do not infer relaxation times from Figure S9.1.

## Direct activity comparison

Using all 12 Table S9.2 rows in the downloadable study bundle, the signed bias (predicted minus measured) was +0.05096 in water activity, RMS error 0.06160, and maximum absolute error 0.10645. None of the 12 predicted intervals overlapped the corresponding measured interval under the two-axis rule above. Using the measured melt-density estimate increased the RMS error to 0.06727. An inverse composition root reproduced the prescribed activities to within 1.5 × 10⁻¹³ for both densities.

These residuals exceed the tabulated measurement errors for this configuration. The model uses reduced organic descriptors, and its room-temperature fit was applied at the experimental temperature without an activity-temperature correction. This comparison alone cannot assign the discrepancy to either approximation. The [executable comparison](citric_acid_hygroscopicity.py) integrates fixed-reservoir kinetic exchange from water-to-solute mass ratios of 0.05 and 80 at each of the 12 activities. Doubling the terminal time from 2000 to 4000 seconds while tightening the relative solver tolerance from 1e-7 to 1e-9 passed the 0.0001 endpoint activity criterion. The fine runs reproduced imposed activity within 3.4e-15 and the independent root mass fractions within 8.0e-14. Dry radius and diffusivity set the diagnostic time scale. These endpoints carry no inference about experimental drying times.

At the measured water activity closest to 0.75 (0.74592), the example also runs the native isothermal condensation process with a fixed water reservoir, transition-regime transport and finite-radius Kelvin pressure. Both initial droplets approach a citric acid mass fraction of 0.69009270. The independently solved Kelvin-corrected equilibrium gives the same result within 1e-6. Native output recording spans a 10 s checkpoint, and the resumed particle masses at 2000 s equal those from uninterrupted continuation.

To reproduce from a fresh checkout, use Python 3.12 or later and the Stage 3B branch. In a virtual environment from the repository root, run the following commands. The example reads the source CSV directly, prints the principal metrics, and displays the figure.

```bash
git clone --branch codex/stage-03b-scientific-comparisons https://github.com/Gorkowski/particula.git
cd particula
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . matplotlib
python docs/studies/literature/particle-water-uptake/citric-acid/citric_acid_hygroscopicity.py
```

On a headless machine, save the figure from the repository root:

```bash
MPLBACKEND=Agg python - <<'PYTHON'
from runpy import run_path

example = run_path("docs/studies/literature/particle-water-uptake/citric-acid/citric_acid_hygroscopicity.py")
figure = example["plot_results"](example["simulate"]())
figure.savefig("citric-acid.png", dpi=150)
PYTHON
```

A maintainer-side clean installation of revision `01755e91ce` reproduced the two-panel figure and the reported bias and RMS error using Python 3.14.8, NumPy 2.5.5, SciPy 1.18.1, Warp 1.19.0 and Matplotlib 3.11.2. Only the editable Particula checkout and Matplotlib were requested from pip.

To rebuild the saved figure and data in the documentation development environment, run `python scripts/docs_examples.py refresh citric-acid-hygroscopicity`. The downloadable example includes the measurement CSV.

## AIOMFAC binary curve

McGill's AIOMFAC-web 3.15 evaluated a single liquid phase at 293.15 K with water and predefined neutral Citric_acid (component 505). Select mass fractions and enter the temperature and citric acid mass fraction from each row of the [52-point curve](https://github.com/Gorkowski/particula/blob/codex/stage-03b-scientific-comparisons/docs/data/thermodynamics/aiomfac-web315-citric-acid.csv). The grid spans 0 to 0.975 in steps of 0.025 and includes the 12 exact experimental compositions. The output water activity is the water mole-fraction activity `a_x(01)`. All component flags were zero. [McGill calculator](https://aiomfac.lab.mcgill.ca/model.html) · [AIOMFAC parameterization](https://doi.org/10.5194/acp-11-9155-2011)

| Difference at the 12 measured compositions | Mean signed | RMS | Maximum absolute |
| --- | ---: | ---: | ---: |
| BAT minus Marsh | +0.05096 | 0.06160 | 0.10645 |
| AIOMFAC minus Marsh | +0.05351 | 0.06370 | 0.10855 |
| BAT minus AIOMFAC | −0.00254 | 0.00273 | 0.00376 |

Both models overpredict particle-phase water activity at all 12 central compositions. The largest experimental discrepancy for each occurs at mass fraction 0.76226, where the measured activity is 0.53894, BAT gives 0.64539 and AIOMFAC gives 0.64749. BAT was parameterized using AIOMFAC predictions, so close model agreement is evidence about the reduced representation for this binary. The Marsh measurements provide the experimental check. These comparisons concern bulk solution activity and do not test kinetic rates. [BAT model development](https://doi.org/10.5194/acp-19-13383-2019)

Data attribution: Marsh et al. (2017), Table S9.2 in the article supplement, reproduced under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) with column headings normalized to machine-readable names. Numeric values are as published.


## Handoff

The quantitative observable for a future inference study is water activity at a measured citric acid mass fraction and 293.15 K. The present comparison fitted no parameters. A future fit would require a separate validation dataset. The two-species humidity pulse in Stage 3A supplies a matched CPU/CUDA forward configuration for Stage 4. This 12-point study establishes experimental residuals and makes no speedup claim.
