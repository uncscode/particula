# Carbon dioxide uptake and first ionization in sodium chloride solutions

At 298.15 K the Gibbs-consistent carbonate extension predicts the measured
decrease in CO₂ uptake with salinity. It overpredicts absorbed carbon by
1.1–6.2% across five brines at 1 atm CO₂. The largest relative error is at
2 mol NaCl per kg of water. In the matched first-ionization cell the predicted
conditional pK₁ exceeds the measured value by 0.0099.

## CO₂ solubility

Harned and Davis equilibrated gas with water or NaCl solution and determined
absorbed carbon by weight titration. Their solubility coefficient is mol CO₂
per kg water per atmosphere partial pressure. The CSV includes all their
25 °C series A–C observations in mol/kg at 1 atm. The lower-weight column
records observations starred by the authors. The comparison uses their smoothed
Table II values at each salt concentration. The authors report reproducibility
of about 0.2% at best, with errors approaching 1% under their highest
temperature and salt conditions. [Harned and Davis (1943), Tables I–II](https://doi.org/10.1021/ja01250a059) · [IUPAC–NIST transcription](https://srdata.nist.gov/solubility/sol_detail.aspx?sysID=62_158)

One kilogram of added water defines the analytical feed. The measured
pure-water coefficient fixes the molal CO₂ activity for the pressure
reservoir. A scalar root varies total carbon at each salinity until the
same aqueous CO₂ activity is reached. AIOMFAC resolves aqueous CO₂,
bicarbonate and carbonate at every trial. The water anchor is excluded from
the error summary. No brine measurement was used to fit parameters in this
comparison. The historical provenance of the backend's ion parameters may
include these or related measurements.

| NaCl (mol/kg water) | Measured carbon (mol/kg water) | Predicted carbon (mol/kg water) | Difference |
| ---: | ---: | ---: | ---: |
| 0.2 | 0.03279 | 0.03316 | +1.1% |
| 0.5 | 0.03054 | 0.03134 | +2.6% |
| 1.0 | 0.02726 | 0.02852 | +4.6% |
| 2.0 | 0.02223 | 0.02361 | +6.2% |
| 3.0 | 0.01852 | 0.01954 | +5.5% |

The relative RMS difference is 4.43%. The largest numerical residuals in
these six calculations were 1.9 × 10⁻¹³ in ln(CO₂ activity / target),
3.6 × 10⁻¹⁵ in log mass action, 2.3 × 10⁻¹⁶ in relative carbon balance and
3.8 × 10⁻¹⁶ in normalized charge. For an independent limiting calculation,
the stated Na⁺ and Cl⁻ interaction parameters give
S(m)/S(0) = exp[−2(0.100 − 0.005)m] when the acid species are neglected.
It gives 0.01945 mol/kg at 3 mol/kg NaCl. The small difference from the
speciated solution (0.01954 mol/kg) shows how much of that prediction is
set by the inherited CO₂–NaCl interaction.

## Conditional first ionization

Harned and Bonner determined the concentration quotient
K₁* = m(H⁺)m(HCO₃⁻)/m(CO₂) from electrochemical cells, extrapolating
added NaHCO₃ to zero. The CSV retains the smoothed 25 °C pK₁ and
observed-minus-smoothed difference in Table II. Add them to recover the
observed pK₁. For the 0.1 mol/kg series the paper reports
−log₁₀ m(CO₂) = 1.7440 in its first 25 °C cell. The calculation fixes that
free dissolved CO₂ molality and the NaCl feed, then evaluates the quotient
from the resulting physical species per kg of equilibrium water.
[Harned and Bonner (1945), Tables I–II](https://doi.org/10.1021/ja01222a037)

The observed pK₁ is 6.11910. The Gibbs-consistent and pinned carbonate
treatments give 6.12898 and 6.12906 respectively. Their difference of
0.00008 is smaller than the empirical model discrepancy. The remaining
0.2–1.0 mol/kg series have published pK₁ values, but the authors report
detailed gas conditions for the 0.1 mol/kg series only. Applying the same
free CO₂ molality gives pK₁ differences from −0.0119 to +0.0134. Varying
free CO₂ from 0.005 to 0.030 mol/kg shifts these predictions by at most
0.00011. That range is a model sensitivity check.

These aqueous comparisons cover fixed temperature, salt concentration and
carbon loading. They provide no measured organic-rich phase compositions,
crystal nucleation rates or concentrated carbonate liquid–liquid boundaries.

## Reproduction

With the in-tree backend or its editable installation active, run from the
repository root. Omitting the output argument prints the same JSON.

```sh
PYTHONPATH=packages/particula-aiomfac/src python studies/harned-carbonate/compare.py --output studies/harned-carbonate/results.json
```
