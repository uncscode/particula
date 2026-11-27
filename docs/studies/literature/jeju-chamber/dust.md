# Why the A2 dust correction leaves a deficit in run 5

The run-5 reconstruction starts with too few coarse particles, even before cloud formation or salt injection. The Welas counter measured 90.6 particles/cm³ at 5–10 µm and 23.8 particles/cm³ at 10–20 µm. Combining Arizona Test Dust (A2) with the sulfate size distribution inferred from run 4 gave only 17.5 and 0.53 particles/cm³ in those bands. Allowing preferential loss of coarse powder during delivery widened the gap.

The manufacturer specifies how dry powder volume is distributed by physical diameter. To infer what Welas would see, we must also estimate the number of particles below the smallest tabulated diameter (0.97 µm), how many particles of each size reach the instrument, and how the counter sizes mineral particles. The supplier’s measurements alone cannot determine the airborne number distribution. [A2 grading limits](https://www.powdertechnologyinc.com/product/iso-12103-1-a2-fine-test-dust/) · [Jeju chamber data](https://doi.org/10.6084/m9.figshare.31744108.v1)

![Manufacturer grading, inferred airborne sulfate and dust, and the measured pre-cloud mixture](../../../images/jeju-chamber/source-fit.png)

*Figure 1. The first panel shows the supplier’s dry-volume grading. The next two show the inferred airborne number distribution and its comparison with the measured pre-cloud mixture.*

Adding the A2 constraint did not improve the earlier six-run fit. With chamber physics held fixed, replacing the original fill increased the joint error from 1.000 to 1.027. After refitting, errors were 0.872 for the original fill and 0.915 for A2. On the fine grid the A2 error fell to 0.889, bringing the difference between the refits within the spread due to numerical resolution. This earlier comparison gave equal weight to optical spectra, vapor and cloud particle imager evidence. The later six-run comparison gives more weight to optical spectra.

## The late deficit persists across source and humidity assumptions

We tested the initial mixture and the vertical vapor gradient separately. Starting from the individual run-5 fit to the full record, we simulated each of two fills with and without the fitted vertical contrast in the model’s target vapor concentration. One fill tracks A2 and sulfate separately. The earlier fill initializes particles directly from the optical spectrum. All four simulations retain the recorded bulk vapor forcing, exchange with the walls and transport between chamber cells.

| Source and vapor target | 2–5 µm (cm⁻³) | 5–10 µm (cm⁻³) | 10–20 µm (cm⁻³) | Vapor RMSE (g/m³) | Full-record optical error (decades) |
|---|---:|---:|---:|---:|---:|
| Measured mean, minutes 31–40 | 142 | 292 | 20.9 | — | — |
| A2, fitted contrast | 12.1 | 4.48 | 22.5 | 0.222 | 1.202 |
| A2, contrast removed | 12.2 | 3.97 | 22.6 | 0.221 | 1.202 |
| Earlier optical fill, fitted contrast | 0.25 | 5.31 | 23.5 | 0.222 | 0.980 |
| Earlier optical fill, contrast removed | 0.27 | 4.70 | 23.8 | 0.221 | 0.986 |

*Table 1. Concentrations and vapor root-mean-square error (RMSE) cover minutes 31–40 of uninterrupted, fine-grid simulations. Optical error covers the full record. All four use the chamber parameters fitted with A2 and the vertical contrast.*

The earlier optical fill reduced the full-record optical error but left even fewer 2–5 µm particles late in the expansion. For either fill, removing the imposed vapor contrast changed the late 5–10 µm concentration by less than one particle/cm³ on either grid. Against 292 particles/cm³ measured, the four simulations gave 4.63–6.48 particles/cm³ with a 2 s timestep and 3.97–5.31 particles/cm³ at 1 s with doubled particle resolution. All four fine-grid simulations met the bulk vapor tolerance of 0.25 g/m³. Bulk vapor agreement therefore leaves the exposure of particles at the sampling port unresolved.

We checked eight GPU trajectories and a standard-grid CPU reference. The largest absolute water-budget discrepancy was below 0.0008 g/m³.

## Restarting from the measured spectrum helps more at minute 30

The restarts ask how particles evolve once the model is given the missing population. At minute 20 or 30, we used the measured spectrum to reconstruct the sampled population while retaining particles that carried original A2. On the fine grid those particles accounted for only 2.6% and 2.0% of the respective 2–10 µm restart targets. We then varied what fraction of the remaining 2–10 µm particles could stop growing.

Assigning the entire remainder to the non-growing class gave different outcomes at the two restarts. For the midpoint A2 grading, all three bands stayed within a factor of two in each of the five measured minutes after minute 30, and the fine-grid vapor RMSE was 0.246 g/m³. After minute 20, four of the five minutes met all three optical limits, but vapor RMSE reached 0.269 g/m³, above the 0.25 g/m³ limit. For a restart to pass, the five-minute geometric mean error factor had to stay at or below two in each band and vapor RMSE at or below 0.25 g/m³. No tested assignment passed at both restarts on both grids. The assigned fraction describes how many slow-growing particles this reconstruction needs. The chamber data leave their composition open.

A repeat could measure the dry mineral fraction by size at the sampling port before cooling and at minutes 20 and 30. That measurement would bound how many particles could be surviving dust. Paired humidified sizes would test their water uptake, and a dew-point measurement at the same port would test whether their exposure differed from the bulk vapor record.

[Six-run results and conclusions](index.md)
