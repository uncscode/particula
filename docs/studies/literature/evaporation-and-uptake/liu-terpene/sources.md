# What the published diameters tell us about evaporation

The two overnight readings for humid β-ocimene aerosol come from the same 220 nm chamber experiment. At 1,138 and 1,265 minutes, 41.6% and 41.0% of the apparent dry volume remained. Both readings depend on the same initial diameter. They constrain the shape of one evaporation curve. Independent chamber fills would be needed to determine its reproducibility. [Liu et al., Figure 4](https://aerosol.chem.uci.edu/publications/Irvine/2026_Liu_EST_Air_SOA_evaporation.pdf#page=5) · [Supporting information, Figure S4](https://aerosol.chem.uci.edu/publications/Irvine/2026_Liu_EST_Air_SOA_evaporation_SI.pdf#page=8)

## Two diameter measurements play different roles

Liu et al. oxidized each precursor separately with OH under low-NOₓ conditions and selected the starting particles by electrical mobility. A denuder removed organic vapors before the particles entered the evaporation vessels. The paper describes activated charcoal in the dry vessel. The organic vapor removal rate in the 75% relative humidity (RH) vessel remains uncertain. That rate matters for predicting how fast a smaller particle can lose material. [Original study and methods](https://doi.org/10.1021/acsestair.5c00226)

| Precursor | Selected mobility diameter (nm) | Initial small-particle density (g cm⁻³) |
| --- | ---: | ---: |
| α-Pinene | 165 | 1.24 ± 0.01 |
| β-Ocimene | 220 | 1.35 ± 0.01 |
| β-Caryophyllene | 165 | 1.17 ± 0.01 |
| β-Farnesene | 140 | 1.20 ± 0.01 |

At the outlet, particles were dried before miniSPLAT measured the peak vacuum aerodynamic diameter. The reported volume fraction is the cube of the ratio of that diameter to the initial peak vacuum aerodynamic diameter. The calculation assumes stable effective density and shape. The mobility diameters in the table identify the cohorts and appear on the horizontal axis of the size comparison. [Liu et al., apparatus and density measurements](https://doi.org/10.1021/acsestair.5c00226) · [Density sensitivity](chamber.md)

## How closely do the two published figures agree?

We recovered 48 distinct marker centers from Figure 4 across four precursors and two humidities. Its lines connect observations. The supplement gives fitted volatility-bin fractions and the instrument's diameter spectra. Of the 26 readable humid volume markers, 20 could be paired with a time-labeled, separately readable spectrum. Three early markers overlap, two lack a matching spectrum, and the 128-minute farnesene peak is obscured. The spectrum labels identify the sampling times more closely than the horizontal position of a Figure 4 marker. [Digitized measurements and spectral peak coordinates](https://raw.githubusercontent.com/uncscode/particula/main/docs/data/liu-terpene/digitized.zip)

For each paired point we read the dominant peak and an adjacent sampled diameter in the native Figure S4 panel. The local spacing is about 5 nm for the monoterpenes and 2 nm for the sesquiterpenes. In the primary reconstruction we allow the mode to fall half a spacing to either side of the sampled peak, plus one image pixel for locating it. The allowance on a Figure 4 marker is two pixels, equivalent to 0.40 percentage points vertically. The table spans both ways of reading each sample. The median absolute difference between the marker and the cubed ratio of the sampled peaks across all 20 pairs was 1.32 percentage points.

| Aerosol | Spectral age (min) | Figure 4 volume remaining | Volume from the S4 peak ratio | Reading range |
| --- | ---: | ---: | ---: | ---: |
| α-Pinene | 1,042 | 24.7% | 22.8% | 20.4–25.4% |
| β-Ocimene | 1,138 | 41.6% | 42.3% | 39.3–45.4% |
| β-Ocimene | 1,265 | 41.0% | 39.3% | 36.5–42.3% |
| β-Caryophyllene | 1,144 | 48.9% | 49.5% | 47.2–51.9% |
| β-Farnesene | 1,185 | 63.5% | 68.0% | 63.1–71.3% |

Farnesene shows why the distinction matters. Its marker gives 63.1–63.9% with the image-reading allowance, while its pair of spectra gives 64.7–71.3%. The plotted whisker spans their disagreement. It assigns no probability to values within that span. The normalized spectra also describe a spread in particle sizes, from which we cannot infer the sampling error of the peak without counts or individual diameters. At the study's 26 November 2026 access check, the authors’ Dryad record did not expose the raw measurements. [Dryad dataset](https://doi.org/10.5061/dryad.931zcrk0c)

## Preserve the common initial diameter

For an initial diameter between *a* and *b* and later diameter between *c* and *d*, the spectral fraction can range from (*c*/*b*)³ to (*d*/*a*)³. A model trajectory must admit one initial diameter consistent with all the paired points used in the comparison. Letting the denominator change independently between points would admit combinations the experiment could never produce. The resulting ranges describe how the published record can be reconstructed. Resolving between-fill variability or a change in density would require additional measurements. [How the ranges constrain the models](size-test.md) · [Main study](index.md) · [Replay instructions and source](https://github.com/uncscode/particula/tree/main/studies/liu-terpene)
