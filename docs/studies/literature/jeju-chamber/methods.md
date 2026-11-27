# How the Jeju chamber and its instruments constrain the simulation

![Inner and outer chambers, instruments and expansion timeline](../../../images/jeju-chamber/setup.svg)

*Figure 1. Cooling and pumping set the expansion history. Welas and the cloud particle imager (CPI) sample near the underside of the chamber. The vapor analyzer samples a warmer region. Equipment positions are schematic. The timeline shows the order of operations. [Facility schematic](https://amt.copernicus.org/articles/18/3781/2025/) · [Experiment sequence](https://doi.org/10.1029/2026GL123395)*

The authors ran six expansions in October–November 2025 at Seogwipo, Jeju. The pooled and individual models use chamber-condition particle concentrations and the original 70 Welas bins. Each minute plotted is the mean of its available measurements. The four air-probe mean, coldest air probe, six-probe wall mean and pressure describe the evolving thermal boundary. Only the aggregate wall temperatures are recoverable from the deposited files. [Author data, version 1](https://doi.org/10.6084/m9.figshare.31744108.v1)

| Run | Fill and salt addition | Nominal local injection time and provenance | Pressure and mean air temperature at addition | Vapor |
|---|---|---|---|---|
| 1 | Ambient air, no added salt | Unsalted control | — | Measured |
| 2 | Ambient air, 5 g NaCl with 1% anticaking agent | 11:16, published | 710 hPa, 5.85 °C | Measured |
| 3 | Ambient air, pure NaCl, nominal 5 g repeat | 16:16, inferred | 700 hPa, 5.32 °C | Measured |
| 4 | Ammonium sulfate, 5 g NaCl | 13:56, inferred | 713 hPa, 5.23 °C | Measured |
| 5 | Ammonium sulfate and Arizona Test Dust, NaCl dose unverified | 17:06, inferred | 688 hPa, 4.15 °C | Measured |
| 6 | Colder ambient air, NaCl dose unverified | 15:16, inferred | 700 hPa, −2.31 °C | Unavailable |

*Table 1. Inferred clocks are given to the minute. In run 5 coarse particles were present before injection. The renewed warm pulse helped locate addition. The source supplement could not be recovered for independent verification of the last two doses.*

### From outdoor air to the initial particle population

For ambient runs 1, 2, 3 and 6, we average the first 60 seconds of in-chamber Welas measurements in each optical diameter bin. These concentrations initialize the resolved particles in all 18 cells. Each run thus carries its own measured size distribution after filling. The fit adds a lognormal population of smaller cloud condensation nuclei (CCN) and estimates an effective ice-active abundance for each run. Salt introduced later enters as a separate population. Runs 4 and 5 use prepared sulfate and sulfate–dust fills, respectively.

Park and colleagues report 2,400–8,100 ambient aerosol particles/cm³ and 580–1,370 CCN/cm³ at 0.6% water supersaturation across the ambient fills. These published ranges provide an independent reference. The fitting objective uses the chamber spectra, vapor and imager evidence. A quantitative check against the CCN observations would require evaluating activation at the counter’s 0.6% supersaturation for each corresponding fill. [Ambient measurements and Table S1](https://doi.org/10.1029/2026GL123395) · [Chamber observations](https://doi.org/10.6084/m9.figshare.31744108.v1)

| Kind of evidence | What enters the calculation |
|---|---|
| Measured forcing and initial populations | Pressure, air and wall thermometry, pre-cloud optical distributions and available LI-COR water vapor. The A2 fill uses the supplier’s dry-volume grading and the measured pre-cloud mixture. |
| Independent instrument calibration | Welas was calibrated with MonoDust 1500. The study follows the published water-equivalent optical sizing. A physical dry diameter, a Welas diameter and a CPI projected length can differ for the same particle. |
| Fitted or inherited effective quantities | Circulation and exchange, heat transfer, CCN loading and size, wet growth and activation, vapor response and wall uptake, dry-particle loss, salt delivery, ice-active abundance and ice growth. Previously inferred instrument response and analyzer correction are frozen in the final fits. |
| Assumptions tested | Spatial vapor contrast, wall-temperature contrast, aerosol delivery and optical response, collision efficiencies, unresolved humidity in run 6 and removal of selected processes. |

*Table 2. A bulk vapor measurement does not identify local supersaturation at the optical sampling point. All chamber spectra were compared in measurement space. CPI length fractions and recognized plate habits separately constrain phase. [Manufacturer A2 grading](https://www.powdertechnologyinc.com/product/iso-12103-1-a2-fine-test-dust/) · [Jeju methods](https://doi.org/10.1029/2026GL123395)*

[Six-run results and conclusions](index.md)
