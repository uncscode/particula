# Source data for the six Jeju chamber reconstructions

[Download the annotated and figure-extracted observations (ZIP)](../../../../website/public/study-data/jeju-chamber/extracted-observations.zip) · [Input inventory and file checksums](../../../../website/public/study-data/jeju-chamber/provenance.json)

## Chamber observations

The six-run size–time plots and measured size-distribution cuts use the original optical-counter bins in Park et al.’s Figshare version 1. Download the six `OPC_exp*.csv` files for the Welas spectra, the six chamber pressure and temperature tables for forcing, and the six CPI particle tables for projected sizes. Five LI-COR tables give water vapor. Run 6 has no deposited analyzer record. The manifest links all 29 files, including six CPI image archives, and records the publisher’s MD5 checksum for each. [Park et al., experiment](https://doi.org/10.1029/2026GL123395) · [Version 1 data and license](https://doi.org/10.6084/m9.figshare.31744108.v1) · [Versioned file catalog](https://api.figshare.com/v2/articles/31744108/versions/1)

| Extraction in the ZIP | Source and units | Reading convention |
|---|---|---|
| `jeju-cpi-annotations.csv` | Park et al., runs 2 and 6. Each row gives the image archive member, printed particle time in UTC and an image box in original pixels. | Six selected silhouettes with an analyst’s habit description. Match the individual printed UTC time to the corresponding CPI particle table to recover projected length and width in µm. The sheet filename has its own clock. The observed plates establish a time by which ice was present. Sampling and visual recognition limit how close that time can be to freezing onset. [Warm image archive](https://ndownloader.figshare.com/files/62817952) · [Cold image archive](https://ndownloader.figshare.com/files/62817931) |

For run 5, Powder Technology gives lower and upper limits on cumulative dry volume at ten diameters. Estimating airborne number also requires a delivery model and a mapping from dry diameter to optical diameter. We used the archived Welas bins for the comparison. Figure S1 was unavailable for a separate visual check. [Manufacturer’s A2 grading](https://www.powdertechnologyinc.com/product/iso-12103-1-a2-fine-test-dust/)

The chamber drawing follows Kim et al.’s instrument layout and Park et al.’s run sequence. Park et al. report the run-2 injection time. For runs 3–6 we inferred the nominal minute from the chamber records. Run 1 received no added salt. [Kim et al., facility figure 1](https://amt.copernicus.org/articles/18/3781/2025/) · [Park et al., protocol](https://doi.org/10.1029/2026GL123395)

## Measured-cloud comparison

The field comparison uses the published RF14 case mean for 1 March 2020. NASA lists the flight granules for FCDP, horizontal-arm 2D-S, meteorology, aerosol and leg flags. Those files require Earthdata access. The manifest records the granule identifiers and direct download URLs separately from the figure readings. [NASA ACTIVATE collection](https://doi.org/10.5067/SUBORBITAL/ACTIVATE/DATA001) · [Sorooshian et al., 2023](https://doi.org/10.5194/essd-15-3419-2023)

| Extraction in the ZIP | Figure and units | Reading convention |
|---|---|---|
| `rf14-dropsondes.csv` | Fig. 10. Altitude in m, temperature in °C and relative humidity in %. | Two color traces sampled in the lower atmosphere and interpolated to 50 m spacing from 650 through 1850 m. Blank cells indicate missing readings. [Original figure](https://essd.copernicus.org/articles/15/3419/2023/essd-15-3419-2023-f10.png) |
| `rf14-flight-context.csv` | Fig. 11. Pixel column and UTC minutes after midnight, altitude (m), temperature (°C), water vapor (g kg⁻¹), number above 100 nm (cm⁻³), sodium (µg m⁻³), wind range (m s⁻¹) and an image-derived cloud flag. | Each sample represents a visible column of traces. Gaps and overlapping traces limit local readings. The flag screens figure context. Fig. 12b applies the publication’s liquid-water and droplet-number thresholds. [Original figure](https://essd.copernicus.org/articles/15/3419/2023/essd-15-3419-2023-f11.png) |
| `rf14-size-traces.csv` | Fig. 12b. Pixel coordinates, phase, diameter (µm) and number density (cm⁻³ per decade of diameter). | Reads the stitched liquid and ice lines. Integrate each visible line over log₁₀ diameter. The study compares liquid from 3–830 µm and ice from 108–1300 µm. A ±3-pixel vertical shift changes density by about ±13%. This measures image reading precision. Probe sampling and phase uncertainty are additional. [Original figure](https://essd.copernicus.org/articles/15/3419/2023/essd-15-3419-2023-f12.png) |

All three field tables are numeric adaptations of Sorooshian et al.’s published figures. The authors’ articles and Park et al.’s Figshare record identify the source licenses. Credit Park et al. for the Jeju experiment and image sheets, Sorooshian et al. for the RF14 figures, and the Particula Jeju study for selecting, classifying and digitizing the readings. [RF14 article and license](https://essd.copernicus.org/articles/15/3419/2023/) · [Jeju dataset and license](https://api.figshare.com/v2/articles/31744108/versions/1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The ZIP contains four CSVs, provenance and reading notes. Use the SHA-256 checksums in the manifest to verify the extracted CSVs. The study scripts generate chamber averages, fitted spectra, sensitivity scores and figures from the cited inputs.

[Six-run results and conclusions](index.md)
