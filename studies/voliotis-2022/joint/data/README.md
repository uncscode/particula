# Manchester experiment 2

The second 309 ppb α-pinene experiment ran on 29 March 2019, the day after experiment 1. Its SOA record expands coverage to all five α-pinene-only runs with archived AMS organics. Four have both the aerosol and precursor time series. Table S1 marks the run 2 precursor unavailable following instrument problems. Run 3 has ancillary chamber data only. The paper's remaining experiments involve o-cresol, isoprene or mixtures and require additional precursor chemistry. [Paper, Table 1](https://acp.copernicus.org/articles/22/14147/2022/) · [Supplement, Table S1](https://acp.copernicus.org/articles/22/14147/2022/acp-22-14147-2022-supplement.pdf)

The [AMS and metadata record](https://doi.org/10.25326/GXV6-3646) and [chamber instrument record](https://doi.org/10.25326/3549-H353) credit the Manchester team. The ACTRIS/EUROCHAMP portal displays the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). The EDF v1.0 input headers label the measurements preliminary. The transformations here retain native values and elapsed sampling times. The AMS record contains 101 suspended-organic concentrations (µg m⁻³), 216 s apart from lights-on to 6 h. The maximum of 282.93 µg m⁻³ occurs at 2.52 h. The paper gives 283.1 µg m⁻³, 67.6 µg m⁻³ seed and 43 ppb nominal NOₓ.

At lights-on the WCPC registered 10,400, compared with 10,500 in run 1. Both use the same counter type. The source leaves the unit blank and the calculation interprets the channel as cm⁻³. Temperature was 24.81 °C and RH 58.0% at 12 s. At 0 s NOₓ was 42.6 ppb, NO 12.7 ppb and O₃ −1.8 ppb. The negative ozone reading is retained in the table. As in the previous study packet, the chamber-context CSV is sampled near 6 min intervals and preserves the original acquisition times. The DMPS EDF provides total mass at unit density; this series includes seed and is not used as the organic observable.

## Borrowing the unmeasured precursor history

Run 1 and run 2 have similar measured five-hour ozone exposures. Integrating positive ozone values from the context tables over 0–18,000 s gives about 370,000 and 374,000 ppb s, respectively. This supports a run-1-like ozone exposure as a trial assumption. It does not establish matching OH, vapor wall loss or α-pinene disappearance. Run 2 has no measured VOC forcing or exposure confidence interval.

A rate sensitivity can multiply the cumulative run 1 apparent hazard, −ln(fraction remaining), by 0.80, 1.00 or 1.25. The lower-dose replicate pair provides the scale for this test. At integer hours 1–5, the run 5/run 4 cumulative hazard ratios are 1.140, 1.250, 1.154, 1.180 and 1.097 from their archived VOC series. These factors describe a sensitivity range by analogy. They do not bound the actual run 2 exposure. [Measured run 1, 4 and 5 series](../../data/observations.csv)

## Rebuild

From this directory, run `python prepare_run2.py --raw-dir /tmp/voliotis-run2`. This uses the shared [EDF preparation utility](../../data/prepare.py), verifies source sizes and the manifest SHA-256 hashes, and writes [observations.csv](observations.csv), [context.csv](context.csv) and [source_manifest.csv](source_manifest.csv). The manifest includes direct download URLs for all five inspected EDFs.
