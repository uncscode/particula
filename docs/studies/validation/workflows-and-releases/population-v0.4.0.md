# Do particle inventories preserve the published study results in 0.4.0?

A parcel can carry the same number of particles as it expands, while an optical probe reports fewer particles per unit air volume. Particula 0.4.0 stores represented population explicitly. This record follows that population into parcel and cell inventories, optical extinction, and three published size spectra. For a study with an existing output file, retain its generating revision, sample volume and bin definition when making the comparison. [Migration guide](../../../guide/migration/v0.4.0/index.md)

## Study inputs and scope

The old package reference is `v0.3.6` (`3435f653ae`). The migration entered upstream at `0fa70c4964`. These checks used the 0.4.0 source on base `b39f1541f3`, Python 3.12.12, NumPy 2.3.4 and Warp 1.11.1, with CPU and native RTX 5060 CUDA execution where indicated. Absolute errors below use the physical unit of the observable. Relative errors divide by the absolute value of each nonzero reference before taking the maximum.

| Comparison | Baseline and volume owner | Quantity checked |
|---|---|---|
| Π Chamber | Saved 0.3.6 ensembles at `ecc1d88a5c`. Interior occupies 85% of modeled air. | Sum interior cohort weights per cm³ into 18 wet-diameter bins. |
| Pallas probes | Field-study source `3435f653ae`, archived tables at `3562abffaf`. Each probe supplies an instrument-normalized density. | Integrate each plotted probe spectrum over five common intervals. |
| Jeju chamber | Fit engine `2db6a77323`, public plot data at `34b9bf25e8`. The data are concentrations at chamber conditions. | Integrate per-decade Welas spectra back to the measured and simulated band concentrations. |
| Parcel export | Moist parcel solver at `a3ff1e82e7`; analytic dry inventory and independently evolved liquid mixing ratio. | Compare native counts and inventories at the physical volume of each parcel sample. |
| Cavity | OpenFOAM fixture at `50d4230deb`, saved CUDA replay at `bac5acf6bb`. Fluxes connect physical cells. | Compare cell inventories, the final tracer field, centroid and local uptake. |
| Sphere optics | Native state tests at `0fa70c4964`, with the independent measurement-array reference corpus at `240a07398a`. | Compare native extinction and absorption with cross sections weighted by integrated number per m³. |

## Three representations reproduce their two-hour coagulation trajectories

The CPU tutorial starts with a lognormal distribution at 10¹⁰ particles m⁻³. Each row below compares 13 unrounded samples, 600 s apart, against the same representation running on v0.3.6. The old radius-PDF calculation integrated its slot density over radius. The new calculation reads `number_density`. Both runs used seed 42 for the initial resolved radii and a separate NumPy generator seeded with 42 for Brownian pair selection. [Coagulation time evolution](../../tutorials/coagulation-populations/coagulation_5_time_evolution.py)

| Population | Final number density (m⁻³) | Maximum absolute difference (m⁻³) | Maximum relative difference |
|---|---:|---:|---:|
| Discrete bins | 9,511,891,884.48 | 0 | 0 |
| Radius PDF | 9,522,210,158.11 | 0 | 0 |
| Resolved | 9,511,400,000.00 | 0 | 0 |

The originally published resolved plot seeded the radii only and printed a final 9.52 × 10⁹ m⁻³. The paired comparison fixes the collision draw so a second run can reproduce the difference attributed to migration. To repeat it, run the tutorial from `0fa70c4964` against package tag `v0.3.6`, setting `coagulation_resolved.coagulation_strategy.random_generator = np.random.default_rng(42)` immediately after constructing the resolved runnable. Compare `number_pmf`, `number_pdf` and `number_resolved` with the corresponding arrays in the current tutorial's downloadable results. The irregular-grid worked example separately checks the PDF quadrature and species inventories at 0.25 and 4 m³. [Population and volume example](../../tutorials/building-an-aerosol/particles/represented_population.py)

## A flush and an expansion give the same density with different inventories

The fixed-volume CPU example steps 4 s at 0.25 s⁻¹, in two equal substeps. In its 2 m³ sample, the two particle density lanes and the partitioning gas matched their initial concentrations times exp(−1) exactly. For the two remaining gas lanes, the maximum absolute difference was 2.22 × 10⁻¹⁶ kg m⁻³ and the maximum relative difference was 1.21 × 10⁻¹⁶. [CPU dilution example](../../examples/particle-processes/sources-and-dilution/cpu_dilution.py)

| Case from 100 particles and 1 µg gas in 0.25 m³ | Final volume (m³) | Particle count | Gas mass (µg) | Number density (m⁻³) | Gas density (µg m⁻³) |
|---|---:|---:|---:|---:|---:|
| Fixed-volume half flush | 0.25 | 50 | 0.5 | 200 | 2 |
| Expansion to twice the original volume | 0.50 | 100 | 1 | 200 | 2 |

The expansion uses the resolved Warp volume step, which updates particle volume and gas concentration together. The comparison exercises both the Warp CPU device and native CUDA. [Reproduce the two inventory budgets](../../../guide/processes/dilution.md#compare-a-flush-with-physical-expansion)

## Four original analysis archives retain their recorded quantities

Loading the v0.3.6 resolved fixture yields 100 particles in 0.5 m³. The discrete fixture yields 12 m⁻³ and the radius PDF integrates to 2.3 × 10⁻⁸ m⁻³. Their raw arrays retain counts for discrete and resolved slots and m⁻¹ for the PDF. The derived-only archive yields its recorded 12 m⁻³. All four retain source schema version 1, and the loader retains the original field descriptions for interpreting historical files. The guide includes a runnable comparison and the exact-restart constraints. [Read a 0.3.6 particle history](../../../guide/getting-started/recording-restart.md#read-a-036-particle-history)

## Π Chamber residents below the optical limit

At time zero, all 1,844 freshwater dry-start residents per cm³ and all 2,365 saltwater dry-start residents per cm³ occupy the simulated 130 nm seed bin. A counter starting at 0.6 µm would miss them. The wet restarts place the same total number inside the optical range. No initial cohort falls below the model's 0.12 µm lower edge or above its 120 µm upper edge, and none exceeds the counter's 40 µm limit. These are reconstructed initial conditions. The chamber observations used to set them describe steady state. [Π Chamber experiment and restart assumptions](../../literature/pi-chamber/haze-and-cloud/index.md)

We regenerated the initial residents from each archived seed, excluded future births and floor residents, and divided each resident weight by the 0.85 interior fraction. We summed each diameter interval in each of eight groups and averaged the groups. A particle on an internal edge enters the bin on its right. The last bin includes the upper endpoint.

| Time step and seed | Largest difference across four starts × 18 bins (cm⁻³) | Largest relative difference | Freshwater / saltwater wet-start cloud number, 1.8–40 µm (cm⁻³) |
|---|---:|---:|---:|
| 0.0625 s, 19273 | 2.51 × 10⁻¹² | 1.36 × 10⁻¹⁵ | 770.627 / 1,038.246 |
| 0.03125 s, 89117 | 1.37 × 10⁻¹¹ | 5.77 × 10⁻¹⁵ | 770.757 / 1,037.831 |

A full CUDA replay with the two archived seeds reproduced all 2,304 spectral ordinates across 16 output times exactly (maximum difference 0 cm⁻³, relative 0). The 128 cloud-number Monte Carlo half-widths and 64 dry/wet separation values also matched exactly. Both dry/wet pairs still first fell below 1% separation at the 32 s sample. This tests reproducibility for the original draws and time grid.

The original cohort test allows a relative difference of 10⁻¹² and an absolute difference of 10⁻⁹ cm⁻³ per bin. These checks preserve both the counter range and the dry-seed population outside it. [Time-zero checks](../../../../examples_tests/pi_chamber_spectra_test.py) · [Restart quantiles](https://github.com/uncscode/particula/blob/main/docs/data/pi-chamber/restart.json)

## The Pallas probes retain their reversal between dates

The FSSP counted fewer 6–13 µm droplets than the ICEMET imager on 15 September 2022, and more the next day. For each curve, the CPU plotting check multiplied its height in cm⁻³ per decade by each interval's Δlog₁₀D and summed. The six diameter edges were 6, 7.5, 9, 10.5, 12 and 13 µm. The reference uses the two observed columns of the nominal, refined, frozen-calibration CDA rows in the attribution table. The observed columns retain each probe's sampling and reporting response. [Pallas study and corrected source data](../../literature/pallas-cloud-probes/index.md) · [Attribution table](https://github.com/uncscode/particula/blob/main/docs/data/pallas-cloud-probes/attribution.csv)

| September date | Matched minutes | FSSP (cm⁻³) | ICEMET (cm⁻³) | Largest absolute / relative difference from the attribution table |
|---|---:|---:|---:|---:|
| 15 | 1,439 | 123.773823489 | 173.472943026 | 0 cm⁻³ / 0 |
| 16 | 1,320 | 91.821414384 | 75.056913611 | 1.42 × 10⁻¹⁴ cm⁻³ / 1.55 × 10⁻¹⁶ |

This deterministic comparison checks the displayed field concentrations. The optical sizing simulations and their uncertainty are documented separately. [Probe model and uncertainty](../../literature/pallas-cloud-probes/specification.md)

## The Jeju fits still undercount the 5–10 µm particles

Both full-record fits underpredict the run-5 concentration in the Welas band. Over the same 41 valid minutes, the shared fit accounts for about 36% of the observed concentration and the individual fit about 11%. Integrating the distributed plot table gives the same result to its eight-significant-digit precision. [Published size comparison](../../literature/jeju-chamber/spectra.md)

| Run-5 record mean, 5–10 µm | Original ten-bin sum (cm⁻³) | Integral of public spectrum (cm⁻³) |
|---|---:|---:|
| Measured | 388.293044721 | 388.293044544 |
| Shared fit | 141.274413902 | 141.274410483 |
| Individual fit | 40.790628930 | 40.790628015 |

The CPU check matches every published run, minute and bin to the original fine-grid output. It multiplies the per-decade values by the logarithmic widths of the 70 optical channels. As in the original study, a band includes a whole channel when its reported center lies at or above the lower limit and below the upper limit. Across six runs, 245 valid minutes, three bands (2–5, 5–10 and 10–20 µm) and three series, the largest of 2,205 differences was 9.28 × 10⁻⁵ cm⁻³. The largest relative difference was 5.81 × 10⁻⁷, below the study renderer's 10⁻⁶ integration tolerance. Rounding both edges and heights for distribution accounts for this precision limit.

Welas supplies optical number concentrations. The model supplies weighted concentrations on the same optical grid. CPI detections are individual image events and enter a separate comparison. All eight full-precision input files matched the original spectrogram verification hashes. The comparison used archived 1 s fits from the RTX 5060 and recomputed their reductions without refitting microphysics. Material attribution of the missing particles remains open. To repeat the calculation, regenerate the results directory following the source and checksum instructions, then run the checker from the checkout root. [Measurement provenance and plot table](../../literature/jeju-chamber/sources.md) · [Replay instructions](../../../../studies/jeju-chamber/README.md)

```sh
python studies/jeju-chamber/check_band_recovery.py "$local_root/results"
```

## A parcel's sample volume changes along its trajectory

The moist parcel carries 100 million particles in three classes as it rises 100 m. It contains 1 kg dry air, with dry radii of 45, 70 and 110 nm and counts of 25, 50 and 25 million. We used the existing ascent case with 1 m/s updraft and 2,000 steps of 0.05 s. The CPU reference and CUDA trajectory each supplied their own volumes to `aerosol_at`. On CPU the volume increased from 0.93650019 to 0.94507236 m³, reducing number density from 1.06780544 × 10⁸ to 1.05812004 × 10⁸ m⁻³. The final CUDA volume was 0.94507238 m³. [Parcel export and solver tests](../../../../particula/dynamics/tests/parcel_test.py)

At 0 and 100 s, the exported slot densities equaled initial class counts divided by that sample's volume, with zero absolute and relative error on both backends. Multiplying total density by sample volume recovered all 100 million particles. Independently summing the dry-sphere masses at 1,770 kg/m³ gave 3.90749331 × 10⁻¹⁰ kg, also with zero error. The exported liquid inventory was 3.83133356 × 10⁻⁹ kg initially and 1.06404971 × 10⁻⁴ kg at 100 s on CPU. Relative to the solver's liquid mixing ratio times dry-air mass, the largest export difference was 2.72 × 10⁻²⁰ kg or 2.55 × 10⁻¹⁶ across both devices. Agreement here qualifies export at the sample's own state. The parcel tests separately compare the trajectories and their numerical integration tolerances.

## The cavity replay preserves the tracer plume and cell budget

The published cavity has 256 cells of volume 3.90625 × 10⁻⁷ m³ each. At time zero, 16 cells contain tracer at 10⁻⁶ kg/m³, for a total of 6.25 × 10⁻¹² kg. Replaying all 40 intervals on CPU and CUDA used those physical cell volumes to turn concentration into gas mass. The maximum field L1 difference from OpenFOAM was 0.0390758 of initial mass, and the largest centroid displacement was 0.351347 mm. Both fit the frozen host-refinement bounds of 0.185870 and 0.895239 mm. [Worked cavity replay](../prescribed-flow/cavity_replay.py) · [OpenFOAM fixture and generation instructions](../../../../studies/prescribed-flow/openfoam-cavity/README.md)

| Observable versus the saved 0.3.6 CUDA replay | Largest absolute difference, CPU / CUDA | Largest relative difference, CPU / CUDA |
|---|---:|---:|
| Final 16 × 16 tracer field | 3.71 × 10⁻²¹ / 0 kg/m³ | 7.40 × 10⁻¹⁵ / 0 |
| Field L1, 41 records | 1.68 × 10⁻¹⁵ / 0 (fraction of initial mass) | 4.84 × 10⁻¹⁴ / 0 |
| Centroid distance, 41 records | 3.01 × 10⁻¹⁷ / 0 m | 1.11 × 10⁻¹² / 0 |
| Gas and condensed fractions with local uptake | 2.56 × 10⁻¹⁵ / 8.89 × 10⁻¹⁶ | 4.33 × 10⁻¹⁵ / 1.05 × 10⁻¹⁵ |

The companion two-cell check isolates the mapping from an aerosol sample into a CFD cell. Densities of 5 and 7 m⁻³ in physical cells of 4 and 3 m³ give 20 and 21 particles. With 2 and 3 kg per particle the dry inventories are 40 and 63 kg. The check recovered those integers exactly for sample-volume pairs (1, 1), (4, 3) and (0.25, 4) m³. Emptying a bin set its cell number and mass inventory to zero while preserving its stored per-particle mass on unpack, so its radius coordinate survived. The cavity forcing is deterministic. Only the separate numbering and face-orientation test uses a random permutation (seed 72). [Cell packing, unpacking and empty-bin tests](../../../../particula/coupling/tests/grid_flow_test.py)

## Native optical coefficients close against a cross-section sum

For spheres of radii 60 and 120 nm at 4 and 8 particles/m³, the 500 nm extinction coefficient is 5.63106521 × 10⁻¹³ m⁻¹. The corresponding absorption coefficient is 9.67440123 × 10⁻¹⁴ m⁻¹. We evaluated both coefficients at 500 and 650 nm with index 1.65 + 0.04i. The independent reduction sums πr² times the single-sphere efficiency times the measured number density. On CPU the largest discrete or resolved difference for sample volumes of 0.25 and 4 m³ was 1.01 × 10⁻²⁸ m⁻¹, with maximum relative difference 1.91 × 10⁻¹⁶. [Optical population weighting](../../../guide/optics/index.md) · [Native-state tests](../../../../particula/optics/tests/state_test.py)

For a continuous PDF we used radii of 70, 83, 145 and 317 nm and density ordinates of (5, 9, 2, 1) × 10¹⁵ m⁻⁴. Trapezoidal integration of the complete product over radius gives 1.44870888 × 10⁻⁴ m⁻¹ extinction at 500 nm. Both coefficient checks at both wavelengths and volumes agreed within 2.72 × 10⁻²⁰ m⁻¹, with maximum relative difference 2.09 × 10⁻¹⁶. The separate measurement-array API accepts declared counts or densities and explicit measure information. Its saved PyMieScatt tests and the GPU reduction tests exercise that path independently. [Population tests and external reference](../../../../particula/optics/tests/population_test.py)

## Which consumers use the particle sample volume?

The file-level audit lists 275 source, test and reference paths across the four migration directories. Each row records the last file revision at the merged baseline, the old and current field expressions, the quantity that owns the volume conversion, and its evidence. The audit checked particle constructors (including positional input), derived densities, Warp buffers, quoted recorder and restart fields, study-local `counts` arrays, measurement transforms, declared example inputs and their local importers. A byte-for-byte match identifies a preserved output file. Independent physical comparisons and fresh replays are named in their own rows. [File-by-file consumer audit (CSV)](https://github.com/uncscode/particula/blob/main/docs/data/population-v040/consumer-audit.csv)

The 49 distinct paths in the CSV’s `saved_json_compared` column identify the numeric JSON files compared against `0fa70c4964`. Of those, 45 had identical file contents. Comparing every stored value in the other four found differences only in run times and, for dense optical recording, native analysis-file sizes. Their physical outputs were identical. Examples that publish console or figure output alone have separate execution manifests. The archive comparison cannot establish a new trajectory for a source-only row.

Nine examples declare historical engines. Four downloads also supply a launcher that loads their hashed spectrum reporting helper and plotting code beside the pinned engine. Keeping the original raw attribute in helpers shared by these downloads preserves the resolved counts those engines exposed. Elsewhere an identical spelling can mean a gas concentration in kg/m³, an observed optical density, an optimizer's candidate population, or a count of instrument records contributing to an average. The receiver and the source data determine which interpretation applies. [Download builder](../../../../scripts/study_downloads.py) · [Historical study launcher](../../../../scripts/historical_study.py)

## Reproduce the study checks

Run from the checkout root with the project dependencies. The command below passed 112 tests with warnings treated as errors and no skips on the CPU/RTX 5060 host. It includes the actual Warp CUDA parcel, cavity and optical reduction paths. The spectral figure integrations and native optical cross-section sums use CPU arrays.

```sh
python -m pytest -q --no-cov -Werror -rs \
  examples_tests/pi_chamber_spectra_test.py examples_tests/pallas_number_spectrum_test.py \
  particula/dynamics/tests/parcel_test.py particula/coupling/tests/cavity_replay_test.py \
  particula/coupling/tests/grid_flow_test.py particula/optics/tests/state_test.py \
  particula/optics/tests/population_test.py particula/optics/tests/gpu_population_test.py
```

To repeat the complete Π Chamber CUDA replay and compare the original ensemble diagnostics, call the maintained example entry point. For the cavity, the same comparison uses the saved host and replay arrays from its example.

```python
import json
import runpy
from pathlib import Path

import numpy as np

def saved(study):
    return json.loads(Path(f"website/results/{study}/data.json").read_text())

pi = runpy.run_path("docs/studies/literature/pi-chamber/haze-and-cloud/pi_chamber_spectra.py")
output = pi["simulate"]()
reference = saved("pi-chamber-spectra")
for run in ("main", "refined"):
    for field in ("bin_cm3", "cloud_mc95", "relative_spectrum_difference"):
        np.testing.assert_array_equal(output[run][field], reference[run][field])

cavity = runpy.run_path("docs/studies/validation/prescribed-flow/cavity_replay.py")
reference = saved("cavity-replay")
for device in ("cpu", "cuda:0"):
    output = cavity["simulate"](device)
    for field in ("replay_kg_m3", "field_l1", "centroid_error_m", "uptake_mass_fractions"):
        actual, baseline = np.asarray(output[field]), np.asarray(reference[field])
        difference = np.abs(actual - baseline)
        nonzero = baseline != 0
        print(device, field, difference.max(), (difference[nonzero] / abs(baseline[nonzero])).max())
```

The following SHA-256 values identify the distributed inputs used for the tabulated comparisons. The Jeju source manifest records each original Figshare file's versioned URL and MD5. The Π Chamber and cavity result manifests provide their generating revisions and input hashes.

| File | SHA-256 |
|---|---|
| Π Chamber restart quantiles | `7f3a9a092788a2b35d439eab3be67a69379f3adfa42ec7f969dbcbeb0465258d` |
| Pallas distributions | `28401d99d31a7363c027e14e193fd5a28603858345580957c88913952070932a` |
| Pallas attribution | `2944e5e691b3470a60ce89ef05e5a527e74ca7dbcfc40154792da86c6d512944` |
| Jeju public figure-data ZIP | `916437bcac0aae8350dd07682a0c592ae365f0d52ed11616a5e7a15332437ef7` |
| Cavity native-order export | `a751851b8bbb3583d8a00abe526e0b2c9f12e693d1ab0ab4fe4efea87c77492d` |
