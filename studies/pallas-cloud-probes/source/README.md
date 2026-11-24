# Reproduce the Pallas cloud-probe comparison

These programs take the September 2022 probe files through minute matching, optical simulation and uncertainty propagation to the numerical tables used in the manuscript. The primary comparison uses the same incoming droplet spectrum for both simulated probes. Their field wavelengths are 633 nm for the forward scattering spectrometer probe (FSSP) and 660 nm for the ICEMET holographic imager.

## Replay

Use the Particula checkout and an environment with Python, NumPy, SciPy, pandas, Matplotlib and NVIDIA Warp. The study used Python 3.12.12, Particula 0.3.6 at commit `3435f653ae2606c4eeef23bdf55b617d2ac58586`, NumPy 2.3.4, SciPy 1.16.2, pandas 2.3.3, Matplotlib 3.10.7, Warp 1.11.1 and an NVIDIA GeForce RTX 5060 with 8 GiB. The field response grid and optical pixel calculations require an accessible CUDA device. CPU routines calculate Mie amplitudes, uncertainty intervals and reference projections.

From the repository root, activate the environment and run

```bash
python studies/pallas-cloud-probes/source/replay.py --work /tmp/pallas-replay
```

The driver downloads the ten original September data and metadata files listed with their archive checksums in [replay.py](replay.py), verifies every file and writes intermediates and results under the selected work directory. For an offline replay, point `--input-dir` to a directory containing those files. An existing work directory reuses verified inputs. Use a fresh directory for an independent reproduction. The `--stage` option accepts `fetch`, `field`, `optics` and `spectra`. The last stage reads the results of `optics`. Each numerical stage overwrites its own outputs in the selected directory. [Doulgeris et al., corrected Zenodo release v3, CC BY 4.0](https://doi.org/10.5281/zenodo.19220993) · [Campaign paper and instrumentation](https://doi.org/10.5194/essd-17-6497-2025)

## From a published figure to its inputs

| Manuscript comparison | Numerical source | Table under the replay directory |
| --- | --- | --- |
| Figure 1. Measured and simulated spectra on 15 and 16 September | [Minute selection and channel integration](study.py), [CDA input spectrum](optics/score.py), [native response probabilities](spectra/transfer.py), [field uncertainty](spectra/budget.py) | `spectra/distributions.csv` gives per-bin number in cm⁻³ and symmetric percent differences. `spectra/uncertainty-budget.csv` gives the integrated band and sensitivity cases. |
| Figure 2. Assigned diameter for a known water droplet at 633 nm | [Optical fields and pulse calibration](optics/instruments.py), [pixel integration](optics/pixels.py), [simulated bead fit](optics/simulate.py), [individual sizing outcomes](spectra/method_only.py) | `spectra/method-sizing.csv` gives means, standard deviations and 16th/84th percentiles of registered diameters in µm for each true size. |
| Figure 3. Number in the 6–13 µm band under each calibration | [Common-input weighting](optics/score.py), [paired wavelength, processing and gain controls](spectra/method_only.py) | `spectra/method-budget.csv` gives band counts, scatter and root-mean-square error. `optics/attribution.csv` gives wavelength and method contrasts for each common input. |

The supplementary detection diagnostic plots the fraction of simulated encounters that trigger, register within the instrument range, and contribute to the 6–13 µm band. The replay writes both `optics/detection-efficiency.png` and its compact numerical table `optics/detection-efficiency.csv`. [Detection curve source](optics/plot.py)

The [manuscript check](check_manuscript.py) reconstructs both dates' band counts from the size-resolved table. It checks the reported uncertainty intervals, sizing errors, and the change in count ordering after removing the approximate imager correction.

## Recover the measured concentration

The reference joins the FSSP and ICEMET files by their corrected UTC minute and discards all occurrences of duplicated keys within each file. Positive counts from both probes, FSSP station temperature above 0 °C, visibility between 1 and 999 m and ICEMET flag zero leave 4,283 minutes from seven dates. The optical comparison uses matching positive CDA concentrations as the incoming population estimate. Of these matched observations, 1,439 minutes on 15 September and 1,320 on 16 September enter the manuscript comparison. The vertical inlet of the cloud droplet analyzer (CDA) can itself lose droplets.

On 14 September, weighting each FSSP spectral channel by 0.05 reproduced archived number. ICEMET and CDA use their channel-specific log₁₀ widths. FSSP channel labels are upper size limits in the metadata and centers in the NetCDF. The analysis carries both interpretations, allocates partial channels uniformly in log diameter and checks the FSSP number, effective diameter and liquid water fields. Agreement in total number leaves uncertainty in the allocation among sizes. Counts use cm⁻³, and spectra use wet droplet diameter in µm. Particula's discrete water population checks the conversion to particle mass and SI concentration. [Reference and reconstruction](study.py) · [Calibration challenge](calibration.py) · [Independent whole-channel check](verify.py)

The `field` stage also tests timing against effective diameter, a calibration sweep, transfer between dates, third-probe coverage and stratification by the CDA count and diameter. The daily 6–45 µm comparison is in `dates.csv`. The 6–13 µm manuscript results use the three-probe overlap and the bands documented above. [Clock and date tests](challenge.py) · [CUDA response sweep](response.py) · [Independent CDA comparison](independent.py) · [Matching across cloud conditions](matched.py)

## Optical response, numerical refinement and uncertainty

Each optical run supplies water droplets from 1 to 40 µm at both wavelengths. FSSP integrates the Mie amplitudes over a 4–12° collection annulus, varies pulse gain and maps intensity to diameter using a smoothed monotone calibration. ICEMET reconstructs an idealized focused field through a finite pupil, integrates contrast over pixels at different positions and depths, and segments the shadow. A simulated glass-bead calibration supplies an approximate small-drop correction anchored at 13.07 µm. The deployed gain, image thresholds, bead correction coefficients and absolute detection efficiencies remain unknown. The primary simulations condition on an isolated droplet entering the effective sampling volume. [Instrument construction and sizing](https://doi.org/10.1007/s10043-020-00583-y) · [ICEMET calibration and depth dependence](https://doi.org/10.5194/amt-15-2993-2022) · [Earlier FSSP configuration at Pallas](https://doi.org/10.5194/amt-13-5129-2020)

The four optical stages use 32, 256, 512 and 1,024 Sobol encounters per diameter. The main 512-draw estimate uses 20 depths, a 0.0625 µm diameter grid and 3 × 3 quadrature per pixel. The nominal convergence case halves the diameter increment, changes the scramble, and uses 28 depths and 4 × 4 quadrature. The software records the device, counts and timings for each stage in `optics/*-run.json`. Wavelength swaps preserve calibration in the main contrast. Separate calculations refit it, vary threshold and pulse gain, remove the imager correction, use a blurred shadow model, or include water dispersion. Full-sphere scattering and 448 independently calculated CPU images check the optical integration and the CUDA pixel kernel. [Optical controls](optics/simulate.py) · [Numerical checks](optics/verify.py)

The field uncertainty calculation uses 16,384 draws per date and setting. It combines equivalent Poisson counts with coherent offsets in each probe's number and diameter scale before rebinning. Working uniform half-widths are ±16% in number and ±20% in size for FSSP and ±16% in number and ±1 µm in size for ICEMET. The latter are assumptions. Doubling or halving those widths, sharing calibration shifts, changing the FSSP edge convention, varying small-drop losses and repeating the random seed test how the result depends on them. Each integrated interval sums counts within a draw before forming the percentage. The signed percentage is 200 × (FSSP − ICEMET)/(FSSP + ICEMET). [Uncertainty calculation](spectra/budget.py)
