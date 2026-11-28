# Reproduce the Terra shortwave footprint comparison

The packet contains 32 Terra FM1/MODIS footprints from 13 and 29 November 2001. The CERES field is outgoing hemispheric shortwave flux in W/m². The two MODIS fields are red and near-infrared band-mean radiances in W m⁻² sr⁻¹ µm⁻¹. Positive residuals mean that the model exceeds the observation. Both overpasses were reserved for evaluation. There was no parameter fitting to the observed outgoing radiance or flux. [Study report](../../../docs/studies/validation/spectral-radiation/observations.md)

## Replay locally

Start from the repository root with Particula installed. The compact inputs are the extracted footprints, layer optical depths, and detector response curves. The solver uses the installed version of Particula. Each footprint is evaluated at the observed illumination and view coordinates, in its own column. The first solve integrates the 217-node shortwave quadrature. The second resolves the two in-band responses at 1 nm. Results include the scene, ideal surface, aerosol amount and phase function perturbations.

```bash
study=studies/spectral-radiation/observations
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python "$study/compare.py" /tmp/comparisons.csv --workers 4
MPLBACKEND=Agg python "$study/summarize.py" /tmp/comparisons.csv /tmp/summary.csv /tmp/residuals.png
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python "$study/compare.py" /tmp/refined.csv --workers 4 --streams 12 --primary-only
```

The six-case, 32-footprint offline replay took 16.7 seconds on an AMD Ryzen AI 9 HX 370 using four worker processes, one BLAS thread each. The published [per-footprint model results](comparisons.csv), [pooled and stratified residuals](residual-summary.csv) and [native-pixel extraction](native-pixels.csv) provide an offline replay target. Different BLAS implementations may change the last printed decimal. Stream refinement compares each of the 32 predicted fluxes and band radiances with its own twelve-stream result.

## Recover the measurements

Download the two original CERES SSF Edition 4A HDF4 files with a NASA Earthdata Login. Store them outside the repository. The source identifiers, SHA-256 hashes, UTC observation times, zero-based footprint indices, sample-selection counts and input quantities are in [footprints.json](footprints.json). [screening.npz](screening.npz) preserves the first rejection reason, eligible-unsampled status or selected status for every row in each original file. Read the two arrays for the UTC hour as `reason_YYYYMMDDhh[index_YYYYMMDDhh]` to restore the labels. Both data products were retrieved on 2026-11-28. NASA Langley ASDC distributes CERES SSF. [CERES collection](https://doi.org/10.5067/TERRA/CERES/SSF_Terra-FM1_L2.004A) · [13 November 08 UTC granule](https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CER_SSF_Terra-FM1-MODIS_Edition4A/2001.11/CER_SSF_Terra-FM1-MODIS_Edition4A_400403.2001111308) · [29 November 08 UTC granule](https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CER_SSF_Terra-FM1-MODIS_Edition4A/2001.11/CER_SSF_Terra-FM1-MODIS_Edition4A_400403.2001112908)

```bash
study=studies/spectral-radiation/observations
python "$study/matchups.py" --screening /tmp/screening.npz /tmp/footprints.json /path/to/granules/CER_SSF_Terra-FM1-MODIS_Edition4A_400403.200111{13,29}08
curl -fLsS 'https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/Terra_RSR_in-band.xlsx' -o /tmp/terra-rsr.xlsx
python "$study/responses.py" /tmp/terra-rsr.xlsx /tmp/terra-response.csv
python "$study/native_pixels.py" /tmp/terra-pixel-cache /tmp/native-pixels.csv
```

Raw HDF4 extraction uses `pyhdf`, the response workbook uses `openpyxl`, and the optional public Level-1B retrieval uses `fsspec[http]` and `h5py`. The MCST in-band workbook has SHA-256 `f14498183f9258ed691999313a13fc425edb2eb2d89ca676f79abbb6f48ad637`. We interpolate the forty detector curves of each band to 1 nm, normalize each detector by area, average equally, then normalize the mean to unit peak. The study uses zero response outside the published in-band samples. TerraFusion orbits 10135 and 10368 on path 169 preserve the original calibrated Collection 6 MOD021KM and MOD03 granules. The audit records those filenames per footprint. It keeps pixels within a 10 km radius and at most 5% reported uncertainty in both bands. At granule boundaries we use the granule containing the most qualifying pixel centers. The SSF colocation and this circular extraction have different spatial weights. [MCST detector responses](https://mcst.gsfc.nasa.gov/calibration/parameters) · [TerraFusion open data](https://registry.opendata.aws/terrafusion/) · [MODIS Level-1B documentation](https://mcst.gsfc.nasa.gov/l1b-documents)

## Column and surface controls

The sampled column has boundaries at 120 km, 10 km, 2 km above the reported surface, and that surface. For each water-column lookup we rescale AFGL water number density to 0.5, 1.5, 3 or 5 cm liquid equivalent at sea level and recompute gaseous absorption at every original REPTRAN node. After vertical truncation at the surface we linearly interpolate layer gas optical depths to the SSF precipitable-water column. The SSF surface pressure scales the vertical AFGL pressure profile used for molecular scattering. Gas abundances other than water, line-broadening pressure and temperatures remain those in the gas calculation. These approximations matter where the atmosphere departs from a rescaled standard profile. To regenerate the committed optical-depth subset, use the same local libRadtran 2.0.6 prefix and data specified for the [independent benchmark](../benchmark/README.md).

```bash
study=studies/spectral-radiation/observations
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python "$study/water_profiles.py" /path/to/libRadtran-prefix /tmp/terra-water
PYTHONPATH=. python "$study/scene_profiles.py" /tmp/terra-water /tmp/scene-profiles.npz
python "$study/polarization.py" /path/to/libRadtran-prefix /tmp/polarization.csv
```

Land aerosol optical depth and Ångström exponent come from MOD04 Deep Blue. For water, 550 and 865 nm MOD04 ocean depths supply the spectral exponent. We require at least 50% footprint coverage by the applicable aerosol product. Eight near-glint water cases borrow the closest qualifying same-orbit retrieval, subject to 250 km and 120 s limits, with donor and distance recorded. The aerosol extinction varies as wavelength to the negative Ångström exponent. The assumed Henyey–Greenstein asymmetry is 0.70 with sensitivity cases at 0.60 and 0.80. Ocean aerosol single-scattering albedo is 0.99. Land uses the 659 nm Deep Blue value where recorded and 0.95 otherwise, extended as a spectrally neutral approximation. Ideal surface controls change only surface reflection. The aerosol-amount controls multiply column AOD by 0.75 or 1.25 with all other properties held fixed.

At land sites the 412, 470 and 659 nm Deep Blue directional reflectances scale the ideal ground spectrum. Between these measurements, the ratio to the reference spectrum is linearly interpolated, and its endpoint values are retained outside their interval. The HLS Table 1 angular kernel ratios vary by wavelength. We normalize the model BRDF to reproduce the prescribed directional surface reflectance at the actual viewing geometry. These ratios are global annual averages. The ocean surface uses the meteorological vector wind magnitude and an isotropic distribution of facet slopes. The SSF convention puts the Sun at 180° relative azimuth. Subtracting 180° expresses the imager angle in the solver's Sun-relative convention. [SSF collection guide](https://ceres.larc.nasa.gov/documents/collect_guide/pdf/SSF_CG.pdf) · [NASA HLS kernel coefficients](https://hls.gsfc.nasa.gov/algorithms/)

The reference packet's [solar acquisition](../reference/README.md) supplies the full 1 nm TSIS-1 table. The unmodeled incident energy cited in the study is the sum of its bin means below 280 nm or above 4000 nm, multiplied by the observed solar zenith cosine and inverse squared Earth–Sun distance. This bounds the omitted reflected flux from the source table. The cloud sensitivity bound multiplies the fractional reported cloud area by the in-range horizontal downwelling flux at the top. The radiance units convert from per metre to per micrometre by multiplying by 10⁻⁶.

The independent MYSTIC control uses two million photons per run, separate fixed seeds for scalar and vector intensity, a sea-level AFGL Rayleigh atmosphere, TSIS-1 illumination at 1 au and transparent gases. The selected land case uses the same 659 nm Lambertian surface reflectance in both bands, and the ocean uses uvspec's wind-dependent Cox–Munk boundary. It tests the effect of polarization alone at two observed geometries. The [paired results](polarization.csv) preserve Stokes I and Monte Carlo standard errors. For the difference of independent estimates the combined standard error is their quadrature sum. The observed aerosols' Mueller matrices were unavailable. [MYSTIC polarized radiation](https://doi.org/10.5194/acp-10-383-2010)
