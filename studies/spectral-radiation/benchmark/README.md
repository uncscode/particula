# Reproduce the spectral transport benchmarks

Run from the repository root using the Particula development interpreter (Python 3.12+, NumPy, SciPy). The native calculation replays independently of the optional external reference program. All area fluxes are horizontal and positive in their named direction. The three-point synthetic slab uses 450, 550 and 850 nm, while the Earth benchmark integrates the full AFGL/REPTRAN interval.

## Replay the native results

```sh
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
bench=studies/spectral-radiation/benchmark
python "$bench/disort.py"
python "$bench/bai.py"
python "$bench/earth.py"
```

These commands read the compact DISORT tables or Bai fixture and regenerate the native comparisons. The [Bai study](../../bai-limonene/README.md) provides the original data retrieval and cloud simulation sources. The visible regression fixture contains height-integrated optical depths and phase moments of two clean, 0.25 m/s reference parcels and the corresponding archived 1 m reflectances. To rebuild from the study packet's sunlight/optics directory (containing `cloud_extinction.npz`, `cloud_phase.npz`, and `cloud_reflectance.csv`), run `python "$bench/bai.py" --source /path/to/sunlight/optics`.

## Generate the external reference

Install libRadtran 2.0.6 in a separate prefix. We used conda-forge's Linux rubin-libradtran build. The numerical package has no dependency on this executable. Download the published REPTRAN data archive to add the eight fine solar lookup files to this reference prefix.

```sh
prefix="$PWD/.artifacts/libRadtran-reference"
micromamba create -y -p "$prefix" -c conda-forge rubin-libradtran=2.0.6
curl --fail --location --output "$prefix/reptran_2024_all.tar.gz" \
  'https://www.libradtran.org/lib/exe/fetch.php?media=download%3Areptran_2024_all.tar.gz'
(cd "$prefix" && echo '55893c80bcc999651bac3bf014ee64aaf602653ba640eb5bebe787a5d8eacce7  reptran_2024_all.tar.gz' | sha256sum -c -)
tar -xzf "$prefix/reptran_2024_all.tar.gz" -C "$prefix/share/libRadtran" \
  --wildcards 'data/correlated_k/reptran/reptran_solar_fine.lookup.*.cdf'
python "$bench/disort.py" "$prefix"
python "$bench/low_sun.py" "$prefix"
python "$bench/directional.py" "$prefix"
python "$bench/earth.py" "$prefix"
python "$bench/native_fixture.py" "$prefix"
```

The native fixture command rebuilds the pinned monochromatic fluxes in `particula/integration_tests/data/shortwave_disort.csv` using 32-stream DISORT with the particle-derived layer optics and both ideal surfaces.

When comparing radiances, the archived inputs convert the physical angle between Sun and sensor to libRadtran's azimuth conventions. A Sun–sensor separation of 60° maps to `phi - phi0 = 120°`. Both direction cosines and the distinction between spectral irradiance and spectral radiance enter the fixture. The land BRDF uses identical MODIS kernels. In the ocean comparison the native water-leaving term is zero, and the external code uses its default pigment concentration (0.01 mg/m³) and salinity (34.3 parts per thousand). [libRadtran manuals](https://www.libradtran.org/doku.php?id=documentation)

## Regenerate the reference atmosphere

```sh
python "$bench/generate_reference.py" "$prefix" --output /tmp/derived-shortwave-gas
```

The external solver's molecular setup supplies total gas absorption per AFGLUS layer. The generator reads positive-weight nodes, weights, and band boundaries from the native 15 cm⁻¹ REPTRAN netCDF file. The full table uses 4,082 nodes. Within 22 adjacent groups of bands the reduced quadrature preserves the selected moments of solar energy, spectral slope, and gas transmission at specified altitude and air mass by positive linear programming. The sparse solution retains 217 native nodes for the recorded SciPy run. SVD or linear programming implementations can select a different feasible sparse support. Compare band budgets and the external flux benchmark after regeneration.

A full and reduced grid store the same atmosphere. For each node the archive gives spectral weight and band bounds in metres, layer absorption depth (dimensionless), and interface height, pressure, temperature, water and ozone number density in SI units. Height descends from 120 km to sea level. All band integrals use the archived weights. The input spectrum is separately normalized to its TSIS energy within each band. The compact native reference is limited to complete REPTRAN bands inside 280–4000 nm.

- [libRadtran download and usage terms](https://www.libradtran.org/doku.php?id=download). The external program and conda-forge package are distributed under GPL-2.0-or-later. The repository includes numerical atmosphere results, the REPTRAN band quadrature used in those results, and their derivation. The original solver and molecular lookup files are retrieved independently.
- [Conda-forge reference package](https://anaconda.org/conda-forge/rubin-libradtran). The executable used here came from build `2.0.6-py314pl5321h3aae1cf_1`, package SHA-256 `7554c2f6704ded36e16512e27d3cba1026a0ec624eb78136f88f63ef61f7eee2`.
- AFGL US Standard 1976 profile, Anderson et al., AFGL-TR-86-0110, `data/atmmod/afglus.dat`, SHA-256 `dab26290ed81c762ed0c607e5dc2d53393c1462a0c3a528bc5e3f5935191cfb5`. The published `reptran_solar_coarse.cdf` input digest is `14aecaa2b2207c539fd22a2b0803dbba5a9a8f072b717430048e838cea8c2be5`. [REPTRAN formulation](https://doi.org/10.1016/j.jqsrt.2014.06.024)

## Time a resident CUDA batch

Run `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python studies/spectral-radiation/benchmark/warp_scaling.py` with Warp and a CUDA device. The script prints preparation, warmed solve and sequential NumPy timings for 1, 16 and 64 two-layer columns. It synchronizes CUDA before stopping each timer, then downloads the results and checks the upwelling spectra.
