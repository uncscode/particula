# Offline solar reference

The ideal Earth reference uses the LASP 1 nm binned full-spectrum TSIS-1 HSRS at 1 au. The table retains the native bin means and a half-bin margin for interpolation at 280 and 4000 nm. The included 280–4000 nm interval excludes solar energy outside the modeled range. Source attribution and version history are available from [LASP LISIRD](https://lasp.colorado.edu/lisird/data/tsis1_hsrs_binned_fs/).

From the repository root, download and rebuild the numerical table with the following commands.

```sh
curl -fLsS 'https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_binned_fs.csv' -o /tmp/tsis1.csv
python studies/spectral-radiation/reference/build_solar.py /tmp/tsis1.csv particula/optics/reference_data/tsis1.txt
```

Coddington et al. describe the full-spectrum extension and its absolute normalization. [Full-spectrum dataset DOI](https://doi.org/10.25980/249q-fs39) · [Reference spectrum paper](https://doi.org/10.1029/2022EA002637)

Original CSV SHA-256 is `e63693db630363c0263847154a6aec4c3828d94ac13605dab27949de58f19bc1`.
