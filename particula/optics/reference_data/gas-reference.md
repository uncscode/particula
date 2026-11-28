# AFGL US standard atmosphere and REPTRAN quadrature

The version 1 Earth reference uses the U.S. Standard Atmosphere 1976 as tabulated by Anderson et al. in AFGL-TR-86-0110. The 49 layers extend from 120 km to sea level. The two NPZ files contain the same air profile. Water and ozone are molecular number densities in m⁻³, pressure is in Pa, and temperature is in K. Absorption optical depths are dimensionless positive integrals across each layer. They were computed with libRadtran 2.0.6 using its default gas profiles and the coarse REPTRAN solar gas tables. These are fixed reference conditions. Scaling water or ozone independently requires recomputing absorption with a gas model.

The full file retains all 4,082 positive-weight quadrature nodes in complete bands within 280–4000 nm. The reduced file groups these nodes into 22 broader bands with 217 representative wavelengths. Node weights are lengths in metres. The band bounds repeat for nodes in the same band. Use band sums of flux density times node weight to obtain W/m². The solar boundary separately conserves the TSIS-1 energy of each band.

The full band edges, representative wavelength definitions and native quadrature weights come from libRadtran's REPTRAN distribution. Its conda-forge package declares GPL-2.0-or-later and supplies the accompanying license. The molecular lookup tables and solver executable are external inputs to the numerical derivation. [Upstream distribution](https://www.libradtran.org/doku.php?id=download) · [Reference algorithm](https://doi.org/10.1016/j.jqsrt.2014.06.024) · [Upstream license text](REPTRAN-COPYING.txt) · [Reproduction commands, hashes and fine-resolution comparison](https://github.com/uncscode/particula/tree/main/studies/spectral-radiation/benchmark)

| File | SHA-256 |
| --- | --- |
| `gas_full.npz` | `f08ef480333b3bfb6daf5c0972a201d5736d7b95a3738b17f826cf7d6437bf8d` |
| `gas_reduced.npz` | `92ddac24d33d8966ca1226f54a8c67ff84b83f68fc24e7512315a026eb648834` |
