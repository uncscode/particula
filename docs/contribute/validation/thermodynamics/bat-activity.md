# Binary BAT reference checks

The [authors' archived inputs and workbook](https://github.com/Gorkowski/Binary_Activity_Thermodynamics_Model/tree/6e0d5a4d1a9af4571358f0539682cbc0cc1e0257/Matlab_source_code/paper_figures) supply an external check on direct BAT activities. Figure_FitO2CMm_example_fits_draft1.m selects entries 5 and 25 (MATLAB indexing) of binary_fit_data_imported.mat, evaluates the hydroxyl fit at 10,001 organic mole fractions from 1 to 0 and reports the corresponding water mole fraction on the horizontal axis. Use the same molar-mass ratios and the authors' estimated organic densities; the workbook rounds its molecular mass descriptions. [Gorkowski, Preston and Zuend (2019), Fig. 2](https://doi.org/10.5194/acp-19-13383-2019)

| Fig2 workbook columns | O:C | H:C | Water/organic molar mass ratio | Organic density (kg/m³) | Maximum absolute error, water | Maximum absolute error, organic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I–K | 1.16667 | 1.33333 | 0.0938 | 1469.18495807944 | 7.8e-16 | 8.9e-16 |
| P–R | 0.16667 | 2.33333 | 0.17632 | 802.78721257797 | 3.91e-4 | 9.60e-4 |

Errors compare the direct CPU function with workbook activity values after converting water mole fraction to organic mole fraction. The first curve checks citric acid. The second curve's discrepancy is unresolved and exceeds the 1e-4 target used separately for interpolating CPU results. Neither curve measures experimental accuracy. Three interior citric-acid points are retained in the activity regression suite. A separate batch regression compares the same particle evaluated alone and among other compositions.

## Figure 9 activity curves and liquid phase separation

The authors archive three binary composition scans as `DATA for figure kappaCCN {mid,lower,zero} O2C.mat`. Each contains 100,001 organic mole fractions from 1 to 0. At Particula revision `b253831069`, evaluate the direct binary activities at `mole_frac_scan` with `fix_molarmass1` (0.09008), `O2C_value`, and `density_org_g_cm3` multiplied by 1000. Compare with the raw `activity_water` and `activity_org` arrays before applying any phase or curvature correction. The Figure 9 caption specifies 298.15 K, while all three archived inputs store 293 K. This distinction matters when reproducing the Kelvin term. [Archived figure inputs](https://github.com/Gorkowski/Binary_Activity_Thermodynamics_Model/tree/6e0d5a4d1a9af4571358f0539682cbc0cc1e0257/Matlab_source_code/paper_figures/SI_figures) · [Published figure](https://doi.org/10.5194/acp-19-13383-2019)

| O:C | Organic density (kg/m³) | Archived phase separation flag | Maximum absolute water activity error | Maximum absolute organic activity error |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 1243.493333 | 0 | 2.78e-6 | 1.67e-6 |
| 0.25 | 1071.717647 | 1 | 1.11e-6 | 3.37e-6 |
| 0 | 701.300000 | 1 | 2.82e-2 | 35.19 |

For O:C 0.25, the lower convex hull of the calculated molar mixing Gibbs energy identifies a gap. Refining both endpoints to equal water and organic activities gives organic mole fractions 0.00353327 and 0.41275286, with shared activities 0.99679965 (water) and 0.41214449 (organic). Values inside this interval on the raw homogeneous curve describe a constrained liquid. The O:C 0 case gives a substantial discrepancy even on the raw activity curve. Two-phase kinetic calculations require verification of the reference endpoints and phase amounts. Only the miscible case is used here for the homogeneous Figure 9 regression.

## Adaptive GPU restart

The adaptive checkpoint records the organic descriptors, the linear organic-mole-fraction interpolation convention and a SHA-256 digest of both activity columns in grid order (little-endian float64). The disk round trip on Warp CPU and CUDA reproduces the scaled state and controller exactly. Changing organic O:C rejects the saved checkpoint. The CUDA resident coupled test also covers both process orders with dilution, unchanged device buffers across the handoff, and a rejected table digest. These checks exercise the adaptive condensation paths.
