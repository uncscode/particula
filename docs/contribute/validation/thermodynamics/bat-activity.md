# Binary BAT reference checks

The [authors' archived inputs and workbook](https://github.com/Gorkowski/Binary_Activity_Thermodynamics_Model/tree/6e0d5a4d1a9af4571358f0539682cbc0cc1e0257/Matlab_source_code/paper_figures) supply an external check on direct BAT activities. Figure_FitO2CMm_example_fits_draft1.m selects entries 5 and 25 (MATLAB indexing) of binary_fit_data_imported.mat, evaluates the hydroxyl fit at 10,001 organic mole fractions from 1 to 0 and reports the corresponding water mole fraction on the horizontal axis. Use the same molar-mass ratios and the authors' estimated organic densities; the workbook rounds its molecular mass descriptions. [Gorkowski, Preston and Zuend (2019), Fig. 2](https://doi.org/10.5194/acp-19-13383-2019)

| Fig2 workbook columns | O:C | H:C | Water/organic molar mass ratio | Organic density (kg/m³) | Maximum absolute error, water | Maximum absolute error, organic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I–K | 1.16667 | 1.33333 | 0.0938 | 1469.18495807944 | 7.8e-16 | 8.9e-16 |
| P–R | 0.16667 | 2.33333 | 0.17632 | 802.78721257797 | 3.91e-4 | 9.60e-4 |

Errors compare the direct CPU function with workbook activity values after converting water mole fraction to organic mole fraction. The first curve checks citric acid. The second curve's discrepancy is unresolved and exceeds the 1e-4 target used separately for interpolating CPU results. Neither curve measures experimental accuracy. Three interior citric-acid points are retained in the activity regression suite. A separate batch regression compares the same particle evaluated alone and among other compositions.
