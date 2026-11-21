# How closely can nominal organic precursor loadings specify the campaign peaks?

The 24 Manchester experiments in Voliotis et al. Table 1 have 18 positive SOA peaks, three reported zeros (runs 11–13) and three missing aerosol endpoints (3, 9, 17). Seven nominal VOC-loading combinations have multiple positive endpoints. The component doses in mixtures follow the order of the precursor list in the paper. The table below uses the published peak column, in µg m⁻³. Its displayed precision is preserved in the [24-row source transcription](campaign_peaks.csv). [Paper, Table 1 and Table 3 caption](https://acp.copernicus.org/articles/22/14147/2022/) · [Publisher’s Table 1 spreadsheet](https://acp.copernicus.org/articles/22/14147/2022/acp-22-14147-2022-t01.xlsx)

| Nominal component doses (ppb) | Runs and reported peak masses, in run order | Minimum worst relative error for positive values |
|---|---|---:|
| α-Pinene 309 | 1, 2, 3 → 273.2, 283.1, missing | 1.78% |
| α-Pinene 155 | 4, 5 → 68.6, 109.5 | 22.96% |
| α-Pinene 103 | 6 → 31.5 | 0% |
| o-Cresol 400 | 7, 8, 9 → 28.2, 56.0, missing | 33.02% |
| o-Cresol 200 | 10 → 22.8 | 0% |
| Isoprene 164 | 11, 12 → 0.0, 0.0 | Undefined |
| Isoprene 55 | 13 → 0.0 | Undefined |
| o-Cresol 200 + isoprene 82 | 14, 15 → 11.2, 9.4 | 8.74% |
| α-Pinene 155 + o-cresol 200 | 16, 17, 18 → 122.3, missing, 130.1 | 3.09% |
| α-Pinene 155 + isoprene 82 | 19, 20, 21 → 96.6, 100.9, 75.2 | 14.59% |
| α-Pinene 103 + o-cresol 133 + isoprene 55 | 22, 23, 24 → 55.5, 51.4, 58.0 | 6.03% |

For every group a deterministic loading-only function must give a common value *m*. Let *a* and *b* be its smallest and largest positive observed endpoints. Minimizing the larger of (*m*−*a*)/*a* and (*b*−*m*)/*b* gives *m* = 2*ab*/(*a*+*b*) and a worst relative error of (*b*−*a*)/(*b*+*a*). All intermediate positive endpoints have smaller or equal relative error. Each distinct loading may take its own fitted peak. The maximum across these independent optima is thus the global minimum worst error for this broad function class. Reported zeros can be reproduced by assigning 0 to the isoprene-only groups. Percentage errors at zero are undefined. Missing outcomes supply no constraint.

The **33.02% campaign minimum** is set by the 400 ppb o-cresol runs. Their peak ratio is 1.986 and the best common peak is 37.51 µg m⁻³. NOₓ, seed, measured histories and other chamber inputs can differentiate experiments that share nominal organic doses. A dynamic model using such information is outside this mathematical bound. The bound applies to these reported numbers and does not estimate latent peaks, chemical rates, instrument uncertainty or the attainable error of a specific physical mechanism.

Run `python loading_only_bound.py` to regenerate [all group optima](campaign_peak_bound.csv) from the [source table](campaign_peaks.csv). The transcription was checked independently against the publisher's spreadsheet cells for every run number, each precursor dose, NOₓ, seed and peak. Source SHA-256 values were `0a2cbdf14b5bc166567b737a003af1113ad31e7280366d4057bc60767ba81c3d` for the XLSX and `a247e86fcf2d29bc429119f05a0286e484a80fcd3bc647d7e8a6326c130ae351` for the [paper PDF](https://acp.copernicus.org/articles/22/14147/2022/acp-22-14147-2022.pdf).
