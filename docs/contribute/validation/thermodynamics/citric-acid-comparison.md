# Citric acid droplet hygroscopicity

## Comparison specification

Use all 12 measured points for aqueous citric acid at 293.15 K in Marsh et al. (2017), Figure S9.1 and Table S9.2. The authors report water activity, positive and negative activity errors, solute mass fraction (MFS) and MFS error. The plotted UNIFAC curve is a model prediction. Retain the measured points and their published error columns. [Primary paper](https://doi.org/10.5194/acp-17-5583-2017) · [Supplement, p. 11](https://acp.copernicus.org/articles/17/5583/2017/acp-17-5583-2017-supplement.pdf) · [Data archive](https://doi.org/10.5523/bris.3eoilb28zwoqr2jifjxj7cntex)

This case was selected because it has a traceable composition observable with errors on both axes and a separate published BAT reference for the same organic. The alternative water/glycerol evaporation study reports a time-dependent radius that also depends on thermal and transport assumptions. [Time-resolved evaporation study](https://doi.org/10.1080/02786826.2011.652750)

Freeze the primary model inputs at the values used for the published citric acid BAT curve (Fig. 2b, water/organic molar mass ratio 0.0938, O:C 1.16667, organic density 1469.184958 kg/m³, hydroxyl-equivalent fit). Compare activity evaluated at the reported MFS against the measured water activity. Use signed mean error, RMS error, maximum absolute error, and the count of points consistent with the reported two-axis uncertainty envelope. Compute that envelope from the extrema of BAT activity across the reported MFS interval and the asymmetric experimental activity limits. Fit no parameters. Report a sensitivity using the experimental melt-density estimate (1550 kg/m³, Table S9.1) separately. The original activity fit used 298.15 K and lacks an explicit temperature correction.

As a kinetic cross-check, hold dry citric acid mass fixed and let water exchange with a prescribed vapor reservoir until the flux vanishes. For bulk activity comparisons take the flat-surface limit. Use distinct initial water contents and refine solver and final-time settings until the recovered activity varies by less than 0.0001. Report water exchange with the reservoir and any remaining mismatch to the standalone root. These data quantify hygroscopicity, so do not infer relaxation times from Figure S9.1.

## Direct activity comparison

Using all 12 [Table S9.2 rows](../../../data/thermodynamics/marsh2017-citric-acid.csv), the signed bias (predicted minus measured) was +0.05096 in water activity, RMS error 0.06160, and maximum absolute error 0.10645. None of the 12 predicted intervals overlapped the corresponding measured interval under the two-axis rule above. Using the measured melt-density estimate increased the RMS error to 0.06727. An inverse composition root reproduced the prescribed activities to within 1.5 × 10⁻¹³ for both densities.

These residuals exceed the tabulated measurement errors for this configuration. The model uses reduced organic descriptors, and its room-temperature fit was applied at the experimental temperature without an activity-temperature correction. This comparison alone cannot assign the discrepancy to either approximation. The kinetic cross-check, executable figure and independent reproduction remain pending.

Data attribution: Marsh et al. (2017), Table S9.2 in the article supplement, reproduced under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) with column headings normalized to machine-readable names; numeric values as published. Verify reuse terms against the supplement cover.
