# Four experiments for modeling secondary organic aerosol formation

Start with **Voliotis et al. (2022), Manchester chamber experiments 4 and 5**. Both used nominally 155 ppb α-pinene and similar seed mass. The authors report 68.6 and 109.5 µg m⁻³ SOA, respectively. Run-specific precursor decay offers a way to ask whether the difference emerges during oxidation or later, as products partition to particles. The studies below are ranked by the comparisons they enable. All simulations described here are proposed.

## 1. Voliotis et al. (2022) · Daytime chamber chemistry

The Manchester campaign includes 24 experiments with α-pinene, o-cresol, isoprene and mixtures. For runs 4 and 5, Table 1 gives initial NOₓ of 26 and 35 ppb and seed mass of 45.7 and 47.8 µg m⁻³. The nominal VOC injection uncertainty is ±15%. Table S1 identifies separate records for total particle mass and number, NO, NO₂ and O₃, and for organic mass and fractional VOC decay. Run 4's organic-data DOI resolves to the 6 July 2019 experiment. The catalog and variable mapping were checked. File-level time alignment and units still need inspection. [Paper, Table 1](https://acp.copernicus.org/articles/22/14147/2022/) · [Supplement, Table S1](https://acp.copernicus.org/articles/22/14147/2022/acp-22-14147-2022-supplement.pdf) · [Run 4 gases and particles](https://doi.org/10.25326/9WDC-BG18) · [Run 4 organics and VOC decay](https://doi.org/10.25326/5AJ0-6150) · [Run 5 gases and particles](https://doi.org/10.25326/AH45-9F95) · [Run 5 organics and VOC decay](https://doi.org/10.25326/VM0N-Q983)

**First comparison.** Estimate effective oxidant exposure for each run from α-pinene decay. Use run 4 to constrain a reduced oxidation-product distribution, then predict run 5 organic mass over time and per unit of reacted α-pinene using the same product parameters. This holds out the aerosol observations while using gas observations as inputs in both runs. Compare mass on the authors' particle-wall-loss basis and vary vapor wall loss separately. Differences between the two runs alone cannot isolate a NOₓ effect.

## 2. Nah et al. (2016) · Ozone and seed-surface controls

Dark α-pinene ozonolysis offers a simpler oxidation mechanism. Table 1 provides six runs spanning 100 and 500 ppb O₃, with zero, low or high ammonium sulfate seed surface area. For the seeded runs, low/high seed gave peak mass yields of 23.3/23.3% at 100 ppb and 28.6/28.1% at 500 ppb. The table includes uncertainties, precursor loadings and SMPS-derived peak organic mass. The numerical endpoints are directly usable from the paper. A downloadable time-series file was not verified. [Paper](https://acp.copernicus.org/articles/16/9361/2016/) · [Table 1 and measurement methods](https://acp.copernicus.org/articles/16/9361/2016/acp-16-9361-2016.pdf)

**First comparison.** Use the four seeded runs to test whether one reduced ozonolysis mechanism reproduces both the ozone response and the weak seed response within reported uncertainty. Sweep vapor wall loss and product volatility on CUDA. Calculate yield using reacted precursor at peak SOA growth. Some α-pinene remained in the 100 ppb experiments.

## 3. Takeuchi et al. (2022) · Nighttime nitrate chemistry

The authors oxidized α-pinene and limonene separately, sequentially and simultaneously. ICARUS experiment APN-2 includes an injection timeline, SMPS number and volume distributions, O₃, NO₂, NO and particle composition files. The chamber received 200 ppb of N₂O₅ over a finite injection period. That quantity describes the nitrate-radical source. The public record exposes filenames and column names, and the bulk download requires a verified email. [Paper](https://doi.org/10.1038/s41467-022-35546-1) · [APN-2 data and timeline](https://icarus.ucdavis.edu/experiment/852)

**First comparison.** Constrain each precursor separately and test whether the same parameters predict particle growth after sequential versus simultaneous oxidation. Use the injection history to drive N₂O₅ and NO₃ chemistry. Growth or a dimer signal alone cannot distinguish gas-phase cross-reactions from particle-phase oligomerization.

## 4. Palm et al. (2018) · Ambient Amazon air in an oxidation flow reactor

During GoAmazon2014/5, the authors oxidized ambient air and measured the organic aerosol mass change. Their standard-addition experiments provide a narrower first target than the entire ambient mixture. Figure 7 reports individual VOC yields against organic aerosol loading. The authors' figure archive downloaded successfully and contains the Igor Pro experiment `Fig7.pxp`. Its numerical waves and units still need mapping. [Paper, Figure 7](https://acp.copernicus.org/articles/18/467/2018/) · [Figure-data ZIP](https://cires1.colorado.edu/jimenez-group/grp-figs/Palm_2018_ACP.zip) · [ARM campaign archive](https://www.arm.gov/research/campaigns/amf2014goamazon)

**First comparison.** Predict the incremental yield from an α-pinene standard addition, then examine what transfers to unspiked air. Represent reactor residence time, oxidant exposure and condensable-vapor losses explicitly. Unmeasured ambient precursors limit how confidently total SOA can be attributed to individual compounds.

## What CUDA contributes to the first study

The selected Particula checkout documents gas-phase mass-action chemistry on CPU and resident CUDA, plus reversible condensation and a separate irreversible reactive-uptake pathway. The benchmark needs a literature-based reduced α-pinene mechanism, product volatilities and chamber-specific forcing. Use reversible condensation for semivolatile products. Run paired chamber conditions and ensembles of product volatility and vapor wall loss as independent boxes. Before interpreting a fit, compare a representative CUDA trajectory with CPU and close the carbon budget across precursor, products, particles and wall losses. [Gas chemistry](../docs/guide/processes/gas-chemistry.md) · [Resolved chemistry and uptake](../docs/guide/processes/resolved-chemistry.md)
