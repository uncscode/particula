# What the optical and ion measurements constrain about sulfur chemistry

Xiong and colleagues measured how much individual particles grew when humidified after SO₂ exposure. They also recorded ion chromatograms at 100 nm. Several reaction histories reproduce the optical series. The late ion peaks supply a second test, but its interpretation depends on how much chloride reaches the detector. [Main comparison](index.md) · [Xiong et al., Figures 2–5 and supplement](https://acp.copernicus.org/articles/26/12183/2026/)

## Reconstructing the optical reference

The authors inferred composition by weighting the cubed growth factors of pure NaCl and NaHSO₄ by their fractions of dry particle volume. Fitting their plotted decline in dissolved NaCl concentration gave 0.653 h⁻¹ at 100 nm, compared with the published 0.6523 h⁻¹. Repeating the calculation across the size series recovered the maximum near 200 nm. For the chemical fits we replaced that two-salt mixture with a sodium, chloride and sulfur inventory and evaluated non-ideal ion activities with AIOMFAC. Measured pure-salt endpoints anchor the modeled growth. At readout humidities up to and including 75%, a fitted weight allows the water uptake of separate salt components to contribute after drying.

## Which parts of the fit are independent?

The fit used 144 optical observations and the four earliest sulfate-to-chloride peak ratios. Growth measurements made at several humidities after the same exposure share the particle history. We therefore gave each combination of diameter and exposure time the same total weight in the optical fit. The full comparison reports unweighted error on the 156 reserved optical observations. The numerical scales used to fit the data and the range read from each chromatogram do not establish a statistical confidence interval. The late chromatograms motivated the chloride-retention and signal-background tests, so those tests are retrospective. [Digitized measurements](https://raw.githubusercontent.com/uncscode/particula/main/docs/data/xiong-sea-spray/observations.zip)

The two leading models differ by only 0.0001 in their reserved growth-factor errors. Doubling the time resolution changed growth predictions by less than 0.000019 through five hours. A larger uncertainty came from estimating water uptake between tabulated compositions. Near a solid–liquid boundary, a direct equilibrium calculation differed from the interpolated value by as much as 0.0104 in growth factor. This local discrepancy gives another reason to avoid ranking the two pathways by such a small difference in overall error.

Automatic differentiation supplied each model’s rate gradients. Across the reference, fitting and sensitivity calculations, 138 fits used 341 optimizer starts. Independent CUDA integrations on an NVIDIA GeForce RTX 5060 checked six networks at three diameters; calculated inventories agreed within 0.000174 mol per mol of initial sodium. [Model and fitting source](https://github.com/uncscode/particula/tree/main/studies/xiong-sea-spray)

## Chloride recovery and sulfate yield

Both leading fits overpredicted sulfate signal relative to chloride at three hours. The central peak-area ratio was 2.62. Integrating the edges of the printed traces gave 1.49–5.00. The parallel model gave 7.60 and the direct route with retained chloride gave 5.13. Assuming the equal ion response used in the paper’s equation 8, a direct route with chloride background and a retained-chloride route without that background gave nearly identical ratios of 3.82 and 3.85. A chloride blank, recovery standards and oxidation-state-preserving collection would make those peaks more informative. [Plotted peak ratios](https://raw.githubusercontent.com/uncscode/particula/main/docs/data/xiong-sea-spray/ion-time.csv)

Both leading models also predicted supersaturation with respect to mirabilite, a hydrated form of sodium sulfate. If the hydrate crystallized during exposure or drying, it could change water uptake at a given sulfate yield. Identifying the solid phase would test that assumption independently of the optical fit.

## Transfer from 200 to 20 ppm SO₂

At 100 nm, reducing the sulfur entry rate tenfold gave a ten-hour sulfate concentration of 1.59–2.62 mol L⁻¹ across treatments. Xiong’s retrieval gave 2.85 mol L⁻¹. Fitting the 1, 2 and 4 h lower-dose observations together with the high-dose calibration gave uptake-rate multipliers of 0.14–0.28 across six treatments. Allowing the multiplier to vary improved late NaCl predictions in all six cases. The parallel pathway gave the smaller late error in the log NaCl concentration ratio, while the retained-chloride pathway gave the smaller late error in sulfate. The two retrieved concentrations share an optical volume estimate. Measuring sulfate and chloride independently at both gas concentrations would test the transfer.
