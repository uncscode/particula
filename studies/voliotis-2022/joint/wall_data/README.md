# Independent Manchester chamber particle-loss calibration

A fit of the Particula neutral rectangular-chamber loss model to 18 readable diameter-bin means gives an effective eddy diffusivity coefficient **E = 13.7 s⁻¹**. The digitized apparent loss falls from 1.58 × 10⁻⁴ s⁻¹ at 44 nm to 4.88 × 10⁻⁵ s⁻¹ at 541 nm. At these same sizes the fitted model gives 3.03 × 10⁻⁴ and 4.30 × 10⁻⁵ s⁻¹. The largest underestimate occurs at 473 nm, where Particula predicts 4.62 × 10⁻⁵ s⁻¹ and the published figure gives approximately 8.00 × 10⁻⁵ s⁻¹.

## Source and extraction

Shao and colleagues measured the decay of ammonium sulfate seed in darkness for at least 4 hours. They injected 50–100 µg m⁻³ of seed with modal diameter near 100 nm and estimated decay by fitting exponentials to DMPS number in each mobility bin. Figure 5 reports mean ±1 standard deviation. Its blue circles represent the nine experiments at 50% RH with air circulation on. The paper directs readers to the authors to obtain the underlying data. No numeric table accompanies Figure 5 in the paper. [Shao et al. (2022), section 3.5 and data availability](https://amt.copernicus.org/articles/15/539/2022/)

Page 10 of the publisher PDF (printed p. 548) contains the figure as a 2067 × 1827 RGB raster. We read blue mean-marker centers, with symmetric error bars as a cross-check where marks overlap. We used frame centers x = 390.5 and 2002 px for 0 and 600 nm, and y = 1589.5 and 34 px for 0 and −2 × 10⁻⁴ s⁻¹. The rates here are positive decay magnitudes. Source resolution is 0.37 nm and 1.29 × 10⁻⁷ s⁻¹ per pixel. Symbol width exceeds one pixel, and overlapping curves make small centers less certain. A 10-pixel shift, for scale, corresponds to 3.7 nm or 1.29 × 10⁻⁶ s⁻¹. [Publisher PDF](https://amt.copernicus.org/articles/15/539/2022/amt-15-539-2022.pdf)

The [overlay](figure5_digitization.png) places crosses at the transcribed centers. Magenta indicates the 14 readily separated blue markers from 105 to 541 nm. Orange indicates four additional readable markers from 44 to 93 nm. Several crowded markers below 100 nm were omitted. The overlay was checked visually against the underlying marker positions and axis ticks. The source figure is © Shao et al. 2022 and the overlay adds crosses and reduces the resolution. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Fit and transfer to the formation study

We used a 3 × 2 × 3 m rectangular box, 298.15 K, assumed pressure 101325 Pa and particle density 1770 kg m⁻³. Mobility diameters were treated as spherical physical diameters. Each measured diameter contributes one squared log ratio of predicted to observed loss. The scalar parameter E is shared across sizes. The spread bars describe variation between chamber runs. Individual size bins share experiments, and the cross-bin covariance is unavailable.

| Bins | Residual minimized | E (s⁻¹) | RMS log ratio | RMS rate error (s⁻¹) |
|---|---|---:|---:|---:|
| 18 readable, 44–541 nm | Log rate | 13.74 | 0.246 | 3.87 × 10⁻⁵ |
| Same 18 | Absolute rate | 8.54 | 0.340 | 2.78 × 10⁻⁵ |
| 14 isolated, 105–541 nm | Log rate | 16.55 | 0.171 | 1.51 × 10⁻⁵ |
| Same 14 | Absolute rate | 13.95 | 0.191 | 1.29 × 10⁻⁵ |

This coefficient calibrates apparent dark seed decay under the chosen model. The authors discuss coagulation below 100 nm, seed charge from unneutralized particles, and chamber bag history over 2017–2019. A neutral deposition model captures only part of this size dependence. Applying this E alongside explicit coagulation could count some small-particle disappearance twice. The 8.5–16.6 s⁻¹ sensitivity to diameter selection and residual weighting is a set of alternative fit specifications. The nine individual loss curves and the Voliotis-run-specific chamber history would be needed to assign uncertainty to transfer between experiments. [Shao et al. (2022), sections 3.5 and 3.5.1](https://amt.copernicus.org/articles/15/539/2022/)

## Reproduce

From the worktree root, using Python with Particula, SciPy, NumPy, and Pillow available, run the following.

```sh
PYTHONPATH=. python studies/voliotis-2022/joint/wall_data/fit_wall.py
```

The [marker coordinates](figure5_markers.csv) reproduce the [derived rates and native predictions](wall_rates.csv) and [fit results](wall_fit.json). To reproduce the overlay and verify the PDF hash, retrieve the publisher PDF and extract its embedded raster using Poppler.

```sh
curl -fLsS https://amt.copernicus.org/articles/15/539/2022/amt-15-539-2022.pdf -o /tmp/shao-2022-mac.pdf
pdfimages -f 10 -l 10 -png /tmp/shao-2022-mac.pdf /tmp/shao-2022-figure5
PYTHONPATH=. python studies/voliotis-2022/joint/wall_data/fit_wall.py \
  --source-pdf /tmp/shao-2022-mac.pdf --source-image /tmp/shao-2022-figure5-000.png
```
