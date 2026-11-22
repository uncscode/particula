# How well does Particula reproduce organic aerosol formation in the Manchester chamber?

![Manchester ternary mixture with a measured aerosol yield of 11.7%, a single-precursor estimate of 16.1% and a Particula bulk fit of 13.8%. At two hours, the measured and refitted size distributions peak in the same diameter bin. The model using bulk-fit chemistry peaks one bin larger.](../../../../images/manchester-soa/graphical-abstract.svg)

How well can a model reproduce both the amount of aerosol a chamber produces and the sizes of its particles? We used Particula to search for chemistry shared across 15 Manchester chamber experiments. For the mixture of α-pinene, o-cresol and isoprene, Voliotis and colleagues measured a wall-corrected organic aerosol yield of 11.7% at 461 µg/m³ of total precursor consumed. Their single-precursor fits gave 16.1% when evaluated at the modeled consumption of each precursor. A Particula bulk model fitted to mass and particle number gave 13.8%. Particle size offered a second test. At two hours, a particle-resolved model using chemistry from the bulk mass fit placed the peak one diameter bin above the measured peak. Including size measurements in the refit brought the peak into the measured 126–158 nm bin, although the refit underestimated the five-hour geometric mean diameter. Across the campaign, several reaction pathways could reproduce similar mass histories, leaving their chemical products unresolved.

## What Manchester’s precursor mixtures revealed

Voliotis and colleagues asked whether yields measured for individual precursors could explain the aerosol produced by their mixtures. They illuminated α-pinene, o-cresol and isoprene with nitrogen oxides and ammonium sulfate seed in an 18 m³ chamber. At matched precursor consumption, the α-pinene/o-cresol pair produced more SOA than expected from the single-precursor yields. Isoprene suppressed the α-pinene yield. A possible enhancement with o-cresol and isoprene was too small to resolve, and the ternary yield was close to the single-precursor expectation. The authors noted that even at matched consumption, different amounts of aerosol remain available for vapors to partition into. [Voliotis et al., Figure 5 and conclusions](https://acp.copernicus.org/articles/22/14147/2022/#section5)

Of 24 experiments, 15 had overlapping precursor decay, suspended organic mass and particle counts. Their first five hours supplied 1,258 mass and 682 number readings. We modeled deposition explicitly when comparing suspended mass and added deposited mass back when comparing the authors’ wall-corrected yields. [Voliotis et al., experimental design and Figure 3](https://acp.copernicus.org/articles/22/14147/2022/) · [Instrument records, coverage and source DOIs](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/data/CAMPAIGN_SERIES.md)

Companion measurements also distinguish mechanisms with similar yields. Online gas and particle measurements and offline filter analyses identified o-cresol-derived nitroaromatics in both o-cresol binary mixtures. A formula found only in a mixture leaves its structure and carbon sources unresolved. Thermal-denuder and mass-spectrometer measurements also found more low-volatility material in the ternary aerosol than single-precursor contributions predicted. [Du et al., online and offline composition](https://amt.copernicus.org/articles/15/4385/2022/) · [Shao et al., filter composition](https://acp.copernicus.org/articles/22/9799/2022/) · [Shao et al., carbon oxidation state](https://ar.copernicus.org/articles/3/619/2025/) · [Binary-mixture measurements](https://acp.copernicus.org/articles/21/14251/2021/) · [Isoprene and ternary-mixture measurements](https://acp.copernicus.org/articles/22/13677/2022/)

Wang and colleagues measured water uptake below and above saturation. In an α-pinene experiment, humidified growth factors fell from 1.65–1.72 to about 1.15 within an hour as organics accumulated. Independent particle-rebound experiments showed strong humidity and composition dependence. We used measured water uptake in the fits below. Composition, volatility and rebound informed the interpretation of the chemistry. [Wang et al., water uptake, Figures 3–4](https://acp.copernicus.org/articles/22/4149/2022/) · [Wang et al., particle rebound, Figures 1–2](https://acp.copernicus.org/articles/21/11303/2021/)

## Fitting shared chemistry to 15 chamber histories

Two runs with nominally 155 ppb α-pinene illustrate why a single mass discrepancy tells us little about missing chemistry. The first reached 68.47 µg m⁻³ and the second 109.46 µg m⁻³. Applying the first run’s fit to the second run’s measured precursor decay and seed conditions gave 76.88 µg m⁻³, rising to 104.23 µg m⁻³ at the favorable bound of the injection uncertainty. The experiments also used different instruments for their initial counts. [Transfer experiment](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/REPORT.md) · [Injection and wall-loss sensitivity](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/results/injection_and_walls.csv)

For the campaign-wide fit, all 15 runs shared chemical parameters. Each supplied its measured parent-precursor decay and could adjust its injection dose by ±15%. We fitted two objectives separately. One minimized the worst mass-trajectory error across runs. The other minimized the worst error over both mass and number. Mass error was the root-mean-square error (RMSE) divided by the observed run peak, with a 1 µg m⁻³ normalization floor. Count error was normalized by initial particle number. The resulting scores measure reproduction of the fitted experiments. [Model and scoring details](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/chains/METHODS.md)

### Wall deposition and precursor competition

With the early chemistry and coagulation model, replacing constant deposition with size-dependent wall loss reduced worst joint error from 24.93% to 21.41%. In a sweep of 4,096 chemistry and wall-mixing combinations, chemistry accounted for 86.3% of peak-mass variance in the median experiment and wall mixing for 99.3% of five-hour count variance. Those shares depend on the chosen ranges. A wall coefficient fitted to dark seed decay transferred poorly to the illuminated chamber. [Wall and chemistry assessment](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/ASSESSMENT.md) · [Per-experiment variance shares](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/uncertainty/sensitivity_by_run.csv)

At the same wall eddy coefficient of 5.11 s⁻¹, adding a promptly condensing product barely affected worst mass error (20.98% to 20.97%). Gas-phase competition reduced it to 17.65%. Allowing isoprene disappearance to divert some α-pinene and o-cresol products toward volatile material gave 16.00%. Manchester’s α-pinene/isoprene suppression is consistent with such diversion, although the o-cresol/isoprene measurements provided no clear evidence of suppression. [Chemical pathway comparisons](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/CHEMICAL_PATHWAYS.md) · [Matched fits](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/pathways/comparison.csv)

### How the agentic search arrived at 70 channels

In each of the first 20 rounds, the agent selected a precursor-compatible equation, fitted its effective rate with CUDA simulations, and recorded the proposal before seeking an experimental analogue. Twelve proposals had direct evidence for a corresponding process and eight had broader reaction-family support. For instance, irradiation can deplete particle-bound peroxides in α-pinene-derived SOA. This supports a loss pathway while leaving its fitted rate in Manchester undetermined. In 15 s replays the 20-channel search reduced worst mass error from 15.87% to 12.09% and worst joint error from 18.04% to 17.92%. [Equations and primary evidence for all 20 rounds](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/discovery/REACTION_EVIDENCE.md) · [Search method and controls](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/discovery/METHODS.md) · [Direct photolysis experiment](https://doi.org/10.1021/es502350u) · [First-stage results and evaporation controls](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/REACTION_DISCOVERY.md)

The first network could form an α-pinene dimer but pooled its downstream products. A second search tested 40 predeclared sequences spanning five product families. Oxidation, fragmentation and further reaction of released vapor added 50 distinct transfers because some sequences share steps. The full network offered 70 channels whose fitted rates could return to zero. Fifteen sequences had a published reaction-family analogue. For the other 25, the literature supported only individual steps or more distant substrates. [All 40 proposals and evidence grades](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/chains/evidence.csv) · [Experimental evidence for α-pinene dimer oxidation](https://doi.org/10.1021/acs.est.2c07630)

Each addition competed with a fresh fit of its predecessor. A separate control spent all 40 rounds refitting the original network. In round 35, refitting the predecessor alone reproduced the apparent improvement from adding fragmentation of isoprene-derived vapor. Most final optimizations reached their iteration limit, so the table reports the best fits found. Each entry uses the larger error for forward and reversed reaction order at a 1.875 s step, with parameters fitted at 15 s. [Control scores for the 40 comparisons](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/chains/rounds.csv) · [Calibrated parameter sets](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/chains/continued.json)

| Chemistry available | Worst mass trajectory error | Worst joint mass or number error |
|---|---:|---:|
| Original 20 channels, refitted | 12.08% | 17.86% |
| Original channels plus α-pinene dimer pathways | 11.58% | 17.75% |
| Original channels plus steps from literature-family-supported proposals | 11.59% | 17.75% |
| Full 70 channels | 10.25% | 18.15% |

In α-pinene/o-cresol run 16, measured suspended mass fell 30.89 µg m⁻³ from its sampled peak to the final observation. The 20-channel model lost only 0.32 µg m⁻³ and the 70-channel model lost 13.72 µg m⁻³ at the 15 s fitting step. Run 24 showed a similar partial recovery of the decline (0.01 and 3.43 µg m⁻³ modeled, 10.23 µg m⁻³ measured). The added channels improved 12 of 15 mass trajectories. [Per-run scores, peaks and final measurements](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/chains/per_run.csv)

![Worst mass and joint errors during the 40 search rounds and after refitting, followed by observed and calculated mass histories for runs 16 and 24.](../../../../images/manchester-soa/product-chains.png)

The unrestricted network gave the smallest worst mass error. For mass and number together, the literature-family-supported network gave 17.75% and remained stable under reaction-order reversal. Fragment retention also mattered. At 15 s the full network’s worst mass error was 10.22% when half the fragmented mass stayed suspended. Retaining one-quarter raised it to 12.74%, and retaining three-quarters gave 11.90%, with rates held fixed and doses reselected. [Phase-split and numerical sensitivities](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/chains/sensitivity.csv) · [Resolution-checked mechanism](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/chains/resolution_checked_mechanism.json)

### Which reactions and products did Particula track?

The precursors have known molecular structures. Downstream, the model tracks how much organic material each precursor contributes to gas and particle pools. For example, two α-pinene-derived particle intermediates can form a dimer pool. That material can oxidize and remain suspended, or fragment and release a daughter into the gas. In a partial fragmentation step the reference model sends half the transferred mass to the gas and retains half in particles. Manchester’s bulk mass and number records leave the molecular structures of these daughters unresolved.

![Skeletal structures of α-pinene, o-cresol and isoprene, a tentative pinic acid assignment from the Manchester campaign, and the fitted α-pinene dimer oxidation and fragmentation pathway.](../../../../images/manchester-soa/reaction-network.svg)

Shao and colleagues tentatively assigned a C₉H₁₄O₄ signal in the α-pinene/isoprene aerosol to pinic acid from its fragmentation pattern. The structure in the figure illustrates that assignment. The model allows one pool to contain many compounds, so it cannot determine whether pinic acid belongs to a particular branch. [Shao et al., Figure 6a and supplement Figure S3](https://acp.copernicus.org/articles/22/9799/2022/#section3)

The same downstream network applies to five families. These comprise dimers from α-pinene, o-cresol or isoprene, the mixed α-pinene/o-cresol product, and material formed by uptake of α-pinene-derived vapor. Each arrow below has a separately fitted rate for each family.

| Starting material | Possible modeled fates |
|---|---|
| Initial particle product (D) | Oxidize to W, cleave to L, split between L and U, or convert to F |
| Oxidized particle product (W) | Cleave to L or split between L and U |
| Retained particle fragment (L) | Oxidize to W or convert to F |
| Released vapor (U) | Form oxidized material that returns to the particle (W), or fragment to terminal volatiles (F) |

The two lists below give all **70 candidate channels** in their archived order. The mass fit and the particle-resolved refit allowed all 70. The bulk mass-and-number fit used in the yield comparison allowed 38. A fitted coefficient can also reach zero. The full inventory gives each channel’s rate law, units, fitted coefficients and inclusion in each calibration. [Annotated reaction inventory](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/REACTION_NETWORK.md) · [70-channel CSV](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/reaction_network.csv)

<details>
<summary>20 channels for precursor-derived intermediates</summary>

The subscripts A, C and I identify α-pinene, o-cresol and isoprene. G is a gas intermediate, V an oxygenated vapor, P an exchangeable particle product, S a stabilized particle product and F terminal volatile material. D labels material routed into one of the five aging families.

| Channel | Effective equation | Proposed process |
|---:|---|---|
| 1 | `G_A + G_C → D_AC` | Gas-phase cross-accretion |
| 2 | `2 P_A → D_AA` | Coupling within the particle |
| 3 | `P_A → F` | Photochemical loss from the particle |
| 4 | `P_I → F` | Photochemical loss from the particle |
| 5 | `V_A → D_AU` | Reactive uptake of oxygenated vapor |
| 6 | `G_C → V_C` | Ring opening to carbonyl vapor |
| 7 | `2 G_C → D_CC` | Gas-phase self-accretion |
| 8 | `2 G_I → D_II` | Gas-phase self-accretion |
| 9 | `V_I → S` | Reactive uptake of oxygenated vapor |
| 10 | `V_A → S` | Gas-phase oxidation to condensable material |
| 11 | `P_C → F` | Photochemical loss from the particle |
| 12 | `V_C → S` | Gas-phase oxidation to condensable material |
| 13 | `2 G_A → D_AA` | Gas-phase self-accretion |
| 14 | `2 P_I → D_II` | Coupling within the particle |
| 15 | `V_C → S` | Reactive uptake of oxygenated vapor |
| 16 | `G_A → V_A` | Formation of carbonyl vapor |
| 17 | `V_A → F` | Gas-phase fragmentation |
| 18 | `G_A + G_I → F` | Mixed termination to volatile products |
| 19 | `G_I → V_I` | Formation of carbonyl vapor |
| 20 | `2 P_C → D_CC` | Coupling within the particle |

</details>

<details>
<summary>50 subsequent transfers across five product families</summary>

AA, AC, AU, CC and II label the five product families. Within each family, D is the initial particle product, W its oxidized particle product, L a retained particle fragment and U a gas-phase daughter. F is a shared terminal volatile pool. The ½ coefficients divide organic mass between phases. Each family has its own fitted rates for the ten transfers.

| Channel | Family | Effective equation | Proposed process |
|---:|---|---|---|
| 21 | α-pinene dimer | `D_AA → W_AA` | Oxidation within the particle |
| 22 | α-pinene/o-cresol cross-product | `D_AC → W_AC` | Oxidation within the particle |
| 23 | α-pinene uptake product | `D_AU → W_AU` | Oxidation within the particle |
| 24 | o-cresol dimer | `D_CC → W_CC` | Oxidation within the particle |
| 25 | isoprene dimer | `D_II → W_II` | Oxidation within the particle |
| 26 | α-pinene dimer | `W_AA → ½ U_AA + ½ L_AA` | Fragmentation of the oxidized product |
| 27 | α-pinene/o-cresol cross-product | `W_AC → ½ U_AC + ½ L_AC` | Fragmentation of the oxidized product |
| 28 | α-pinene uptake product | `W_AU → ½ U_AU + ½ L_AU` | Fragmentation of the oxidized product |
| 29 | o-cresol dimer | `W_CC → ½ U_CC + ½ L_CC` | Fragmentation of the oxidized product |
| 30 | isoprene dimer | `W_II → ½ U_II + ½ L_II` | Fragmentation of the oxidized product |
| 31 | α-pinene dimer | `D_AA → ½ U_AA + ½ L_AA` | Fragmentation with partial evaporation |
| 32 | α-pinene/o-cresol cross-product | `D_AC → ½ U_AC + ½ L_AC` | Fragmentation with partial evaporation |
| 33 | α-pinene uptake product | `D_AU → ½ U_AU + ½ L_AU` | Fragmentation with partial evaporation |
| 34 | o-cresol dimer | `D_CC → ½ U_CC + ½ L_CC` | Fragmentation with partial evaporation |
| 35 | isoprene dimer | `D_II → ½ U_II + ½ L_II` | Fragmentation with partial evaporation |
| 36 | α-pinene dimer | `U_AA → W_AA` | Vapor aging and return to the particle |
| 37 | α-pinene/o-cresol cross-product | `U_AC → W_AC` | Vapor aging and return to the particle |
| 38 | α-pinene uptake product | `U_AU → W_AU` | Vapor aging and return to the particle |
| 39 | o-cresol dimer | `U_CC → W_CC` | Vapor aging and return to the particle |
| 40 | isoprene dimer | `U_II → W_II` | Vapor aging and return to the particle |
| 41 | α-pinene dimer | `D_AA → L_AA` | Cleavage with material retained |
| 42 | α-pinene dimer | `L_AA → F` | Loss of retained fragment to volatiles |
| 43 | α-pinene/o-cresol cross-product | `D_AC → L_AC` | Cleavage with material retained |
| 44 | α-pinene/o-cresol cross-product | `L_AC → F` | Loss of retained fragment to volatiles |
| 45 | α-pinene uptake product | `D_AU → L_AU` | Cleavage with material retained |
| 46 | α-pinene uptake product | `L_AU → F` | Loss of retained fragment to volatiles |
| 47 | o-cresol dimer | `D_CC → L_CC` | Cleavage with material retained |
| 48 | o-cresol dimer | `L_CC → F` | Loss of retained fragment to volatiles |
| 49 | isoprene dimer | `D_II → L_II` | Cleavage with material retained |
| 50 | isoprene dimer | `L_II → F` | Loss of retained fragment to volatiles |
| 51 | α-pinene dimer | `D_AA → F` | Conversion to terminal volatiles |
| 52 | α-pinene/o-cresol cross-product | `D_AC → F` | Conversion to terminal volatiles |
| 53 | α-pinene uptake product | `D_AU → F` | Conversion to terminal volatiles |
| 54 | o-cresol dimer | `D_CC → F` | Conversion to terminal volatiles |
| 55 | isoprene dimer | `D_II → F` | Conversion to terminal volatiles |
| 56 | α-pinene dimer | `U_AA → F` | Fragmentation in the gas phase |
| 57 | α-pinene/o-cresol cross-product | `U_AC → F` | Fragmentation in the gas phase |
| 58 | α-pinene uptake product | `U_AU → F` | Fragmentation in the gas phase |
| 59 | o-cresol dimer | `U_CC → F` | Fragmentation in the gas phase |
| 60 | isoprene dimer | `U_II → F` | Fragmentation in the gas phase |
| 61 | α-pinene dimer | `W_AA → L_AA` | Cleavage of oxidized product within the particle |
| 62 | α-pinene dimer | `L_AA → W_AA` | Further oxidation of retained fragment |
| 63 | α-pinene/o-cresol cross-product | `W_AC → L_AC` | Cleavage of oxidized product within the particle |
| 64 | α-pinene/o-cresol cross-product | `L_AC → W_AC` | Further oxidation of retained fragment |
| 65 | α-pinene uptake product | `W_AU → L_AU` | Cleavage of oxidized product within the particle |
| 66 | α-pinene uptake product | `L_AU → W_AU` | Further oxidation of retained fragment |
| 67 | o-cresol dimer | `W_CC → L_CC` | Cleavage of oxidized product within the particle |
| 68 | o-cresol dimer | `L_CC → W_CC` | Further oxidation of retained fragment |
| 69 | isoprene dimer | `W_II → L_II` | Cleavage of oxidized product within the particle |
| 70 | isoprene dimer | `L_II → W_II` | Further oxidation of retained fragment |

</details>

## The fitted chemistry against mixture yields and water uptake

### Agreement with measured yield changes as the precursors are consumed

We kept the parameters fixed in two bulk calibrations, the 70-channel mass fit and the mass-and-number fit restricted to literature-supported reaction families. We compared each with four digitized points per mixture from Manchester’s Figure 5. For each point, we matched the total precursor mass consumed and compared wall-corrected yields. Separate-parent simulations retained the mixture run’s initial seed conditions and consumption histories to estimate the yield expected from independent precursors. The published points do not specify individual runs or precursor consumption vectors. [Published mixture yields](https://acp.copernicus.org/articles/22/14147/2022/#section3) · [Digitization, additive references and run coverage](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/companions/README.md)

The ternary comparison in the opening figure uses a second way to estimate yield from independent precursors. We applied Voliotis et al.’s published single-precursor fits to the amounts consumed in the run 24 replay. At 461 µg/m³ total consumption, this gives 16.1%. The 11.7% measurement comes directly from the published ternary yield curve. Across its four sampled points, the bulk mass-and-number fit stayed within one percentage point of the first yield and overshot the other three, with the largest gap at 222 µg/m³ consumed (18.8% modeled, 8.0% measured). [Pointwise ternary yields and precursor consumption](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/companions/yield_points.csv)

At about 299 µg m⁻³ consumed in α-pinene/o-cresol, Manchester measured a **27.4%** yield. The mass fit gave **15.1%**, up from 9.8% in separate-precursor replays, and the mass-and-number fit gave **27.5%**, up from 11.3%. Each modeled yield averages the two eligible simulations, runs 16 and 18. The mass-and-number fit overpredicted the three other sampled yields for this mixture. Its mean absolute error over all four was 11.1 percentage points, compared with 7.0 for the mass fit. The two calibrations also differed in available reaction pathways. For the other mixtures, the mass fit’s mean absolute yield errors were 5.0 percentage points for α-pinene/isoprene, 0.6 for o-cresol/isoprene and 5.5 for the ternary mixture. For o-cresol/isoprene, the two calibrations even gave different signs for the interaction, with one fewer observation in reach of the joint fit. [Per-run yields and separate-parent controls](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/companions/yield_points.csv)

![Four panels compare measured wall-corrected yields with both frozen fits at matched total precursor consumption.](../../../../images/manchester-soa/mixture-yields.png)

### Water uptake transferred differently between instruments

With the bulk chemistry and deposition fixed, we fitted organic hygroscopicity (κ) to run 6 (α-pinene) and predicted run 15 (o-cresol/isoprene), then reversed the test. Each prediction used the target run’s pre-light seed κ. Allowing a fitted factor to scale the contribution of organic volume to the sampled particles reduced cross-run cloud-activation κ RMSE from 0.100 to 0.076. For humidified 100 nm particles, cross-run error rose from 0.036 to 0.052. Both chemistry calibrations gave this pattern. Only the post-light water records were held out in each direction, since the bulk chemistry had used both experiments. [Run 6 data](https://doi.org/10.25326/59RG-Y252) · [Run 15 data](https://doi.org/10.25326/E0EP-QR41) · [Water model and reproduction commands](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/companions/README.md) · [Fitted parameters](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/companions/water_parameters.csv) · [Water scores by run and size](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/companions/water_scores.csv)

![Observed hygroscopicity and modeled curves for humidified 100 nm particles and cloud-droplet activation in each of the two experiments.](../../../../images/manchester-soa/water-transfer.png)

## What changes when each particle carries its own composition?

We repeated all 15 experiments with individual particles carrying sulfate and 19 organic inventories. A particle could acquire vapor, undergo aging reactions, merge with a Brownian collision partner or deposit at a rate that depended on its size. A concentration-dependent source could form 3 nm organic particles. The frozen control used the earlier mass-calibrated chemistry and starting particle parameters. The refit used automatic differentiation to propose continuous parameter changes and stochastic CUDA simulations to score them. Both comparisons below use the same particle-resolved model. [Methods and fit bounds](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/METHODS.md)

Wang and colleagues’ five-hour number-size distribution for ternary run 24 supplied 272 usable time–diameter cells. Archived mobility measurements supplied 212 modal diameters in seven runs and 31 surface-area readings in run 18. Modal diameter is the dry diameter at the peak of the particle number distribution per logarithmic diameter interval. We fitted these size measurements alongside suspended organic mass, instrument-visible counts and water uptake. The geometric-mean diameter curve from run 24 was reserved for comparison and comes from the same underlying size scans. [Wang et al., supplementary Figure S1(a)](https://acp.copernicus.org/articles/22/4149/2022/acp-22-4149-2022-supplement.pdf#page=5) · [Size measurements and extraction](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/size_data/README.md)

With 2,048 initial particles per run and 15 s chemistry steps, mean absolute modal-diameter error fell from **46.2 to 19.1 nm**. Run 24 spectrum RMSE fell from 9,682 to 3,899 particles cm⁻³ per decade of diameter. The refit nevertheless ended run 24 at a geometric mean of 203 nm against 223 nm measured (220 nm in the frozen control), and run 18 surface-area error rose from 19.5% to 29.4%. Using size-selected composition to calculate water uptake improved agreement at 100 nm and worsened it at 150 nm. [Comparison and water-uptake controls](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/README.md)

![Run 24 measured and fitted number-size spectra and geometric mean diameter, with size-selected water uptake for runs 6 and 15.](../../../../images/manchester-soa/resolved-comparison.png)

With the other refitted parameters held fixed, removing wall deposition raised the combined score from 0.108 to 0.202. Removing coagulation or nucleation individually gave 0.109, within the variation across numerical replays. Manchester reported that the sulfate seed suppressed observable new-particle formation. [Process comparisons](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/results/checks.csv) · [Voliotis et al., experimental conditions](https://acp.copernicus.org/articles/22/14147/2022/#section2)

### Size and mass respond differently to joint parameter changes

We sampled 48 combinations of all 101 active parameters within bounded ranges around the refit. The combinations varied the 70 reaction rates, precursor conversion and doses, seed width, condensation, wall loss, nucleation and organic κ. We replayed each combination in every experiment with two particle random seeds. For each observable, we first calculated the error between the simulation and the Manchester measurements. The table averages that error over the two seeds. The last column counts how many parameter sets had lower error than the earlier frozen parameters. Frozen and refitted controls used the same numerical settings (768 initial particles per run and 30 s chemistry steps). [Per-observable comparisons and ranges](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/results/perturbations/paired_seed_summary.csv) · [Finer replays](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/results/perturbations/summary.csv)

| Error relative to chamber measurements | Frozen control | Central refit | Median of 48 draws (10th–90th percentile) | Draws improving on frozen |
|---|---:|---:|---:|---:|
| Modal diameter, seven runs (mean absolute error, nm) | 46.2 | 20.5 | 22.1 (17.7–29.6) | 48/48 |
| Number spectrum, run 24 (RMSE, particles cm⁻³ per decade) | 9,840 | 4,029 | 4,151 (3,359–5,352) | 48/48 |
| Mean normalized organic-mass error, 15 runs | 9.5% | 9.2% | 18.2% (13.4–24.2%) | 0/48 |
| Mean normalized particle-count error, 15 runs | 6.9% | 6.7% | 7.3% (6.5–9.0%) | 13/48 |
| Water-uptake error, runs 6 and 15 (κ score) | 0.091 | 0.046 | 0.052 (0.039–0.063) | 48/48 |
| Normalized surface-area error, run 18 | 19.3% | 27.1% | 29.7% (22.7–34.4%) | 1/48 |

The combined score improved in 47 draws. The spectrum, modal diameter and water improvements also persisted in all 12 prespecified finer replays with a new random seed. Restoring one parameter group at a time in 12 matched draws changed mass error most for precursor conversion and the first 20 rates (median absolute changes of 4.7 and 4.0 percentage points). Restoring the wall coefficient changed it by 0.54 points. Seed width had the largest median effect on modal-diameter error, 2.3 nm. These sensitivities apply to the sampled ranges. [Parameter-group comparisons](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/results/perturbations/group_effects.csv) · [Sensitivity design](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/METHODS.md)

![Measured and modeled number spectrum at five hours in run 24, and surface area over five hours in run 18. The measured surface area stays above the shaded band.](../../../../images/manchester-soa/perturbation-comparison.png)

At every one of the 31 run 18 sample times, the measured surface area exceeded the pointwise 90th percentile of the 48 draws for the random seed shown. At the measured peak, the 10th–90th percentile span was 1,298–1,671 µm² cm⁻³ against 2,317 µm² cm⁻³ measured. The shaded band shows the spread induced by the chosen perturbations. The bars on the run 24 spectrum show the allowance for reading the published heatmap. [Run 18 measured and simulated surface area](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/results/perturbations/surface_envelope_nm2_cm3.csv)

## What the simulations add to the Manchester measurements

The separate-precursor replays support Manchester’s finding of enhanced yield for α-pinene mixed with o-cresol. Two fitted mechanisms reproduce the direction of that effect and disagree about its magnitude. The ternary experiment shows what a size measurement adds. Including the measured spectrum in the particle-level fit reduced the error in particle number across diameter bins over five hours, yet the final geometric mean diameter moved farther from the measured value.

The offset in run 18 was already present at lights-on. Recovering its original dry-size scans is the first priority. Integrating particle number and surface area over the same measured diameter window would test how much of the discrepancy arose when we reconstructed the seed distribution, before asking any added reaction to explain its evolution.

For chemistry, the missing observable is the time history of an identified product. In an isolated α-pinene dimer chain we exchanged the oxidation and fragmentation rates of 0.1 and 1 h⁻¹. Predicted dimer inventories differed by up to 94.75 µg m⁻³ across the mass-calibrated runs. Total suspended mass differed by less than 10⁻¹² µg m⁻³, and the particle counts were identical. Measuring a structurally identified dimer and its oxidized daughter through the aerosol peak would test which step was slower under this mechanism. Coincident gas-phase fragments would further constrain how much material left the particles. [Rate-swap results](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/results/chains/rate_swap.csv)

![In an isolated α-pinene dimer chain, reversing the slow and fast rates leaves total suspended mass unchanged but separates the modeled oxidized-dimer concentrations in run 1.](../../../../images/manchester-soa/dimer-rate-swap.png)

The [follow-up network discovery study](../manchester-network-growth/index.md) tests additional reactions against a fresh fit of the earlier chemistry and checks the selected fit against the measured size distributions.

## Reproduce the search and particle-resolved comparisons

The [study-card renderer](../../../../../studies/voliotis-2022/toc.py) redraws the graphical abstract from the archived fits.

The calculations used Particula 0.3.3, Python 3.12 and Warp 1.11.1 on an NVIDIA GeForce RTX 5060. The precursor and aerosol time series are reformatted AERIS/EUROCHAMP records deposited by Gordon McFiggans, Aristeidis Voliotis and the Manchester team under CC BY 4.0. The study includes the source DOIs and checksums, literature evidence for the reactions, selected parameter sets, and scores behind the figures. Replaying the final calibrations requires only the published files. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Instrument data and attribution](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/data/README.md) · [Study archive](https://github.com/uncscode/particula/tree/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022) · [Complete fitting and replay instructions](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/chains/METHODS.md)

From a checkout containing the study, the following command replays the archived bulk calibrations and checks the finer-step scores. The particle-resolved instructions give the separate commands and dependencies for refitting the chemistry, checking particle conservation and reproducing all 222 perturbation and control replays. [Particle-resolved fitting and CUDA replay instructions](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/METHODS.md) · [Saved scores for every perturbation and control](https://github.com/uncscode/particula/blob/7d4fefbfcc593d6e5aafb9d6ecb4e8aa2f4d3645/studies/voliotis-2022/joint/resolved/results/perturbations/scores.csv)

```bash
python -m pip install -e . pandas matplotlib "warp-lang==1.11.1"
export PYTHONPATH="$PWD"
python studies/voliotis-2022/joint/chains/replay_chain.py \
  studies/voliotis-2022/joint/results/chains/resolution_checked_mechanism.json
```
