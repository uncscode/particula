# How the regional chemistry bounds the inferred salt allocation

The summit scans measured what fraction of particles activated. To translate those fractions into amounts of salt, the fit also needs a chemical budget and a model of what happens to each ingredient as the droplet takes up water. The daily chemical measurements came from Matorova, about 6 km from the summit and 220 m lower. They constrain the quantities available in a candidate population while leaving the chemistry at a particular summit diameter open. [Matorova ion measurements](https://doi.org/10.48597/J4SV-VVBN) · [Matorova organic and elemental carbon](https://doi.org/10.48597/CAMS-V8ZB)

## What do the filter samples measure?

Among eight measured ions, sodium and chloride together accounted for 58% of ion mass on 11 November and 10% on 20 November. Both dates had complete coverage. For the fit, sodium chloride and ammonium sulfate equivalents were capped by the less abundant stoichiometric partner in each pair. Organic carbon supplied a second constraint through each surrogate’s own carbon content and dry density. A salt equivalent describes an amount those ions could supply. Identifying the compounds or their distribution among particles requires more information.

As a separate bulk check on the ten count-comparison dates, we converted organic carbon to organic matter using a mass ratio of 2.1 and assumed densities of 1.20 and 1.77 g cm⁻³ for organic matter and ions. Under those assumptions the available ions averaged 44% of reconstructed ion-plus-organic dry volume, ranging from 32% to 57% across dates. Only four dates had complete ion coverage. The ion and carbon inlets also sampled different size ranges. Comparing 44% directly to the summit's 49.6% at 120 nm would mix a regional bulk estimate with a size-specific fitted quantity. [Organic-matter conversion](https://doi.org/10.1080/02786820119445) · [Density assumptions](https://doi.org/10.1029/2007JD009241)

## Which features survive changing the ingredients?

Sorbitol represents a water-soluble organic and squalane a hydrocarbon-rich oil. For each dry composition, AIOMFAC supplies water activity in the aqueous phase. The calculation limits dissolved squalane to its equilibrium abundance with a pure-oil reservoir, includes its dry volume in droplet size, and assumes a water-rich surface at activation. A full multicomponent liquid–liquid equilibrium remained unverified. For a binary solution containing 64.57% sorbitol by mass, calculated water activity was 0.756 against a measured 0.803. At 9.93% sorbitol the corresponding values were 0.989 and 0.992. [Sorbitol–water measurements](https://doi.org/10.1021/je990303c)

| Change in the daily calibration | Mean activation RMSE (percentage points) | Mean bulk CCN absolute error (cm⁻³) |
| --- | ---: | ---: |
| Sorbitol and squalane, fitted size allocation | 8.52 | 18.87 |
| Sorbitol as the only organic | 11.49 | 17.86 |
| Squalane as the only organic | 11.45 | 21.52 |
| Same class abundances at every size | 14.31 | 20.58 |
| Pure salt and pure organic classes, allocated by size | 9.22 | 19.13 |
| Glycerol in place of sorbitol | 8.11 | 18.97 |

Substituting glycerol shifted the mean fitted salt fractions to 5.8% at 50 nm and 48.0% at 120 nm. Changing the regional organic-carbon budget by ±25% moved the hydrophilic share of organic volume to 59–66%. Each carbon sensitivity retained greater mean salt volume fraction at 120 nm. The enrichment therefore survived these chemical choices, while the identity and precise proportions of the organic material remained uncertain. [Sorbitol and salt water activities](https://doi.org/10.1021/je0003187) · [Squalane phase behavior](https://www.nature.com/articles/micronano201787)

## How much of the spread comes from numerical resolution?

The original run used an NVIDIA GeForce RTX 5060 for 351 model–date cases and 135 omitted-setting refits. Doubling the water grid changed critical diameter by at most 0.008 nm in the reference blend. Refining the ternary composition grid from 15 to 45 classes moved mean χ from 0.345 to 0.382 and changed the mean 120 nm salt fraction by less than 0.1 percentage point. Across 32 independent draws of 10,000 particles per date, the mean RMS sampling discrepancy in activated fraction was 0.52 percentage points. The choice of organic surrogate and which diameters were measured had larger consequences for the interpretation. [Scientific model and replay](https://github.com/uncscode/particula/tree/main/studies/pallas-activation)
