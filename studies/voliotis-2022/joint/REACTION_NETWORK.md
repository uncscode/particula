# The 70 chemical channels fitted to the Manchester chamber experiments

Read the equations as transfers between effective organic inventories. A parent label records the source of carbon. Several products can share an inventory and its fitted rate. The original 20-channel search used prospective product labels such as `S_AC`. The full mechanism routes those products to the actual destinations shown here, including `D_AC` for the mixed accretion product. The CSV retains both labels, model indices, kinetic units, evidence links and unrounded coefficients. [Machine-readable reaction inventory](reaction_network.csv)

## Species and kinetic conventions

**Parent pools.** A, C and I designate α-pinene, o-cresol and isoprene. Each parent supplies a delayed gas intermediate G, an oxygenated vapor V and an exchangeable particle pool P. S receives stable unlabelled particle product and F collects terminal volatile material. Measured parent disappearance drives the effective G and V sources. The fitted parent-specific delay converts G to P, and a fixed background rate of 1 h⁻¹ sends V and the secondary vapors U to F. These source and background terms are additional to the 70 candidate channels. Their rates, assignments and reaction order are part of the model specification. [Source terms](discovery/METHODS.md) · [CUDA chemistry](discovery/gpu.py)

**Tracked families.** AA and CC receive both gas- and particle-formed homodimers from α-pinene and o-cresol, respectively. II receives both routes from isoprene. AC receives the gas cross-accretion product of α-pinene and o-cresol, and AU receives the particle product of α-pinene vapor uptake. Each family has its own original particle product D, oxidized particle product W, retained particle residue L and released vapor U. These names describe lumped product classes. The two terms with a coefficient of ½ split transferred *organic mass* equally between particle residue and vapor. Oxygen uptake, detailed molecular stoichiometry and individual isomers are unresolved. [Pool assignments and branching](chains/METHODS.md) · [Canonical channel bank](chains/bank.py)

**How to read the rates.** In the tables, `1` means first order, with mass transfer per air volume per hour equal to k times the source mass concentration. `A` means first order weighted by the ratio of the modeled particle surface area per air volume to 0.001 m² m⁻³. Both coefficients have units h⁻¹. `2` means bimolecular, with k in m³ air µmol⁻¹ h⁻¹. For unlike reactants the event rate is k nₐ nᵦ, and for identical reactants it is k nₐ² with two reactant equivalents consumed per event. Here n is the mass concentration divided by the parent’s surrogate molar mass (184, 140 and 118 g mol⁻¹ for A, C and I). The kernel transfers the mass of both reactants to the destination. In the particle-resolved model the bimolecular particle rates use local composition, evaluating the channel on each particle’s contribution per air volume with k divided by its share of total dry volume. Area-weighted uptake uses the total instantaneous particle area. [Bulk finite-step rate laws](discovery/gpu.py) · [Resolved gas and particle kernels](resolved/kernels.py)

The first-order finite-step fraction is 1 − exp(−kₑ Δt), where kₑ includes any area weighting and Δt is in hours. For a self-reaction the consumed fraction is z/(1 + z), with z = 2 kₑ Δt nₐ. For unlike partners the common molar extent is kₑ Δt nₐ nᵦ / [1 + kₑ Δt max(nₐ,nᵦ)]. Gas-to-particle destinations join the suspended inventories. The resolved model distributes incoming particle material according to the fitted surface-area/volume allocation. Released vapors re-enter the shared gas calculation at the next split step. [Particle-resolved methods](resolved/METHODS.md)

**Calibrations.** “Mass” gives the full-network bulk mass-only coefficient. “Joint” gives the restricted bulk mass-and-number coefficient. “Resolved” gives the full particle-resolved multi-observable coefficient. The first and third allowed all 70 rates to vary. The restricted fit admitted the first 20 plus 18 downstream steps belonging to at least one family-supported motif. “Excluded” means the channel was outside that mask. An exact 0 indicates an admitted coefficient fitted to zero. Small positive values, including values near numerical zero, retain their sign. Tables round to four significant digits. Parameter values express best fits under the stated bounds and conditions. [Bulk fit vectors and active indices](results/chains/resolution_checked_mechanism.json) · [Restricted mask construction](chains/compare.py) · [Particle-resolved fit vector and channel keys](resolved/results/selected.json)

## First 20 channels in reaction-search order

The literature ledger records each proposed substrate and product label beside its experimental analogue and the assumptions required to use it here. “Direct” and “reaction-family” grades describe the published process comparison. The fitted rates remain conditional on the Manchester model. [Primary evidence for the first 20 proposals](results/discovery/REACTION_EVIDENCE.md)

| # | Model key | Pool transfer | Phase | Process | Law | Mass | Joint | Resolved |
|---:|---|---|---|---|:---:|---:|---:|---:|
| 1 | `ac_gas` | `G_A + G_C → D_AC` | gas → particle | Cross-accretion | 2 | 0.4969 | 1.275e-16 | 0.5869 |
| 2 | `aa_particle` | `2 P_A → D_AA` | particle → particle | Particle coupling | 2 | 0.08088 | 0.06103 | 0.06749 |
| 3 | `a_photo` | `P_A → F` | particle → gas | Photochemical loss | 1 | 5 | 0.7697 | 4.615 |
| 4 | `i_photo` | `P_I → F` | particle → gas | Photochemical loss | 1 | 4.424 | 4.907 | 4.798 |
| 5 | `a_uptake` | `V_A → D_AU` | gas → particle | Reactive uptake | A | 0.2045 | 1.292 | 0.2431 |
| 6 | `c_ringopen` | `G_C → V_C` | gas → gas | Ring opening | 1 | 1.129e-17 | 0.05315 | 0.03895 |
| 7 | `cc_gas` | `2 G_C → D_CC` | gas → particle | Self-accretion | 2 | 3.704e-16 | 0.01045 | 0.01084 |
| 8 | `ii_gas` | `2 G_I → D_II` | gas → particle | Self-accretion | 2 | 7.251e-18 | 0.06324 | 0.1396 |
| 9 | `i_uptake` | `V_I → S` | gas → particle | Reactive uptake | A | 0.02561 | 0 | 0 |
| 10 | `a_oxid` | `V_A → S` | gas → particle | Oxidation to condensable product | 1 | 2.032e-17 | 0.005215 | 0.03622 |
| 11 | `c_photo` | `P_C → F` | particle → gas | Photochemical loss | 1 | 0 | 0.6621 | 0.08414 |
| 12 | `c_oxid` | `V_C → S` | gas → particle | Oxidation to condensable product | 1 | 2.114e-17 | 0.1841 | 4.145e-05 |
| 13 | `aa_gas` | `2 G_A → D_AA` | gas → particle | Self-accretion | 2 | 0.1454 | 0.2087 | 0.2045 |
| 14 | `ii_particle` | `2 P_I → D_II` | particle → particle | Particle coupling | 2 | 0 | 0.0001446 | 0.03547 |
| 15 | `c_uptake` | `V_C → S` | gas → particle | Reactive uptake | A | 0.1024 | 0.08474 | 0.1324 |
| 16 | `a_carbonyl` | `G_A → V_A` | gas → gas | Carbonyl formation | 1 | 1.245 | 0.03211 | 1.202 |
| 17 | `a_gasfrag` | `V_A → F` | gas → gas | Vapor fragmentation | 1 | 0.2828 | 4.372 | 0.2472 |
| 18 | `ai_term` | `G_A + G_I → F` | gas → gas | Cross-termination | 2 | 0.3163 | 0.0005579 | 0.3831 |
| 19 | `i_carbonyl` | `G_I → V_I` | gas → gas | Carbonyl formation | 1 | 2.697 | 0.003717 | 1.991 |
| 20 | `cc_particle` | `2 P_C → D_CC` | particle → particle | Particle coupling | 2 | 3.756 | 0.01729 | 3.751 |

## Channels 21 through 70 in bank order

The agent tested eight motifs for each of the five families. Motifs can share a step or introduce two at once, giving ten unique transfers per family over 40 trials. For example, AA oxidation enters both the retained-oxidation trial and the oxidation-then-fragmentation trial. The evidence grade applies to the whole proposed sequence. The CSV lists every motif containing a channel, with its grade in the same order as its round number. [All 40 trials, equations and evidence grades](chains/evidence.csv) · [Scope and primary-source limitations](chains/EVIDENCE.md)

| # | Model key | Pool transfer | Phase | Process | Law | Mass | Joint | Resolved |
|---:|---|---|---|---|:---:|---:|---:|---:|
| 21 | `AA_oxidize` | `D_AA → W_AA` | particle → particle | Retained oxidation | 1 | 0.2057 | 0.09567 | 0.2465 |
| 22 | `AC_oxidize` | `D_AC → W_AC` | particle → particle | Retained oxidation | 1 | 0.02794 | Excluded | 0.02171 |
| 23 | `AU_oxidize` | `D_AU → W_AU` | particle → particle | Retained oxidation | 1 | 0.07486 | Excluded | 0.09561 |
| 24 | `CC_oxidize` | `D_CC → W_CC` | particle → particle | Retained oxidation | 1 | 0.04663 | 2.953e-17 | 0.02124 |
| 25 | `II_oxidize` | `D_II → W_II` | particle → particle | Retained oxidation | 1 | 0.09966 | 3.406e-13 | 0.08098 |
| 26 | `AA_oxsplit` | `W_AA → ½ U_AA + ½ L_AA` | particle → gas + particle | Oxidized-product fragmentation | 1 | 1.626 | 0.1031 | 1.487 |
| 27 | `AC_oxsplit` | `W_AC → ½ U_AC + ½ L_AC` | particle → gas + particle | Oxidized-product fragmentation | 1 | 0.2181 | Excluded | 0.3501 |
| 28 | `AU_oxsplit` | `W_AU → ½ U_AU + ½ L_AU` | particle → gas + particle | Oxidized-product fragmentation | 1 | 1.093 | Excluded | 0.6785 |
| 29 | `CC_oxsplit` | `W_CC → ½ U_CC + ½ L_CC` | particle → gas + particle | Oxidized-product fragmentation | 1 | 0.1219 | 0 | 0.1056 |
| 30 | `II_oxsplit` | `W_II → ½ U_II + ½ L_II` | particle → gas + particle | Oxidized-product fragmentation | 1 | 0.1002 | 0 | 0.2325 |
| 31 | `AA_split` | `D_AA → ½ U_AA + ½ L_AA` | particle → gas + particle | Original-product fragmentation | 1 | 1.149 | 0.1038 | 0.8886 |
| 32 | `AC_split` | `D_AC → ½ U_AC + ½ L_AC` | particle → gas + particle | Original-product fragmentation | 1 | 0.1133 | Excluded | 0.1719 |
| 33 | `AU_split` | `D_AU → ½ U_AU + ½ L_AU` | particle → gas + particle | Original-product fragmentation | 1 | 0.03081 | Excluded | 0.06367 |
| 34 | `CC_split` | `D_CC → ½ U_CC + ½ L_CC` | particle → gas + particle | Original-product fragmentation | 1 | 0.2727 | 0.0002587 | 0.2975 |
| 35 | `II_split` | `D_II → ½ U_II + ½ L_II` | particle → gas + particle | Original-product fragmentation | 1 | 0.1015 | 1.366e-05 | 0.2347 |
| 36 | `AA_gasage` | `U_AA → W_AA` | gas → particle | Vapor aging and condensation | 1 | 0.08339 | 0.09888 | 0.1687 |
| 37 | `AC_gasage` | `U_AC → W_AC` | gas → particle | Vapor aging and condensation | 1 | 0.07469 | Excluded | 0 |
| 38 | `AU_gasage` | `U_AU → W_AU` | gas → particle | Vapor aging and condensation | 1 | 0.002538 | Excluded | 0.2364 |
| 39 | `CC_gasage` | `U_CC → W_CC` | gas → particle | Vapor aging and condensation | 1 | 0.08648 | Excluded | 0.09683 |
| 40 | `II_gasage` | `U_II → W_II` | gas → particle | Vapor aging and condensation | 1 | 0.09988 | Excluded | 0.2099 |
| 41 | `AA_cleave` | `D_AA → L_AA` | particle → particle | Retained cleavage | 1 | 1.045 | 0.09864 | 1.219 |
| 42 | `AA_resloss` | `L_AA → F` | particle → gas | Residue volatilization | 1 | 0.2238 | 0.1089 | 0.2232 |
| 43 | `AC_cleave` | `D_AC → L_AC` | particle → particle | Retained cleavage | 1 | 0.2047 | Excluded | 0.2205 |
| 44 | `AC_resloss` | `L_AC → F` | particle → gas | Residue volatilization | 1 | 0.773 | Excluded | 1.269 |
| 45 | `AU_cleave` | `D_AU → L_AU` | particle → particle | Retained cleavage | 1 | 0.6235 | Excluded | 0.8003 |
| 46 | `AU_resloss` | `L_AU → F` | particle → gas | Residue volatilization | 1 | 1.101 | Excluded | 1.312 |
| 47 | `CC_cleave` | `D_CC → L_CC` | particle → particle | Retained cleavage | 1 | 0.06172 | 0 | 0.05099 |
| 48 | `CC_resloss` | `L_CC → F` | particle → gas | Residue volatilization | 1 | 0.186 | 7.672e-08 | 0.1921 |
| 49 | `II_cleave` | `D_II → L_II` | particle → particle | Retained cleavage | 1 | 0.09985 | 3.876e-13 | 0.04358 |
| 50 | `II_resloss` | `L_II → F` | particle → gas | Residue volatilization | 1 | 0.1006 | 6.942e-14 | 0.1607 |
| 51 | `AA_volatile` | `D_AA → F` | particle → gas | Extensive fragmentation | 1 | 3.296 | 0.1112 | 2.996 |
| 52 | `AC_volatile` | `D_AC → F` | particle → gas | Extensive fragmentation | 1 | 0.03571 | Excluded | 0.02897 |
| 53 | `AU_volatile` | `D_AU → F` | particle → gas | Extensive fragmentation | 1 | 0.0001921 | Excluded | 0.1279 |
| 54 | `CC_volatile` | `D_CC → F` | particle → gas | Extensive fragmentation | 1 | 3.858 | Excluded | 3.217 |
| 55 | `II_volatile` | `D_II → F` | particle → gas | Extensive fragmentation | 1 | 0.1034 | Excluded | 0.1434 |
| 56 | `AA_gasfrag` | `U_AA → F` | gas → gas | Vapor fragmentation | 1 | 0.1127 | 0.1001 | 0.07743 |
| 57 | `AC_gasfrag` | `U_AC → F` | gas → gas | Vapor fragmentation | 1 | 0.1032 | Excluded | 0.113 |
| 58 | `AU_gasfrag` | `U_AU → F` | gas → gas | Vapor fragmentation | 1 | 0.1126 | Excluded | 0.0904 |
| 59 | `CC_gasfrag` | `U_CC → F` | gas → gas | Vapor fragmentation | 1 | 0.101 | Excluded | 0.0854 |
| 60 | `II_gasfrag` | `U_II → F` | gas → gas | Vapor fragmentation | 1 | 0.1 | Excluded | 0.2241 |
| 61 | `AA_oxcleave` | `W_AA → L_AA` | particle → particle | Oxidized-product cleavage | 1 | 0.1049 | Excluded | 0.0003714 |
| 62 | `AA_resoxid` | `L_AA → W_AA` | particle → particle | Residue oxidation | 1 | 0.05673 | Excluded | 0.07517 |
| 63 | `AC_oxcleave` | `W_AC → L_AC` | particle → particle | Oxidized-product cleavage | 1 | 0.1303 | Excluded | 0.1713 |
| 64 | `AC_resoxid` | `L_AC → W_AC` | particle → particle | Residue oxidation | 1 | 0.05222 | Excluded | 0.04976 |
| 65 | `AU_oxcleave` | `W_AU → L_AU` | particle → particle | Oxidized-product cleavage | 1 | 0.2245 | Excluded | 0.2004 |
| 66 | `AU_resoxid` | `L_AU → W_AU` | particle → particle | Residue oxidation | 1 | 0.001517 | Excluded | 0.1269 |
| 67 | `CC_oxcleave` | `W_CC → L_CC` | particle → particle | Oxidized-product cleavage | 1 | 0.1026 | Excluded | 0.1207 |
| 68 | `CC_resoxid` | `L_CC → W_CC` | particle → particle | Residue oxidation | 1 | 0.09508 | Excluded | 0.06818 |
| 69 | `II_oxcleave` | `W_II → L_II` | particle → particle | Oxidized-product cleavage | 1 | 0.1 | Excluded | 0.1247 |
| 70 | `II_resoxid` | `L_II → W_II` | particle → particle | Residue oxidation | 1 | 0.09997 | Excluded | 0.1218 |
