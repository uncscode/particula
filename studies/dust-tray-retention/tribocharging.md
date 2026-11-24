# Does contact charging change which dust reaches the trays?

Dust grains can acquire charge when they touch and separate. Jia et al. used roughly five minutes of 14 m/s wind to erode dry soil before the deposition phase. Grains could contact each other, the bed and tunnel surfaces during loading and recirculation. The study reports no particle charge distribution, near-tray electric field or charge lifetime. Laboratory experiments with a single granular material found that smaller grains tended to become negative and larger grains positive at low humidity. Experiments dispersing calcite and talc also show that composition and contact with the disperser affect polarity. Neither experiment supplies a charge distribution for Jia et al.’s mixed cropland soil. [Tunnel protocol](https://acp.copernicus.org/articles/26/14245/2026/) · [Particle–particle contact and humidity](https://doi.org/10.1021/acs.iecr.7b02328) · [Fine mineral-powder dispersion](https://doi.org/10.1002/ceat.201900558)

## Charge transfer during loading

The earlier ±5-elementary-charge scenarios prescribed charge at each size. Here we assigned a signed mean charge from hypothetical grain contacts during loading, then tracked its consequences after 30 minutes. We kept the measured early deposit, the bulk-monitor constraint and the twelve-layer transport model common to each comparison. All magnitudes, contact probabilities, lifetimes and fields below are sensitivity assumptions.

In the contact model a grain samples partners in proportion to their available surface area. Each encounter transfers equal and opposite charge, with the smaller partner taking the negative sign. The expected mean charge of a grain of diameter D is πD²σ times the difference between the probabilities of encountering a smaller and a larger partner. The scale σ summarizes the unmeasured number and strength of loading contacts. We used 0, 1 and 10 µC/m². The contact population has zero net charge by construction. A size-selected airborne subset can carry net charge. At 2 m/s and σ = 1 µC/m² the 9.62 µm channel had a modeled mean of about +1,730 elementary charges when contacts followed the reconstructed initial airborne size distribution. This illustrates why a fixed ±5-charge case alone cannot bound contact charging. [Size-resolved charge assumptions (`tribo-source-charge.csv`)](study-results.zip)

A second 2 m/s test used a coarser population of contact partners in the bed, with surface area per log diameter centered at 10 µm, geometric standard deviation 1.8 and an upper cutoff of 63 µm. The two contact populations change which sizes acquire each sign while holding the dust available for the late forecast fixed. Each size has one signed mean in this calculation. Within-size charge dispersion, wall-to-grain charging and renewed charging after 30 minutes remain unconstrained.

## Collisions and electrical drift give different tests

| 2 m/s comparison | Three-hour volume median (µm) |
|---|---:|
| Neutral particles, all three collision kernels | 6.268 |
| Persistent charge, σ = 1 / 10 µC/m², contacts within inferred source | 6.264 / 6.121 |
| Persistent charge, σ = 1 / 10 µC/m², coarser bed contacts | 6.272 / 6.272 |
| Field test with collisions off, zero field | 6.275 |
| Downward / upward 1 kV/m, σ = 10 µC/m², contacts within inferred source | 6.234 / 6.410 |
| Downward / upward 1 kV/m, σ = 10 µC/m², coarser bed contacts | 6.895 / 5.836 |
| Measured tray median | **9.800** |

In the collision runs, opposite signs attract and like signs repel in the Brownian kernel. Aggregates inherit their parents’ net charge. Neutral shear and differential-settling kernels remain active. With contacts within the inferred source, the strongest persistent charge increased the fraction of the initial particle count removed by mergers from 2.9% to 25.2%, including airborne and already collected particles in the denominator. Yet it shifted the volume median by only −0.15 µm. Giving charge a ten-minute exponential lifetime from release yielded 6.268 µm. A 60-minute lifetime yielded 6.202 µm. The measured 10–20 µm volume share was 28.39% at three hours, compared with 20.01% for the strongest persistent collision case. These calculations isolate Coulomb effects on Brownian collisions. Electrostatic focusing of mechanically driven encounters and changes in sticking on contact could give other responses.

In the drift runs a uniform vertical field acts on each grain’s charge through the force qE. Depending on polarity and field direction, it can enhance downward motion, oppose gravity or send a grain toward the ceiling. We solved vertical transport with gravity, this force, wind-dependent mixing and absorbing boundaries, holding charge constant and switching off merging to isolate drift. At σ = 10 µC/m², an upward 1 kV/m field moved the forecast above the zero-field value for the inferred-source contacts and below it for the coarser bed contacts. Even the ±10 kV/m stress test with coarser bed contacts gave only 5.45–7.11 µm. None of these single-mechanism cases recovered 9.80 µm. Self-generated fields, electrostatic adhesion, image forces near surfaces, rebound, and the coupled action of drift and aggregation remain open for this experiment. [All 35 runs (`tribo-cases.csv`)](study-results.zip)

## What measurement would distinguish the mechanisms?

Measure both polarities as a function of airborne particle size after loading and at 30 minutes, together with the direction and strength of the local field above the tray. A bulk charge measurement can hide substantial populations of opposite sign. To test net retention directly, compare matched 2 m/s runs with a neutralizer on and off while checking that the early airborne size and mass distributions and flow are equivalent. A reproducible change in 10–20 µm retention with neutralization would support an electrical contribution. Comparing the arriving flux with what remains on each tray would help locate that effect before or after impact.

## Replay

Run `OPENBLAS_NUM_THREADS=1 "$JIA_PY" tribocharging.py` using the interpreter and checkout in [the replay guide](README.md). The early mechanical reconstruction is common to every run and therefore does not test changes in the original emission or first 30 minutes. Charge leakage is accounted for separately. The dry mass and signed charge inventory checks had relative errors below 3 × 10⁻¹³, normalized by starting mass and absolute charge, respectively. Refining the strongest inferred-source 2 m/s collision case from 12 to 24 layers and halving the time step changed its median by less than 0.001 µm. [Source](tribocharging.py) · [Numerical results (`tribo-cases.csv`)](study-results.zip)
