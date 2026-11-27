# A hybrid neural forecast for one million aerosol particles

<span id="pair-sampling-sets-the-benchmark-for-neural-aerosol-forecasts"></span>

![One simulated second with a million particles and 32 vapors took 35.5 minutes with full native physics and 7.11 minutes with the hybrid. The sampled-physics control took 7.07 minutes.](../../../images/studies/weighted-particle-embedding/forecast-toc.svg)

A forecast coupling a neural collision term to physical vapor transfer took 7.11 minutes to advance one million particles and 32 vapors by one second. Full native physics took 35.5 minutes on the same GPU, a 4.99× speedup for the whole forecast. The hybrid changed both how the solver chose collision partners and how it evaluated their collision rate. Applying the physical rate to the same sampled pairs took 7.07 minutes. That control traced the speed gain to pair selection. We used the same kind of control to test neural rates for vapor transfer and new particle formation.

## How fast was the hybrid forecast at one million particles?

<span id="where-did-the-coagulation-speedup-come-from"></span>

<span id="where-did-the-speedup-come-from"></span>
<span id="full-native-baseline--1-simulated-second"></span>

Brownian motion brings particles together as gas molecules jostle them. To simulate those encounters, the hybrid shuffled the particles and paired adjacent entries. For each candidate pair, a neural network estimated the jump-distance term in the physical collision formula. This term accounts for the distance a particle travels as its Brownian motion changes direction. The solver used that rate and the pair’s probability of selection to draw mergers. Physical rules combined the partners’ masses and charges, and the native vapor solver advanced the particles between collision steps. All one million initially neutral particles remained individually represented in the 1 cm³ box until they merged.

<figure class="scientific-figure">

![Full native physics took 35.5 minutes, the hybrid with a neural Brownian term took 7.11 minutes, sampled physics took 7.07 minutes and omitting mergers took 6.87 minutes.](../../../images/studies/weighted-particle-embedding/native-collisions.png)

<figcaption>Elapsed time includes setup and the entire one-second forecast on an RTX 5060. Each speedup uses full native physics as its baseline. Phase error is the largest observed discrepancy in a scored gas or particle mass relative to time-refined physics. The full native timing summarizes two trials. Each sampled method ran once at this size on a shared seed.</figcaption>
</figure>

<span id="how-accurate-were-the-one-second-collision-runs"></span>

The hybrid met both accuracy targets on its single timed run. Against the time-refined physical reference, its largest particle-count discrepancy was 0.00354% (target 1% or less), and its largest phase-mass discrepancy was 0.07501% (target below 0.1%).

### What did the sampled-physics control show?

To isolate the contribution of the network, we kept the candidate pairs and random draws fixed and switched back to the physical formula for jump distance. The resulting forecast took 7.07 minutes (5.02× faster than full native physics). The physical control and the hybrid each accepted 1,092 mergers and agreed on particle counts and phase masses at every observation. Sampling alone therefore reproduced the speed gain. Whether changing the rate gives the same merger outcomes on fresh draws remains open.

Skipping collisions altogether saved little more time and pushed a gas species past the error limit. Turning mergers off saved another 12.2 seconds relative to sampled physics, but the largest phase-mass discrepancy rose from 0.0750% to 0.1103%, above the 0.1% target. Particle count stayed within its limit. A merger changes the surface available for subsequent vapor exchange, so the error can appear in the gas before it becomes apparent in the count. [Collision timers and the limiting gas species](coagulation.md)

Shima and colleagues used random pairs of weighted computational droplets to simulate cloud coalescence. Here sampling reduced the number of pairs evaluated while preserving a separate record for each particle. Weighting the records to represent several particles is another way to reduce cost, with its own accuracy tests. [Shima et al., the super-droplet method](https://doi.org/10.1002/qj.441) · [Weighted and explicit particles at scale](scaling.md)

### How does the speedup vary by coagulation kernel?

A coagulation kernel gives the rate at which two particles collide. Charge can change that rate through attraction or repulsion, and particles that settle at different speeds can overtake one another. We kept the pair sampler and changed the kernel to see whether the saving carried over. Drawing disjoint pairs saved the most time for charged Brownian collisions. At 32,768 particles the full native forecast took about 22 seconds with charged Brownian motion and about one second with differential settling. Sampling pairs with physical rates brought both forecasts to about 0.4 seconds. That difference in baseline cost explains why the same approach yielded a 54.28× speedup for charged Brownian motion and 2.62× for settling. Switching the sampled rates to neural estimates had little effect on either time.

<figure class="scientific-figure">

![Speedup for five coagulation kernels in boxes of 4,096 and 32,768 particles. Physical and neural rates on sampled pairs gave similar gains, with the largest gain for charged Brownian motion.](../../../images/studies/weighted-particle-embedding/kernel-forecast-times.png)

<figcaption>Whole-forecast speedup relative to the full native physics for each kernel. Circles use physical rates on sampled pairs and squares use neural rates. Each marker gives the median across eight seeds. Lines span the middle half of the speedups, and the dashed line at 1× marks equal time. Both panels cover one simulated second with 32 vapors on the RTX 5060. This comparison used four vapor substeps per half interval and started timing after warm-up. The million-particle comparison above used 128 substeps and included setup.</figcaption>
</figure>

At 32,768 particles both sampled methods stayed within the count and phase-mass targets across the eight seeds when compared with native physics on the same time step. In the smaller shear box, even two native runs could differ by more than the mass target when their random draws differed. [Absolute timings, forecast errors and the shear repeatability check](coagulation.md#how-do-the-alternatives-time-against-full-physics-for-each-kernel)

The same shuffling rule worked across the five kernels because the collision law enters after candidate selection. The fitted rate had a narrower range. Applying the neutral Brownian network unchanged to settling or shear gave large pair-rate errors. [Testing the rate network on another kernel](coagulation.md#what-happens-if-we-reuse-the-neutral-brownian-network-unchanged)

## What happened when we replaced other process rates?

A separate ten-second experiment started from a physical solver that already sampled collision pairs. The million-particle box contained 32 vapors, with 16 initially favoring condensation, eight favoring evaporation and eight near equilibrium. A vapor can add mass to existing particles through condensation. A precursor can also form new particles through nucleation. For condensation, the network estimated a factor controlling how quickly mass crossed each particle surface. For nucleation, it estimated the rate at which new particles formed. Each run changed just one rate. The times below are whole-forecast means over three matched seeds after setup and warm-up. This experiment used a different vapor solver and time step from the one-second native benchmark above.

| Rate changed in the ten-second forecast | Whole-forecast time |
|---|---:|
| Physical rates throughout, with sampled collisions | 8.370 s |
| Neural jump distance for sampled collisions | 8.699 s |
| Neural vapor-transfer factor | 276.1 s |
| Neural birth rate | 8.375 s |

Evaluating a neural transfer factor for each particle–vapor combination made the forecast take 33 times as long. The largest phase-mass difference from the physical run at the same step was about 0.00133%. Halving the physical vapor step changed phase mass by up to 0.636%, so agreement at the original step left a larger integration error unresolved. [Process timings, errors and box-size comparison](process-comparisons.md)

Fitting one coefficient in the birth-rate law reproduced the physical birth count on all three seeds and left the total time effectively tied. Physical nucleation took only 0.141 seconds of the original forecast. Survival after birth poses a different question. For that test we followed newborn particles through growth and scavenging and counted how many reached 5 and 10 nm. [Newborn growth and survival](formation.md)

### Where does a condensation step spend its time?

<span id="why-was-the-condensation-speedup-smaller"></span>

Consider one vapor shared by many particles. If its gas concentration exceeds the equilibrium concentration at a particle’s surface, it condenses there. If the gas concentration is lower, material evaporates from that particle. The concentration difference drives the exchange. A transfer coefficient, called *conductance*, sets how quickly it proceeds. It depends on particle size, how rapidly the vapor diffuses through the gas and how readily its molecules enter the particle on contact.

The 128-unit network replaced one part of the conductance formula. From the vapor’s mean free path relative to the particle radius and the probability that a molecule enters on contact, it estimated a dimensionless transfer factor. The solver combined that factor with the radius and vapor diffusivity to get the conductance. The solver still had to determine how much of that vapor remained in the common gas reservoir and update the mass on every particle. What one particle takes up leaves less vapor available for the others. The solver balances those transfers so each species keeps its total mass.

We timed these operations separately. In each 0.125-second model interval, vapor exchange ran for half the interval before collisions and new particle formation, then for the other half afterward. Across the ten-second physical forecast, vapor exchange took 87.4% of the runtime. Within a separately timed vapor half step, calculating conductance took 16.5 of 46.0 milliseconds (36%). Even an instantaneous conductance estimate would leave the gas balance and particle updates to do.

The measured speedup came from combining physical operations on the GPU. One pass accumulated the contributions needed to solve for each vapor’s gas concentration. A second applied that solution to the particles, avoiding large intermediate arrays. This cut the forecast from 8.37 to 5.17 seconds (1.62×). It reproduced the same coarse time step, including its 1.23% phase-mass discrepancy from finer integration.

We also tried learning a correction to the time step itself. Starting from the same masses, we advanced the vapor once over a half interval and again in two smaller steps. Their difference estimated the integration error. We used either a fixed weight or a single fitted weight to correct the result from the two smaller steps. Both forecasts took about 15.2 seconds and brought the phase discrepancy below the 0.1% target. The two comparisons ranked the corrections differently. Against 16 physical substeps across three seeds, the worst discrepancy was smaller for the fixed correction (0.0551% versus 0.0705%). Against 64 substeps on one seed, it was smaller for the learned correction (0.0325% versus 0.0525%). A test on an unfamiliar mixture and pulse history would show whether the learned correction offers a repeatable advantage. [Condensation formulas and time-step checks](condensation-options.md)

## Which details of an aerosol are worth carrying into a forecast?

<span id="what-does-the-network-see-and-what-does-physics-still-do"></span>

An inexpensive forecast must still distinguish particles that will respond differently to the next vapor pulse. Zaveri and colleagues illustrated the issue in an aging urban plume. Particles of the same dry diameter carried different mixtures and activated as cloud droplets at different supersaturations. A size bin can preserve the total mass of each species while losing which materials share a particle. [Zaveri et al., particle-resolved urban plume](https://doi.org/10.1029/2009JD013616)

We tested this distinction with two synthetic aerosols that had identical dry size spectra and bulk composition. One distributed its water-attracting material evenly within each size class. The other concentrated that material on a subset of particles. A learned *embedding* summarized each particle and combined the summaries using particle counts as weights. With eight coordinates, it retained the difference in water uptake through 47.5 ms on the held-out test families and 50 ms on the challenge families. Dry size alone gave no qualifying lead. A physical reconstruction that retained composition within each size class recovered the difference through the final observation at 80 ms. [Matched mixtures and water-uptake forecasts](representation.md)

These experiments connect two approaches to learning aerosol dynamics. Wang and colleagues learned how a size distribution evolves under Brownian coagulation using a model symmetric in the two collision partners. Harder and colleagues learned one-step changes to a seven-mode aerosol population and tested species mass balance. Our collision benchmark asks whether a learned interaction can accelerate a coupled forecast. The paired vapor pulse asks whether a compact population record still contains the information needed for that forecast. [Wang et al., learning coagulation](https://doi.org/10.1029/2022MS003252) · [Harder et al., physics-informed aerosol microphysics](https://doi.org/10.1017/eds.2022.22)

## Which neural experiment should come next?

A chamber-model ensemble needs many forecasts at different humidities, mixtures and vapor supplies. A particle-resolved calculation can generate training trajectories for those forecasts, including the individual particles that grow or disappear. We propose testing three uses in sequence. Each has a physical control and an observable that can expose a failure.

| Candidate and intended use | Matched comparison | What would justify using the network? |
|---|---|---|
| Estimate a charge-dependent encounter rate in a population gaining and losing ions | Keep the candidate pairs, ion clock and random draws paired. Change only the physical or learned rate. Repeat on fresh seeds and reserve charge distributions outside training. | Recover merger counts and the size and charge distributions of survivors. A faster rate must also lower total forecast time after sampling and vapor transfer. |
| Correct vapor transfer over one interval in repeated humidity or organic-vapor pulses | Freeze the trained correction. Hold out temperatures, vapor accommodation coefficients and pulse schedules. Compare with the fixed correction and a time-refined physical solve on the same initial particles. | Smaller worst gas- or particle-phase error at equal runtime, or a faster whole forecast at the same accuracy, across the held-out conditions. |
| Predict how many newly formed particles survive to 5 and 10 nm | Keep precursor debit and growth physics explicit. Compare a learned survival estimate with tracked newborn cohorts at unseen background concentrations and ion supplies. | At each diameter, bound the error in arrivals per 1,000 births across seeds and compare it with the uncertainty from refining the physical reference. This tests particles born during the forecast, including those lost before detection. |

<span id="what-counts-as-an-accurate-forecast"></span>

An accurate forecast has to put each species in the right phase as well as conserve its total mass. At every observation, the particle count must agree with the physical reference to within 1%, and each scored phase mass must differ by less than 0.1%. For reference concentrations at or below 10⁻¹¹ kg m⁻³, the absolute tolerance is 10⁻¹⁴ kg m⁻³. We also check the combined gas and particle inventory against the initial amount plus any imposed vapor pulse. [Physical corrections under changing forcing](process-clocks.md) · [Newborn survival calculations](formation.md)

An experimental test would ask a different question. Laboratory measurements on ammonium sulfate–succinic acid particles found size-dependent water uptake and linked it to how the materials were distributed inside each particle. Our paired boxes vary composition across particles of the same size. For a chamber comparison, we would select dry particles by size, record the humidity history, and measure the distribution of wet diameters through the pulse. If both the learned and physical forecasts reproduce bulk water but miss which particles grow, the next candidate input is the particles’ internal phase structure. [Laboratory measurements of water uptake and particle morphology](https://doi.org/10.1021/jp510268p)

## Explore the experiments

<span id="sampled-pair-physical-baseline--10-simulated-seconds"></span>
<span id="how-do-speed-and-accuracy-change-at-one-million-particles"></span>
<span id="did-the-vapors-actually-condense-and-evaporate"></span>
<span id="what-does-each-process-cost"></span>

- [Coagulation across kernels](coagulation.md#which-parts-transfer-to-another-collision-kernel) compares Brownian, charged Brownian, settling, shear and combined rates against each kernel’s full native timing and forecast.
- [GPU scaling](scaling.md) varies particle count, number of vapors and number of weighted representatives.
- [Process timings and errors](process-comparisons.md) compares one neural substitution at a time and calculates how much whole-forecast time each could save.
- [Condensation](condensation-options.md) separates the transfer formula, gas balance and time-step error.
- [Process clocks](process-clocks.md) tests when to update vapor, collisions and charge, and when to insert a full physical step.
- [New particle formation](formation.md) follows each newborn cohort through growth and loss.
- [Particle representation](representation.md) compares matched mixtures after the same pulse.

[Recreate the simulations and figures](../../../../studies/weighted-particle-embedding/README.md) · [Download the quantitative figure data, plotting code and source checksums (ZIP)](../../../../website/public/study-data/weighted-particle-embedding/figure-data.zip)
