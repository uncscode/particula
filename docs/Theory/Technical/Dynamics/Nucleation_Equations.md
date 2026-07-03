# Nucleation Discussion

Nucleation, or new particle formation (NPF), is the gas-to-particle conversion process that creates new aerosol particles directly from vapor molecules. Unlike condensation, which grows existing particles, nucleation generates new particles by assembling molecular clusters that become thermodynamically stable once they exceed a critical size. Nucleation controls aerosol number concentrations in many environments, seeds the growth that produces cloud condensation nuclei, and is the starting point of the NPF-to-cloud-droplet size range.

This follows Chapter 11 of Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.

## Homogeneous Nucleation (Classical Nucleation Theory)

Classical nucleation theory (CNT) describes single-species (homomolecular) homogeneous nucleation: clusters of a vapor form spontaneously in a supersaturated gas without pre-existing surfaces. The competition is between the free-energy cost of creating new surface and the free-energy gain of moving molecules from a supersaturated vapor into the condensed phase.

### Saturation Ratio

**Equation 1: Saturation Ratio**

S = pᵢ / pᵢ^sat

**Where:**

- **S**: Saturation ratio (dimensionless).
- **pᵢ**: Partial pressure of the nucleating species **i** in the gas phase.
- **pᵢ^sat**: Saturation vapor pressure of pure species **i** over a flat surface.

**Description:**

Nucleation requires supersaturation (**S > 1**). The larger the saturation ratio, the smaller the critical cluster and the faster the nucleation rate. For **S ≤ 1**, cluster formation is always uphill in free energy and no nucleation occurs.

### Gibbs Free Energy of Cluster Formation

**Equation 2: Free Energy of an r-Sized Cluster**

ΔG(r) = 4 × π × r² × σ − (4 × π × r³ / (3 × v₁)) × k_B × T × ln S

**Where:**

- **ΔG(r)**: Gibbs free energy change to form a spherical cluster of radius **r**.
- **r**: Cluster radius.
- **σ**: Surface tension of the cluster (bulk-liquid value in CNT).
- **v₁**: Volume of one molecule in the condensed phase (**v₁ = molar massᵢ / (ρ × N_A)**).
- **k_B**: Boltzmann constant.
- **T**: Temperature.
- **ρ**: Density of the condensed phase.
- **N_A**: Avogadro's number.

**Description:**

The first term is the surface-energy penalty, which grows as **r²**. The second term is the volume free-energy gain from transferring molecules out of the supersaturated vapor, which grows as **r³**. For **S > 1** the sum passes through a maximum at the critical radius: clusters smaller than the critical size tend to evaporate, and clusters larger than it tend to grow.

### Critical Radius

**Equation 3: Critical Cluster Radius (Kelvin Relation)**

r* = (2 × σ × v₁) / (k_B × T × ln S)

**Where:**

- **r***: Critical cluster radius.

**Description:**

The critical radius is where **dΔG/dr = 0**. It is the same Kelvin relation that governs the equilibrium vapor pressure over curved surfaces in condensation (see the [condensation equations](Condensation_Equations.md#kelvin-effect-correction-factor)). A cluster of radius **r*** is in unstable equilibrium with the vapor: any fluctuation pushes it toward growth or evaporation.

### Nucleation Barrier

**Equation 4: Free-Energy Barrier**

ΔG* = (16 × π × σ³ × v₁²) / [3 × (k_B × T × ln S)²] = (4/3) × π × σ × (r*)²

**Where:**

- **ΔG***: Free-energy barrier at the critical radius.

**Description:**

The barrier height controls the nucleation rate exponentially. Because **ΔG*** scales as **σ³ / (ln S)²**, nucleation rates are extraordinarily sensitive to both surface tension and saturation ratio: small changes in either can shift the rate by many orders of magnitude. This sensitivity is the main practical difficulty in applying CNT quantitatively.

### Nucleation Rate

**Equation 5: Classical Homogeneous Nucleation Rate**

J = z × β* × N₁ × exp( −ΔG* / (k_B × T) )

**Where:**

- **J**: Nucleation rate (new stable clusters per unit volume per unit time).
- **z**: Zeldovich non-equilibrium factor (typically ~0.01-1), accounting for the fact that some clusters at the barrier top still evaporate.
- **β***: Rate at which vapor molecules collide with the critical cluster (condensation flux onto the critical cluster).
- **N₁**: Number concentration of vapor monomers.

**Description:**

The prefactor **z × β* × N₁** is kinetic: how often clusters at the critical size are hit by another monomer, weighted by the population of monomers available. The exponential is thermodynamic: the probability of fluctuating over the barrier. A commonly used closed form for a single species (Seinfeld & Pandis, 2016, Eq. 11.47) is:

J = ( (2 × σ) / (π × m₁) )^(1/2) × v₁ × N₁² × exp( −ΔG* / (k_B × T) )

where **m₁** is the molecular mass of the nucleating species.

## Binary and Multicomponent Nucleation

Atmospheric nucleation is rarely a single-species process. The most studied system is binary sulfuric acid-water nucleation, where the critical cluster contains both species and the free-energy surface is two-dimensional (a saddle point rather than a simple maximum).

**Equation 6: Binary Cluster Free Energy (Conceptual Form)**

ΔG(n₁, n₂) = n₁ × (μ₁,liquid − μ₁,gas) + n₂ × (μ₂,liquid − μ₂,gas) + 4 × π × r² × σ(x)

**Where:**

- **n₁, n₂**: Number of molecules of species 1 and 2 in the cluster.
- **μᵢ,liquid − μᵢ,gas**: Chemical potential difference for species **i** between the cluster liquid and the gas phase.
- **σ(x)**: Composition-dependent surface tension at cluster mole fraction **x**.

**Description:**

The critical cluster sits at the saddle point of **ΔG(n₁, n₂)**, and the nucleation rate follows the lowest free-energy path through that saddle. In practice, binary H₂SO₄-H₂O nucleation rates are evaluated with fitted parameterizations (for example Vehkamäki et al., 2002) rather than by direct evaluation of CNT, because CNT's bulk-property assumptions fail for clusters of a few molecules. Ternary systems (adding ammonia or amines) and ion-induced nucleation further lower the effective barrier and are handled by dedicated parameterizations or cluster-dynamics models.

## Empirical Nucleation Parameterizations

Because CNT is quantitatively unreliable for atmospheric systems, boundary-layer NPF is often represented with empirical rate laws fitted to observations. Two standard forms relate the nucleation rate to the sulfuric acid vapor concentration:

**Equation 7: Activation-Type Nucleation**

J = A × [H₂SO₄]

**Equation 8: Kinetic-Type Nucleation**

J = K × [H₂SO₄]²

**Where:**

- **[H₂SO₄]**: Gas-phase sulfuric acid number concentration.
- **A**: Activation coefficient (typically ~10⁻⁷ to 10⁻⁵ s⁻¹, site- and condition-dependent).
- **K**: Kinetic coefficient (typically ~10⁻¹⁴ to 10⁻¹² cm³ s⁻¹, site- and condition-dependent).

**Description:**

The activation form assumes existing thermodynamically stable clusters are activated by a single sulfuric acid molecule; the kinetic form assumes the rate-limiting step is a collision between two sulfuric-acid-containing molecules or clusters. Both are convenient source terms for aerosol dynamics models: they convert a modeled gas concentration directly into a particle number production rate at a prescribed formation size (typically ~1-2 nm diameter).

## Survival to Detectable and Model-Resolved Sizes

Freshly nucleated clusters must grow through the smallest sizes, where coagulational scavenging by pre-existing particles is fastest, before they matter for the resolved aerosol population. The apparent formation rate at a larger diameter **d** is related to the "true" nucleation rate at diameter **d*** by a survival probability.

**Equation 9: Kerminen-Kulmala Survival Relation**

J_d = J_d* × exp( γ × (1/d − 1/d*) × CS' / GR )

**Where:**

- **J_d**: Apparent particle formation rate at diameter **d**.
- **J_d***: Nucleation rate at the initial cluster diameter **d***.
- **γ**: Proportionality constant (~0.23 nm² m² h⁻¹ in the original formulation's units).
- **CS'**: Condensation sink of the pre-existing particle population (scavenging strength).
- **GR**: Growth rate of the freshly formed particles.

**Description:**

The exponential expresses the competition between growth (**GR**, escape to safety at larger sizes) and coagulational loss to the existing aerosol surface (**CS'**). High pre-existing surface area suppresses observable NPF even when the nucleation rate itself is large. In a simulation, this relation is a consistency check: if the model injects particles at a size larger than the true cluster size, the injection rate should be the survival-corrected **J_d**, not the raw **J_d***.

## Nucleation as a Source Term in Aerosol Dynamics

In an aerosol dynamics model, nucleation enters the population balance as a source of new particles at the smallest resolved size:

**Equation 10: Number Source Term**

dN/dt |_nucleation = J

**Equation 11: Coupled Gas Depletion**

dCᵢ/dt |_nucleation = − J × n*ᵢ × (molar massᵢ / N_A)

**Where:**

- **dN/dt |_nucleation**: Rate of new particle number production per unit volume.
- **dCᵢ/dt |_nucleation**: Rate of change of gas-phase mass concentration of species **i** due to nucleation.
- **n*ᵢ**: Number of molecules of species **i** in a freshly formed particle at the injection size.

**Description:**

Each nucleation event moves a small but nonzero mass of each participating vapor into the particle phase, so number production and gas depletion must be applied together to conserve mass. Numerical treatment differs by representation:

- **Binned/sectional:** add **J × Δt** particles (and the corresponding mass) to the smallest bin each timestep.
- **Particle-resolved with fixed slots:** activate inactive particle slots with the injection-size mass and composition. Because nucleation rates can be large, one computational particle typically represents many real particles via its concentration/weighting factor; the number of slots activated per step and the weight assigned to each is a resolution decision. When inactive slots run out, a resampling or volume-scaling policy is required.
- **Stiffness coupling:** freshly nucleated particles sit at the fast-equilibration end of the condensation stiffness range, so the nucleation source interacts directly with the time-integration scheme chosen for condensation.

**Implementation status:**

- No nucleation implementation exists in particula yet (CPU or GPU). This
  document is the theory reference for the planned nucleation/particle-source
  process.
- The planned design is a CPU reference process first, followed by a GPU
  version that activates inactive particle slots in fixed-shape arrays. Track
  this work in the
  [Data-Oriented Design and GPU Roadmap](../../../Features/Roadmap/data-oriented-gpu.md).

## Variable Descriptions

**Understanding the Parameters:**

1. **Saturation Ratio (S):**

   - The single most important control on homogeneous nucleation; rates change by orders of magnitude over small changes in **S**.
   - Set by the gas-phase concentration and the temperature-dependent saturation vapor pressure, so accurate vapor pressures are prerequisites for any nucleation calculation.

2. **Surface Tension (σ):**

   - Enters the barrier as **σ³**; the dominant uncertainty in CNT.
   - Bulk surface tension is a poor approximation for clusters of a few molecules (the main criticism of CNT); composition dependence matters for multicomponent clusters.

3. **Molecular Volume (v₁):**

   - Condensed-phase volume per molecule; links the molar mass and liquid density of the nucleating species.

4. **Critical Radius (r*):**

   - Typically ~0.5-2 nm for atmospheric conditions; clusters at this size contain only tens to hundreds of molecules.
   - The same Kelvin physics that penalizes condensation onto the smallest particles.

5. **Zeldovich Factor (z):**

   - Corrects the equilibrium cluster distribution for the fact that the barrier crossing is a diffusive process in cluster-size space.

6. **Condensation Sink (CS'):**

   - Integral measure of how quickly vapors and small clusters are scavenged by the pre-existing particle population.
   - Couples nucleation to the rest of the aerosol: more pre-existing surface means less survival of fresh clusters.

7. **Growth Rate (GR):**

   - Diameter growth rate of freshly formed particles, set by condensation of available vapors.
   - Together with **CS'**, controls what fraction of nucleated clusters survive to climate- and health-relevant sizes.

8. **Formation-Size Molecule Count (n*ᵢ):**

   - Composition of the particle injected into the model at the formation size; needed for mass conservation between the gas and particle phases.

**Applications and Implications:**

- **Aerosol Number Budgets:** Nucleation is the dominant source of particle number in many clean and moderately polluted environments, feeding the accumulation mode through subsequent growth.

- **Cloud Condensation Nuclei:** A substantial fraction of CCN originate as nucleated particles that grew by condensation and coagulation, so NPF connects gas-phase chemistry to cloud microphysics.

- **Chamber and Flow-Tube Experiments:** Prescribed-precursor experiments (the multi-box and parcel use cases) often begin with a nucleation burst; simulating them requires a particle source, not just growth of an initial population.

**Assumptions and Limitations:**

- **Capillarity Approximation:** CNT treats molecular clusters as spherical droplets with bulk liquid density and surface tension. For critical clusters of tens of molecules this is a strong assumption and the main source of CNT's quantitative error.

- **Steady-State Cluster Distribution:** The classical rate assumes the sub-critical cluster population is in steady state with the vapor; rapid changes in vapor concentration or temperature violate this.

- **Parameterization Validity Ranges:** Empirical forms (Equations 7-8) and fitted parameterizations (for example Vehkamäki et al., 2002) are only valid within the temperature, humidity, and concentration ranges of the underlying data; extrapolation can produce unphysical rates.

- **Injection-Size Convention:** Models inject particles at a chosen formation size, not at the true critical size. The nucleation rate must be survival-corrected (Equation 9) to be consistent with that choice.

**Further Considerations:**

- **Ion-Induced and Heterogeneous Pathways:** Ions and pre-existing surfaces lower the nucleation barrier; these pathways need separate rate expressions.

- **Cluster Dynamics Models:** Explicit cluster-population models (for example ACDC-type birth-death schemes) replace the CNT barrier picture with molecule-by-molecule kinetics and are the current standard for sulfuric acid-base systems.

- **Differentiability:** Nucleation rate expressions are smooth functions of gas concentrations and temperature, but the act of activating discrete particle slots is a discrete event. For gradient-based optimization, a binned or expected-value (mean-field) source term is differentiable, while discrete slot activation requires the same surrogate treatment as stochastic coagulation.

---

## Conclusion

Nucleation converts supersaturated vapor into new particles through a barrier-crossing process whose rate is exponentially sensitive to saturation ratio and surface tension. For aerosol dynamics modeling, the practical requirements are: an accurate temperature-dependent vapor pressure, a nucleation rate expression (CNT-based or empirical) valid for the target conditions, a survival correction to the model's injection size, and a mass-conserving numerical source term that adds particles to the resolved population. In particle-resolved simulations with fixed slot counts, that source term becomes slot activation, which ties nucleation directly to slot management, resampling policy, and time-integration stiffness at the smallest sizes.

---

## References

1. **Seinfeld, J. H., & Pandis, S. N. (2016).** *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change* (3rd ed.), Chapter 11: Nucleation. Wiley.

2. **Vehkamäki, H., Kulmala, M., Napari, I., Lehtinen, K. E. J., Timmreck, C., Noppel, M., & Laaksonen, A. (2002).** An improved parameterization for sulfuric acid-water nucleation rates for tropospheric and stratospheric conditions. *Journal of Geophysical Research: Atmospheres*, 107(D22), 4622. DOI: [10.1029/2002JD002184](https://doi.org/10.1029/2002JD002184)

3. **Kerminen, V.-M., & Kulmala, M. (2002).** Analytical formulae connecting the "real" and the "apparent" nucleation rate and the nuclei number concentration for atmospheric nucleation events. *Journal of Aerosol Science*, 33(4), 609-622. DOI: [10.1016/S0021-8502(01)00194-X](https://doi.org/10.1016/S0021-8502(01)00194-X)

4. **Kulmala, M., Lehtinen, K. E. J., & Laaksonen, A. (2006).** Cluster activation theory as an explanation of the linear dependence between formation rate of 3 nm particles and sulphuric acid concentration. *Atmospheric Chemistry and Physics*, 6, 787-793. DOI: [10.5194/acp-6-787-2006](https://doi.org/10.5194/acp-6-787-2006)

5. **Zhang, R., Khalizov, A., Wang, L., Hu, M., & Xu, W. (2012).** Nucleation and growth of nanoparticles in the atmosphere. *Chemical Reviews*, 112(3), 1957-2011. DOI: [10.1021/cr2001756](https://doi.org/10.1021/cr2001756)
