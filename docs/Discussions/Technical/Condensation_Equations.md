# Condensation Discussion

Iso thermal and non-isothermal condensation processes are important in aerosol dynamics. The condensation process is the addition of a gas phase species to the particle phase. This can be a reversible process, where the species can evaporate back into the gas phase. The condensation process is important in the formation of cloud droplets, and the growth of particles in the atmosphere.

## Condensation Equations (Isothermal)

With the gas phase and particle phase defined, we can start the condensation process. Excluding the latent heat of vaporization for the gas-particle phase transition, which is important in cloud droplet formation.

This follows Chapter 2 (EQ 2.41) by Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling (D. Topping & M. Bane, Eds.). Wiley. https://doi.org/10.1002/9781119625728. Also Chapter 12 and 13 (EQ 13.3) of Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.

The isothermal condensation or evaporation process is defined by the following equation:

dmi/dt = N × [k_cond × (pᵢ, gas − pᵢ, particle surface)] × [molar massᵢ / (R × T)]

Where:

- mᵢ is the mass of species i in the particle phase, of a specific bin
- N is the number of particles
- k_cond is the per-particle for order condensation coefficient
- pᵢ, gas is the partial pressure of species i in the gas phase
- pᵢ, particle surface is the partial pressure of species i at the surface of the particle, accounting for Kelvin effect and activity coefficients.
- R is the ideal gas constant
- T is the temperature
- molar massᵢ is the molar mass of species 

- m is the mass of the droplet
- radius_wet is the wet radius of the droplet
- Dᵢ is the vapor diffusion coefficient of species i
- pᵢ, gas is the partial pressure of species i in the gas phase
- pᵢ, particle surface is the partial pressure of species i at the surface of the particle, accounting for Kelvin effect and activity coefficients.
- Lᵢ is the latent heat of vaporization of species i
- κ is the thermal conductivity of air
- T is the temperature
- R is the ideal gas constant
- Rᵢ is the specific gas constant for species i (R / molar massᵢ)
- dm/dt is the rate of change of mass of species i in the particle phasei
- dmi/dt is the rate of change of mass of species i in the particle phase

The first order condensation coefficient is defined as:

k_cond = 4π × radius_particle × Dᵢ × f(Kn, α)

Where:

- radius_particle is the radius of the particle
- Dᵢ is the vapor diffusion coefficient of species i
- f(Kn, α) is the correction factor for the molecular regime to continuum regime transition. This is a function of the Knudsen number and the accommodation coefficient.

The correction factor is defined as:

f(Kn, αᵢ) = [0.75 × αᵢ × (1 + Kn)] / [(Kn² + Kn) + 0.283 × αᵢ × Kn + 0.75 × αᵢ]

Where:

- αᵢ is the accommodation coefficient of species i
- Kn is the Knudsen number
- Kn = λᵢ / radius_particle
  - λ is the mean free path of the gas molecules of species i
  - radius is the radius of the particle

## Condensation Equations (Latent Heat)

The condensation process now includes the latent heat of vaporization, which is significant in cloud droplet formation.

The condensation or evaporation process accounting for latent heat is defined by the following equation:

dm/dt = numerator / denominator

numerator = N × 4π × radius_wet × Dᵢ × (pᵢ, gas − pᵢ, particle surface)

denominator = [Dᵢ × Lᵢ × pᵢ] / [κ × T] × [(Lᵢ / (R × T)) − 1] + Rᵢ × T

Where:

- dm/dt is the rate of change of mass of species i in the particle phase
- m is the mass of the droplet
- radius_wet is the wet radius of the droplet
- Dᵢ is the vapor diffusion coefficient of species i
- pᵢ, gas is the partial pressure of species i in the gas phase
- pᵢ, particle surface is the partial pressure of species i at the surface of the particle, accounting for Kelvin effect and activity coefficients.
- Lᵢ is the latent heat of vaporization of species i
- κ is the thermal conductivity of air
- T is the temperature
- R is the ideal gas constant
- Rᵢ is the specific gas constant for species i (R / molar massᵢ)

### Partial Pressures

The partial pressures of species $i$ in the gas phase and at the surface of the particle are defined as:

pᵢ, gas = concᵢ, gas × (R × T) / molar massᵢ

Where:
- concᵢ, gas is the concentration of species i in the gas phase

At the surface of the particle, the partial pressure is defined as:

pᵢ, particle surface = pᵢ^pure × γᵢ × xᵢ × kᵢ, Kelvin

Where:

- pᵢ^pure is the saturation vapor pressure of species i, sometimes called pᵢ^sat, pᵢ^vap, or pᵢ^0
- γᵢ is the activity coefficient of species i
- xᵢ is the mole fraction of species i in the particle phase
- kᵢ, Kelvin is the Kelvin effect correction factor
  - kᵢ, Kelvin = exp(kᵢ, Kelvin radius / radius_particle)
  - kᵢ, Kelvin radius = [2 × σ_surface × molar massᵢ ] / [R × T × density]
    - σ_surface is the effective surface tension of the particle.
    - density is the effective density of the particle.
