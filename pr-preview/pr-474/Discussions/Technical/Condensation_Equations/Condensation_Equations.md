# Condensation Discussion

Iso thermal and non-isothermal condensation processes are important in aerosol dynamics. The condensation process is the addition of a gas phase species to the particle phase. This can be a reversible process, where the species can evaporate back into the gas phase. The condensation process is important in the formation of cloud droplets, and the growth of particles in the atmosphere.

## Condensation Equations (Isothermal)

With the gas phase and particle phase defined, we can start the condensation process. Excluding the latent heat of vaporization for the gas-particle phase transition, which is important in cloud droplet formation.

This follows Chapter 2 (EQ 2.41) by Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling (D. Topping & M. Bane, Eds.). Wiley. https://doi.org/10.1002/9781119625728. Also Chapter 12 and 13 (EQ 13.3) of Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.

The isothermal condensation or evaporation process is defined by the following equation:

$$
\frac{dm_{i}}{dt} = N \frac{k_{cond} (p_{i, gas} - p_{i, particle~surface})}{RT/ molar~mass_{i}}
$$

Where:

- $m_{i}$ is the mass of species $i$ in the particle phase, of a specific bin
- $N$ is the number of particles
- $k_{cond}$ is the per-particle for order condensation coefficient
- $p_{i, gas}$ is the partial pressure of species $i$ in the gas phase
- $p_{i, particle~surface}$ is the partial pressure of species $i$ at the surface of the particle, acounting for Kelvin effect and activity coefficients.
- $R$ is the ideal gas constant
- $T$ is the temperature
- $molar mass_{i}$ is the molar mass of species $i$
- $dm_{i}/dt$ is the rate of change of mass of species $i$ in the particle phase

The first order condensation coefficient is defined as:

$$
k_{cond} = 4 \pi~radius_{particle}~D_{i}~f(Kn, \alpha)
$$

Where:

- $radius_{particle}$ is the radius of the particle
- $D_{i}$ is the vapor diffusion coefficient of species $i$
- $f(Kn, \alpha)$ is the correction factor for the molecular regime to continuum regime transition. This is a function of the Knudsen number and the accommodation coefficient.

The correction factor is defined as:

$$
f(Kn, \alpha_{i, accom.}) = \frac{0.75 \alpha_{i, accom.} (1 + Kn)}{
    (Kn^2 + Kn) + 0.283 \alpha_{i, accom.} Kn + 0.75 \alpha_{i, accom.}}
$$

Where:

- $\alpha_{i, accom.}$ is the accommodation coefficient of species $i$
- $Kn$ is the Knudsen number
- $Kn = \frac{\lambda_{i}}{radius_{particle}}$
  - $\lambda$ is the mean free path of the gas molecules of species $i$
  - $radius$ is the radius of the particle


### Partial Pressures

The partial pressures of species $i$ in the gas phase and at the surface of the particle are defined as:

$$
p_{i, gas} = conc_{i, gas} RT/molar~mass_{i}
$$

Where:
- $conc_{i, gas}$ is the concentration of species $i$ in the gas phase

At the surface of the particle, the partial pressure is defined as:

$$
p_{i, particle~surface} = p^{pure}_{i} \gamma_{i} x_{i} k_{i,Kelvin}
$$

Where:

- $p^{pure}_{i}$ is the saturation vapor pressure of species $i$, sometimes called $p^{sat}_{i}$, $p^{vap}_{i}$, or $p^{0}_{i}$
- $\gamma_{i}$ is the activity coefficient of species $i$
- $x_{i}$ is the mole fraction of species $i$ in the particle phase
- $k_{i,Kelvin}$ is the Kelvin effect correction factor
  - $k_{i,Kelvin} = exp(k_{i, Kelvin~radius}/radius_{particle})$
  - $k_{i, Kelvin~radius} = 2 \sigma_{surface}~molar~mass_{i} / (R  T ~ density)$
    - $\sigma_{surface}$ is the effective surface tension of the particle.
    - $density$ is the effective density of the particle.