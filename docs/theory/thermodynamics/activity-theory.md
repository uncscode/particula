# Particle-phase activity in kinetic condensation

A droplet can evaporate in humid air or grow in subsaturated air. The direction depends on how the ambient vapor pressure compares with the vapor pressure at the droplet surface. Particle-phase water activity gives the composition contribution to that surface pressure. It changes as water or organic material enters or leaves the particle.

## From particle masses to activities

Consider a homogeneous liquid containing water and one organic compound. For particle masses $m_w,m_o$ in kg and molar masses $M_w,M_o$ in kg/mol, its organic mole fraction is

$$x_o = \frac{m_o/M_o}{m_o/M_o+m_w/M_w}, \qquad x_w=1-x_o.$$

The mole-fraction activity of species $i$ is $a_i=x_i\gamma_i$, referenced to the pure liquid at the same temperature. Ideal mixing gives $\gamma_i=1$. The Binary Activity Thermodynamics (BAT) model describes departures from ideality using a fit to the excess molar Gibbs energy. Composition varies along one coordinate for each fixed water–organic pair. The organic descriptors include molar mass, organic density, O:C and, where applicable, functional-group translation. The original fit used AIOMFAC calculations at 298.15 K. Particula's binary activity function takes no temperature argument. Temperature still enters the pure-component vapor pressure and curvature correction. [Gorkowski, Preston and Zuend (2019), sections 3.1–3.3](https://acp.copernicus.org/articles/19/13383/2019/)

### Mathematical Formulation

The BAT model computes activity coefficients from the excess Gibbs free energy of mixing. The activity coefficients are derived as:

$$\ln \gamma_w = g^E - x_{org} \cdot \frac{\partial g^E}{\partial x_{org}}$$

$$\ln \gamma_{org} = g^E + (1 - x_{org}) \cdot \frac{\partial g^E}{\partial x_{org}}$$

where:

- $g^E$ is the dimensionless excess Gibbs energy of mixing
- $x_{org}$ is the organic mole fraction
- $\gamma_w$ and $\gamma_{org}$ are the activity coefficients for water and organic

The activities are then:

$$a_w = \gamma_w \cdot (1 - x_{org})$$

$$a_{org} = \gamma_{org} \cdot x_{org}$$

## BAT and κ in a kinetic simulation

The κ formulation parameterizes water uptake through hygroscopicity and dry volume. For one dry component, the usual volume relation is $a_w=V_w/(V_w+\kappa V_{dry})$. BAT supplies both water and organic activities from the same binary composition. Choose the activity model according to the known material properties and the species whose vapor flux is needed. [Petters and Kreidenweis (2007)](https://doi.org/10.5194/acp-7-1961-2007)

For an ideal vapor above a spherical liquid droplet, write the equilibrium **surface** partial pressure as

$$p_{i,s}=a_i\,p_i^{sat}(T)\,K_i(r,T),$$

where $K_i$ is the Kelvin factor from the selected surface model. At a common temperature, a positive $p_{i,gas}-p_{i,s}$ drives condensation and a negative difference drives evaporation. The kinetic transport law sets the rate. The solver updates the masses and reevaluates activity at the new composition. Each particle can therefore have a different $a_w$ in the same ambient RH.

For example, at 90% RH a particle with $a_w=0.85$ has an inward water driving force in the flat-surface limit. A sufficiently large curvature correction ($K_w>0.90/0.85$) reverses the sign. Equality $\mathrm{RH}/100=a_w K_w$ describes zero net water flux under these assumptions. In a standalone equilibrium calculation one can instead prescribe water activity and solve for the compatible composition. [Equilibria theory](equilibria-theory.md)

## Single liquids and miscibility gaps

A homogeneous activity curve can pass through compositions where two liquids are thermodynamically favored. The coexistence compositions have equal water activities and equal organic activities across phases. In a binary equilibrium calculation, a common tangent to the molar Gibbs energy gives the two endpoints, and the overall mole fraction determines their molar proportions by the lever rule. Between the endpoints, evaluating the homogeneous curve at the bulk composition gives a different physical state. Using coexistence activities in a kinetic droplet model also assumes internal redistribution is fast on the timescale of vapor exchange. Surface composition and internal transport require additional treatment when this assumption fails. [BAT paper, Fig. 3 and section 4](https://acp.copernicus.org/articles/19/13383/2019/)

## Evaluating the binary function

The functional API returns both activities, both mass fractions and both activity coefficients. The existing particle strategy `ActivityNonIdealBinary.activity()` returns organic activity. For direct evaluation of particle-phase water activity use the first output below. At fixed organic properties these same two activity curves can be tabulated against $x_o$. Off-grid errors matter especially near zero net flux, where a small surface-pressure error can change the sign of growth.

### Example Usage

```python
from particula.activity import bat_activity_coefficients

# Calculate activities for a water-organic mixture
activity_water, activity_organic, mass_water, mass_organic, gamma_water, gamma_organic = (
    bat_activity_coefficients(
        molar_mass_ratio=0.09,  # M_water / M_organic
        organic_mole_fraction=0.3,
        oxygen2carbon=0.4,
        density=1400.0,
    )
)

print(f"Water activity: {activity_water:.3f}")
print(f"Organic activity: {activity_organic:.3f}")
print(f"Water activity coefficient: {gamma_water:.3f}")
print(f"Organic activity coefficient: {gamma_organic:.3f}")
```

## Reference implementation

Gorkowski, K., Preston, T. C., and Zuend, A. (2019). Relative-humidity-dependent organic aerosol thermodynamics via an efficient reduced-complexity model. *Atmospheric Chemistry and Physics*, 19, 13383–13407. The authors provide the original MATLAB implementation and study data for checking direct activity calculations. [Paper](https://doi.org/10.5194/acp-19-13383-2019) · [Supplement](https://doi.org/10.5194/acp-19-13383-2019-supplement) · [Model and data](https://github.com/Gorkowski/Binary_Activity_Thermodynamics_Model)
