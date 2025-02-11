# Droplet Coagulation Kernel Ayala 2008

Here, we discuss the implementation of the geometric collision kernel for cloud droplets as described in Part II by Ayala et al. (2008). Part I provides a detailed explanation of the direct numerical simulations. Where as Part II is the parameterization of the collision kernel for cloud droplets in turbulent flows. The implementation involves calculating the geometric collision rate of sedimenting droplets based on the turbulent flow properties and droplet characteristics.

Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 1. Results from direct numerical simulation. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015

Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 2. Theory and parameterization. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075016

## Γ₁₂: Kernel Description

In the parameterization below, the input parameters are:

- The radii a₁ and a₂ of the droplets
- The water density ρ_w
- Turbulent air flow requires:
    - The density ρ
    - The viscosity ν
    - The turbulence dissipation rate ε
    - The Taylor-microscale Reynolds number R_λ
- The gravitational acceleration |g|

The output is the collision kernel Γ₁₂

This is valid under the conditions when a_k ≪ η, ρ_w ≫ ρ, and Sv > 1, the geometric collision kernel can be calculated as follows:

Γ₁₂ = 2πR² ⟨|wᵣ|⟩ g₁₂

## ⟨|wᵣ|⟩: Radial Relative Velocity

There are two options for calculating the radial relative velocity:

### ⟨|wᵣ|⟩ Dodin 2002:

Using the spherical formulation, Dodin and Elperin (2002), decomposed the relative velocity into turbulent and gravity-induced components and assumed that the turbulent component is normally distributed.

Dodin Z and Elperin T 2002 Phys. Fluids 14 2921–24

⟨|wᵣ|⟩ = √(2⁄π) σ f(b)

where:

f(b) = (½)√π (b + 0.5⁄b) erf(b) + (½) exp(−b²)

b = [g |τₚ₁ − τₚ₂|]⁄[√2 σ]

### ⟨|wᵣ|⟩ Ayala 2008:

Here both particle inertia and gravitational effects are accounted for in the relative velocity calculation. Derived by Ayala et al. (2008) based on DNS results.

⟨|wᵣ|⟩ = √(2⁄π) √[σ² + (π⁄8)(τₚ₁ + τₚ₂)² |g|²]

### σ²: Direct Numerical Simulation Fit

σ² = ⟨(v′^(2))²⟩ + ⟨(v′^(1))²⟩ − 2 ⟨v′^(1) v′^(2)⟩

The term ⟨(v′^(2))²⟩ is the square of the RMS fluctuation velocity of droplet 2, and ⟨v′^(1) v′^(2)⟩ is the cross-correlation of the fluctuating velocities of droplets 1 and 2.

The square of the RMS fluctuation velocity is given by, for k-th droplet:

⟨(v′^(k))²⟩ = [u′² ⁄ τₚₖ] [b₁ d₁ Ψ(c₁, e₁) − b₁ d₂ Ψ(c₁, e₂) − b₂ d₁ Ψ(c₂, e₁) + b₂ d₂ Ψ(c₂, e₂)],

Cross term is defined as:

⟨v′^(1) v′^(2)⟩ = [u′² f₂(R)]⁄[τₚ₁ τₚ₂] × [b₁ d₁ Φ(c₁, e₁) − b₁ d₂ Φ(c₁, e₂) − b₂ d₁ Φ(c₂, e₁) + b₂ d₂ Φ(c₂, e₂)].

### f₂(R): Longitudinal velocity correlation

f₂(R) = [1 ⁄ 2√(1 − 2β²)]  
{  
 [1 + √(1 − 2β²)]  
 e^[-2R⁄((1 + √(1 − 2β²)) Lₑ)]  
 − [1 − √(1 − 2β²)]  
 e^[-2R⁄((1 − √(1 − 2β²)) Lₑ)]  
}

#### b₁, b₂, c₁, c₂, d₁, d₂, e₁, e₂: Definitions

b₁ = (1 + √(1 − 2z²)) / (2√(1 − 2z²))

b₂ = (1 − √(1 − 2z²)) / (2√(1 − 2z²))

c₁ = ((1 + √(1 − 2z²))T_L) / 2

c₂ = ((1 − √(1 − 2z²))T_L) / 2

d₁ = (1 + √(1 − 2β²)) / (2√(1 − 2β²))

d₂ = (1 − √(1 − 2β²)) / (2√(1 − 2β²))

e₁ = ((1 + √(1 − 2β²))L_e) / 2

e₂ = ((1 − √(1 − 2β²))L_e) / 2

#### z and β: Definitions

z = τ_T ⁄ T_L

β = (√2 λ) ⁄ L_e

#### Φ(α, φ): Definitions

For the case when taking vₚ₁ > vₚ₂.

Φ(α, φ) = term_1 + term_2 + term_3

term_1 = { 
 1 / ( (vₚ2/φ) − (1/τₚ2) − (1/α) ) − 
 1 / ( (vₚ1/φ) + (1/τₚ1) + (1/α) ) 
} × [ (vₚ1 − vₚ2) / (2φ ((vₚ1 − (vₚ2/φ)) + (1/τₚ1) + (1/τₚ2))² ) ]

term_2 = { 
 4 / [ ( (vₚ2/φ) + (1/τₚ2) + (1/α) )² − ( (vₚ2/φ) − (1/τₚ2) − (1/α) )² ] 
} × [ (vₚ2) / (2φ ((1/τₚ1) − (1/α) + ((1/τₚ2) + (1/α))(vₚ1/vₚ2)) ) ]

term_3 = { 
 2φ / ( (vₚ1/φ) + (1/τₚ1) + (1/α) ) − 
 2φ / ( (vₚ2/φ) − (1/τₚ2) − (1/α) ) 
} × [ 1 / (2φ ((vₚ1 − (vₚ2/φ)) + (1/τₚ1) + (1/τₚ2)) ) ]

#### Ψ(α, φ): Definitions

For the case when taking for the k-th droplet:

Ψ(α, φ) = 1 / ( (1/τₚₖ) + (1/α) + (vₚₖ/φ) ) − 
 (vₚₖ) / (2φ ((1/τₚₖ) + (1/α) + (vₚₖ/φ))²)

#### g₁₂: Radial Distribution Function

The radial distribution function is given by:

g₁₂ = ( (η² + r_c²) / (R² + r_c²) )^(C₁/2)

Where C₁ and r_c are derived based on droplet and turbulence properties.

##### C₁: Calculation

C₁ = y(St) ⁄ [|g| ⁄ (v_k ⁄ τ_k)]^(f₃(R_λ))

y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St

f₃(R_λ) = 0.1886 exp(20.306 / R_λ)

Where:

St = max(St₁, St₂)

Since the fitting for y(St) was done for a limited range of St in DNS, it should be set to zero for large St when the function y(St) becomes negative.

##### r_c: Expression

(r_c / η)² = |St₂ - St₁| F(aₒ, R_λ)

solving for r_c:

r_c = η √(|St₂ - St₁| F(aₒ, R_λ))

where:

aₒ = aₒ + (π / 8) (|g| / (v_k / τ_k))²

F(aₒ, R_λ) = 20.115 (aₒ / R_λ)^(1/2)

---

## Derived Parameters

### τ_k: Kolmogorov Time

The smallest timescale in turbulence where viscous forces dominate:

τ_k = √(ν⁄ε)

### η: Kolmogorov Length Scale

The smallest scale in turbulence:

η = [ν³⁄ε]^(¼)

### v_k: Kolmogorov Velocity Scale

A velocity scale related to the smallest turbulent eddies:

v_k = [ν ε]^(¼)

### u′: Fluid RMS Fluctuation Velocity

Quantifies turbulence intensity:

u′ = [R_λ^(½) v_k]⁄[15^(¼)]

### T_L: Lagrangian Integral Scale

Describes large-scale turbulence:

T_L = u'² / ε

### Lₑ: Eulerian Integral Scale

Length scale for large eddies:

Lₑ = 0.5 u'³ / ε

### aₒ: Coefficient

A Reynolds-dependent parameter:

aₒ = [11 + 7 R_λ] ⁄ [205 + R_λ]

### τ_T: Lagrangian Taylor Microscale Time

Time correlation decay for turbulent trajectories:

τ_T = τ_k √[2 R_λ ⁄ (15^(½) aₒ)]

### λ: Taylor Microscale

Length scale linked to fluid flow:

λ = u' √[15 ν² ⁄ ε]

### τ_p: Droplet Inertia Time

Adjusts droplet inertia:

τ_p = [2⁄9] [ρ_w⁄ρ] [a²⁄(ν f(Re_p))]

with:

f(Re_p) = 1 + 0.15 Re_p^(0.687)

### v_p: Droplet Settling Velocity

The settling velocity under gravity:

v_p = τ_p |g|

### Re_p: Particle Reynolds Number

Characterizes droplet flow:

Re_p = 2 a v_p / ν

### St: Stokes Number

Non-dimensional inertia parameter:

St = τ_p⁄τ_k

---

## Variable Descriptions

Here are the variables, their definitions.

### Droplet (Particle) Properties

- a₁, a₂: Radii of the droplets. These determine size-dependent properties such as droplet inertia and terminal velocity. 

- ρ_w: Density of water. The mass per unit volume of water, typically 1000 kg/m³. It is essential for calculating droplet inertia and terminal velocity.

- ρ: Density of air. The mass per unit volume of air, affecting drag and settling velocity. Typical sea-level values are around 1.225 kg/m³.

- ν: Kinematic viscosity. The ratio of dynamic viscosity to fluid density, quantifying resistance to flow.

- τ_p: Droplet inertial response time. The characteristic time it takes for a droplet to adjust to changes in the surrounding airflow, critical for droplet motion analysis.

- v′^{(i)}_p: Particle RMS fluctuation velocity. The root mean square of the fluctuating velocity component, representing variability in turbulent flow.

- $f_u$: Particle response coefficient. Measures how particles respond to fluid velocity fluctuations, helping quantify their turbulent motion.

- f(R): Spatial correlation coefficient. Describes the correlation of fluid velocities at two points separated by a distance R, influencing droplet interactions.

- g₁₂: Radial distribution function (RDF). A measure of how particle pairs are spatially distributed due to turbulence and gravity.

### Turbulent Flow Properties

- Yᶠ(t): Fluid Lagrangian trajectory. The path traced by a fluid particle as it moves through turbulence.

- ε: Turbulence dissipation rate. The rate at which turbulent kinetic energy is converted into thermal energy per unit mass.

- R_λ: Reynolds number. A dimensionless number that characterizes the flow regime, depending on turbulence intensity and scale.

- λ_D: Longitudinal Taylor-type microscale. A characteristic length scale of fluid acceleration in turbulence, related to energy dissipation and viscosity.

- T_L: Lagrangian integral scale. The timescale over which fluid particles maintain velocity correlations, describing large-scale turbulence behavior.

- u′: Fluid RMS fluctuation velocity. The root mean square of fluid velocity fluctuations, characterizing turbulence intensity.

- *S*: Skewness of longitudinal velocity gradient. A measure of asymmetry in velocity gradient fluctuations, significant for small-scale turbulence analysis.

- Yᶠ(t): Fluid Lagrangian trajectory. The path traced by a fluid particle as it moves through turbulence.

- τₜ: Lagrangian Taylor microscale time. A timescale describing the decay of velocity correlation along a fluid particle trajectory.

### g: Gravitational Acceleration

The acceleration due to gravity, approximately 9.81 m/s² on Earth's surface. This force drives droplet sedimentation in turbulent air.


