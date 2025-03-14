# Droplet Coagulation Kernel Ayala 2008

Here, we discuss the implementation of the geometric collision kernel for cloud droplets as described in Part II by Ayala et al. (2008). Part I provides a detailed explanation of the direct numerical simulations. Where as Part II is the parameterization of the collision kernel for cloud droplets in turbulent flows. The implementation involves calculating the geometric collision rate of sedimenting droplets based on the turbulent flow properties and droplet characteristics.

Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 1. Results from direct numerical simulation. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015

Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 2. Theory and parameterization. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075016

## $\Gamma_{12}$: Kernel Description

In the parameterization below, the input parameters are:

- The radii $a_1$ and $a_2$ of the droplets
- The water density $\rho_w$
- Turbulent air flow requires:
    - The density $\rho$
    - The viscosity $\nu$
    - The turbulence dissipation rate $\varepsilon$
    - The Taylor-microscale Reynolds number $R_\lambda$
- The gravitational acceleration $|g|$

The output is the collision kernel $\Gamma_{12}$

This is valid under the conditions when $a_k << \eta$, $\rho_w >> \rho$, and $Sv > 1$ , the
geometric collision kernel can be calculated as follows:

$$
\Gamma_{12} = 2\pi R^2 \langle |w_r| \rangle g_{12}
$$

## $\langle |w_r| \rangle$: Radial Relative Velocity

There are two options for calculating the radial relative velocity:

### $\langle |w_r| \rangle$ Dodin 2002:

Using the spherical formulation, Dodin and Elperin (2002), decomposed the relative velocity into turbulent and gravity-induced components and assumed that the turbulent component is normally distributed.

Dodin Z and Elperin T 2002 Phys. Fluids 14 2921–24

$$
\langle |w_r| \rangle = \sqrt{\frac{2}{\pi}} \sigma f(b)
$$

where:

$$
f(b) = \frac{1}{2}\sqrt{\pi}\left(b + \frac{0.5}{b}\right)\text{erf}(b) + \frac{1}{2}\exp(-b^2)
$$

$$
b = \frac{g|\tau_{p1} - \tau_{p2}|}{\sqrt{2} \sigma}
$$

### $\langle |w_r| \rangle$ Ayala 2008:

Here both particle inertia and gravitational effects are accounted for in the relative velocity calculation. Derived by Ayala et al. (2008) based on DNS results.

$$
\langle |w_r| \rangle = \sqrt{\frac{2}{\pi}} \left(\sigma^2 + \pi/8 (\tau_{p1} + \tau_{p2})^2 |g|^2\right)^{1/2}
$$

### $\sigma^2$ Direct Numerical Simulation Fit

$$
\sigma^2 = \langle (v'^{(2)})^2 \rangle + \langle (v'^{(1)})^2 \rangle - 2 \langle v'^{(1)} v'^{(2)} \rangle
$$

The term $\langle (v'^{(2)})^2 \rangle$ is the square of the RMS fluctuation velocity of droplet 2, and $\langle v'^{(1)} v'^{(2)} \rangle$ is the cross-correlation of the fluctuating velocities of droplets 1 and 2.

The square of the RMS fluctuation velocity is given by, for k-th droplet:

$$
\left\langle \left(v'^{(k)}\right)^2 \right\rangle = \frac{u'^2}{\tau_{pk}}
\left[b_1 d_1 \Psi(c_1, e_1) - b_1 d_2 \Psi(c_1, e_2) - b_2 d_1 \Psi(c_2, e_1) + b_2 d_2 \Psi(c_2, e_2)\right],
$$

Cross terms is defined as:

$$
\left\langle v'^{(1)} v'^{(2)} \right\rangle = \frac{u'^2 f_2(R)}{\tau_{p1} \tau_{p2}}\\
\times \left[b_1 d_1 \Phi(c_1, e_1) - b_1 d_2 \Phi(c_1, e_2) - b_2 d_1 \Phi(c_2, e_1) + b_2 d_2 \Phi(c_2, e_2)\right].
$$

### $f_2(R)$ Longitudinal velocity correlation

This is the longitudinal two-point velocity correlation function, which is a function of the separation distance R between two points in the flow. The function evaluated at r = R is given by:

$$
f_2(R) = \frac{1}{2(1 - 2\beta^2)^{1/2}} 
\Bigg\{
\left(1 + \sqrt{1 - 2\beta^2}\right) \\
\times \exp\left[-\frac{2R}{(1 + \sqrt{1 - 2\beta^2})L_e}\right] \quad - \left(1 - \sqrt{1 - 2\beta^2}\right) \\
\times \exp\left[-\frac{2R}{(1 - \sqrt{1 - 2\beta^2})L_e}\right]
\Bigg\}
$$

#### $b_1, b_2, c_1, c_2, d_1, d_2, e_1, e_2$: Definitions

$$
b_1 = \frac{1 + \sqrt{1 - 2z^2}}{2\sqrt{1 - 2z^2}}
$$

$$
b_2 = \frac{1 - \sqrt{1 - 2z^2}}{2\sqrt{1 - 2z^2}}
$$

$$
c_1 = \frac{(1 + \sqrt{1 - 2z^2})T_L}{2}
$$

$$
c_2 = \frac{(1 - \sqrt{1 - 2z^2})T_L}{2}
$$

$$
d_1 = \frac{1 + \sqrt{1 - 2\beta^2}}{2\sqrt{1 - 2\beta^2}}
$$

$$
d_2 = \frac{1 - \sqrt{1 - 2\beta^2}}{2\sqrt{1 - 2\beta^2}}
$$

$$
e_1 = \frac{(1 + \sqrt{1 - 2\beta^2})L_e}{2}
$$

$$
e_2 = \frac{(1 - \sqrt{1 - 2\beta^2})L_e}{2}
$$

#### $z$ and $\beta$: Definitions

$$
z = \frac{\tau_T}{L_e}
$$

$$
\beta = \frac{\sqrt{2} \lambda}{L_e}
$$

#### $\Phi(\alpha, \phi)$: Definitions

For the case when taking $v_{p1}>v_{p2}$.

$$
\Phi(\alpha, \phi) =
\Bigg\{
\frac{1}{\left(v_{p2}/\phi - (1/\tau_{p2}) - (1/\alpha)\right)} 
- \frac{1}{\left(v_{p1}/\phi + (1/\tau_{p1}) + (1/\alpha)\right)} \Bigg\} \\
\quad \times \frac{v_{p1} - v_{p2}}{2\phi \left((v_{p1} - v_{p2}/\phi) + (1/\tau_{p1}) + (1/\tau_{p2})\right)^2} 
+ \Bigg\{
\frac{4}{\left(v_{p2}/\phi\right)^2 - \left((1/\tau_{p2}) + (1/\alpha)\right)^2} \\
\quad -
\frac{1}{\left(v_{p2}/\phi + (1/\tau_{p2}) + (1/\alpha)\right)^2} 
- \frac{1}{\left(v_{p2}/\phi - (1/\tau_{p2}) - (1/\alpha)\right)^2} \Bigg\} \\
\quad \times \frac{v_{p2}}{2\phi \left((1/\tau_{p1}) - (1/\alpha) + ((1/\tau_{p2}) + (1/\alpha))(v_{p1}/v_{p2})\right)} \\
\quad + \Bigg\{
\frac{2\phi}{\left((v_{p1}/\phi) + (1/\tau_{p1}) + (1/\alpha)\right)} 
- \frac{2\phi}{\left((v_{p2}/\phi) - (1/\tau_{p2}) - (1/\alpha)\right)} \\
\quad -
\frac{v_{p1}}{\left((v_{p1}/\phi) + (1/\tau_{p1}) + (1/\alpha)\right)^2} 
+ \frac{v_{p2}}{\left((v_{p2}/\phi) - (1/\tau_{p2}) - (1/\alpha)\right)^2} \Bigg\} \\
\quad \times \frac{1}{2\phi \left((v_{p1} - v_{p2}/\phi) + (1/\tau_{p1}) + (1/\tau_{p2})\right)}
$$

#### $\Psi(\alpha, \phi)$: Definitions

For the case when taking for the k-th droplet:

$$
\Psi(\alpha, \phi) = \frac{1}{(1/\tau_{pk}) + (1/\alpha) + (v_{pk}/\phi)} 
- \frac{v_{pk}}{2\phi \left((1/\tau_{pk}) + (1/\alpha) + (v_{pk}/\phi)\right)^2}
$$

#### $g_{12}$: Radial Distribution Function

The radial distribution function is given by:

$$
g_{12} = \left(\frac{\eta^2 + r_c^2}{R^2 + r_c^2}\right)^{C_1/2}
$$

Where $C_1$ and $r_c$ are derived based on droplet and turbulence properties.

##### $C_1$: Calculation

$$
C_1 = \frac{y(St)}{\left(|\mathbf{g}| / (v_k / \tau_k)\right)^{f_3(R_\lambda)}}
$$

$$
y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St
$$

$$
f_3(R_\lambda) = 0.1886 \exp\left(\frac{20.306}{R_\lambda}\right)
$$

Where:

$$ St = max(St_1, St_2) $$

Since the fitting for $y(St)$ was done for a limited range of St in DNS,
it should be set to zero for large $St$ when the function $y(St)$ becomes negative.

##### $r_c$: Expression

$$
\left(\frac{r_c}{\eta}\right)^2 = |St_2 - St_1| F(a_o, R_\lambda)
$$

solving for $r_c$:

$$
r_c = \eta \sqrt{|St_2 - St_1| F(a_o, R_\lambda)}
$$

where:

$$
a_{Og} = a_o + \frac{\pi}{8} \left(\frac{|\mathbf{g}|}{v_k / \tau_k}\right)^2
$$

$$
F(a_{Og}, R_\lambda) = 20.115 \left(\frac{a_{Og}}{R_\lambda}\right)^{1/2}
$$

---

## Derived Parameters

### $\tau_k$: Kolmogorov Time

The smallest timescale in turbulence where viscous forces dominate:

$$
\tau_k = \left(\frac{\nu}{\varepsilon}\right)^{1/2}
$$

### $\eta$: Kolmogorov Length Scale

The smallest scale in turbulence:

$$
\eta = \left(\frac{\nu^3}{\varepsilon}\right)^{1/4}
$$

### $v_k$: Kolmogorov Velocity Scale

A velocity scale related to the smallest turbulent eddies:

$$
v_k = (\nu \varepsilon)^{1/4}
$$

### $u'$: Fluid RMS Fluctuation Velocity

Quantifies turbulence intensity:

$$
u' = \frac{R_\lambda^{1/2} v_k}{15^{1/4}} 
$$

### $T_L$: Lagrangian Integral Scale

Describes large-scale turbulence:

$$
T_L = \frac{u'^2}{\epsilon}
$$

### $L_e$: Eulerian Integral Scale

Length scale for large eddies:

$$
L_e = 0.5 \frac{u'^3}{\epsilon}
$$

### $a_o$: Coefficient

A Reynolds-dependent parameter:

$$
a_o = \frac{11+7 R_\lambda}{205 + R_\lambda}
$$

### $\tau_T$: Lagrangian Taylor Microscale Time

Time correlation decay for turbulent trajectories:

$$
\tau_T = \tau_k \left(\frac{2 R_\lambda}{15^{1/2} a_o}\right)^{1/2}
$$

### $\lambda$: Taylor Microscale

Length scale linked to fluid flow:

$$
\lambda = u' \left(\frac{15 \nu^2}{\epsilon}\right)^{1/2}
$$

### $\tau_p$: Droplet Inertia Time

Adjusts droplet inertia:

$$
\tau_p = \frac{2}{9} \frac{\rho_w}{\rho} \frac{a^2}{\nu f(Re_p)}
$$

with:

$$
f(Re_p) = 1 + 0.15 Re_p^{0.687}
$$

### $v_p$: Droplet Settling Velocity

The settling velocity under gravity:

$$
v_p = \tau_p |g|
$$

### $Re_p$: Particle Reynolds Number

Characterizes droplet flow:

$$
Re_p = \frac{2 a v_p}{\nu}
$$

### $St$: Stokes Number

Non-dimensional inertia parameter:

$$
St = \frac{\tau_p}{\tau_k}
$$

---

## Variable Descriptions

Here are the variables, their definitions.

### Droplet (Particle) Properties

- $a_1, a_2$: Radii of the droplets. These determine size-dependent properties such as droplet inertia and terminal velocity. 

- $\rho_w$: Density of water. The mass per unit volume of water, typically $1000 \, \text{kg/m}^3$. It is essential for calculating droplet inertia and terminal velocity.

- $\rho$: Density of air. The mass per unit volume of air, affecting drag and settling velocity. Typical sea-level values are around $1.225 \, \text{kg/m}^3$.

- $\nu$: Kinematic viscosity. The ratio of dynamic viscosity to fluid density, quantifying resistance to flow.

- $\tau_p$: Droplet inertial response time. The characteristic time it takes for a droplet to adjust to changes in the surrounding airflow, critical for droplet motion analysis.

- $v'^{(i)}_p$: Particle RMS fluctuation velocity. The root mean square of the fluctuating velocity component, representing variability in turbulent flow.

- $f_u$: Particle response coefficient. Measures how particles respond to fluid velocity fluctuations, helping quantify their turbulent motion.

- $f(R)$: Spatial correlation coefficient. Describes the correlation of fluid velocities at two points separated by a distance $R$, influencing droplet interactions.

- $g_{12}$: Radial distribution function (RDF). A measure of how particle pairs are spatially distributed due to turbulence and gravity.

### Turbulent Flow Properties

- $u$: Local air velocity. The instantaneous velocity of air at a given point. Turbulence causes $u$ to vary in space and time.

- $\varepsilon$: Turbulence dissipation rate. The rate at which turbulent kinetic energy is converted into thermal energy per unit mass.

- $R_\lambda$: Reynolds number. A dimensionless number that characterizes the flow regime, depending on turbulence intensity and scale.

- $\lambda_D$: Longitudinal Taylor-type microscale. A characteristic length scale of fluid acceleration in turbulence, related to energy dissipation and viscosity.

- $T_L$: Lagrangian integral scale. The timescale over which fluid particles maintain velocity correlations, describing large-scale turbulence behavior.

- $u'$: Fluid RMS fluctuation velocity. The root mean square of fluid velocity fluctuations, characterizing turbulence intensity.

- $S$: Skewness of longitudinal velocity gradient. A measure of asymmetry in velocity gradient fluctuations, significant for small-scale turbulence analysis.

- $Y^f(t)$: Fluid Lagrangian trajectory. The path traced by a fluid particle as it moves through turbulence.

- $\tau_T$: Lagrangian Taylor microscale time. A timescale describing the decay of velocity correlation along a fluid particle trajectory.

### $g$: Gravitational Acceleration

The acceleration due to gravity, approximately 9.81 $\text{m/s}^2$ on Earth's surface. This force drives droplet sedimentation in turbulent air.


