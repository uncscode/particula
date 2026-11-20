# Ion–particle encounters

## Dimensional collision kernel

For an ion and an aerosol particle, let $R$ be their summed radii (m),
$m_r$ their reduced mass (kg) and $f_r$ their reduced friction coefficient
(kg/s). The attraction or repulsion energy relative to thermal energy
determines the dimensionless kinetic and continuum enhancement factors
$\Gamma_k$ and $\Gamma_c$. Particula converts a dimensionless collision
kernel $H$ to a pair coefficient in m³/s using

$$
K = H \frac{f_r R^3}{m_r} \frac{\Gamma_k^2}{\Gamma_c}.
$$

The choice of $H$ specifies the transition between limiting regimes and the
range of electrostatic interactions that the fit represents. Use this pair
coefficient with the relevant number concentrations to estimate encounter
rates. The pair calculation alone does not evolve a charge distribution.

[Compare collision kernels for a negatively charged ion and positively charged particles](../../studies/examples/particle-processes/coagulation/charged-particles/ionparticle-coagulation.py).
