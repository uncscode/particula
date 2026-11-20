# Air viscosity and molecular mean free path

## Air viscosity

The Sutherland relation connects dynamic viscosity to absolute temperature,

$$
\mu(T) = \mu_0 (T/T_0)^{3/2}(T_0+C)/(T+C).
$$

Here $\mu_0$ is the reference viscosity in Pa s, and $T$, $T_0$, and the Sutherland constant $C$ are in kelvin. Particula's air correlation uses an implementation-defined $C$. Its value at $T=T_0$ is $\mu_0$. [Evaluate the viscosity at 298.15 K](../../studies/examples/gas-transport/dynamic-viscosity.py).

## Molecular mean free path

For an ideal gas, viscosity $\mu$ in Pa s, pressure $p$ in Pa, molar mass $M$ in kg/mol and temperature $T$ in kelvin give

$$
\lambda = \frac{2\mu}{p}\sqrt{\frac{\pi R T}{8M}}.
$$

The result is a length in meters. At a fixed temperature and composition the mean free path varies inversely with pressure. [Calculate it for air at one atmosphere](../../studies/examples/gas-transport/mean-free-path.py).
