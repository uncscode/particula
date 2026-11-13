# particula.particles.properties.lognormal_size_distribution

*module*

Lognormal size distribution properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py#L1-L1)

## particula.particles.properties.lognormal_size_distribution.get_lognormal_pdf_distribution

*function*

```python
def get_lognormal_pdf_distribution(x_values: NDArray[np.float64], mode: NDArray[np.float64], geometric_standard_deviation: NDArray[np.float64], number_of_particles: NDArray[np.float64]) -> NDArray[np.float64]
```

Compute a lognormal probability density function (PDF) for given modes.

This function superimposes multiple lognormal PDFs, each with its own mode,
geometric standard deviation, and particle count. It then returns their sum
across the provided x_values. Mathematically, for each mode i:

- PDFᵢ(x) = (1 / [x · ln(gsdᵢ) · √(2π)]) ×
             exp(- [ln(x) - ln(modeᵢ)]² / [2 · (ln(gsdᵢ))² ])

**Parameters**

- `x_values` (`NDArray[np.float64]`) — 1D array of the size points at which the PDF is evaluated.
- `mode` (`NDArray[np.float64]`) — Array of lognormal mode (scale) values for each mode.
- `geometric_standard_deviation` (`NDArray[np.float64]`) — Array of GSD values for each mode.
- `number_of_particles` (`NDArray[np.float64]`) — Number of particles in each mode.

**Returns**

- (`NDArray[np.float64]`) — 1D array of the total PDF values summed across all modes.

**Examples**

```py title="Example"
import numpy as np
import particula as par
x_vals = np.linspace(1e-9, 1e-6, 100)
par.particles.get_lognormal_pdf_distribution(
    x_values=x_vals,
    mode=np.array([5e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.5, 2.0]),
    number_of_particles=np.array([1e9, 5e9])
)
# Output: [...]
```

> **References**
>
> - [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
> - [Probability Density Function Wikipedia](https://en.wikipedia.org/wiki/Probability_density_function)
> - [Scipy Lognorm Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py#L16-L90)

## particula.particles.properties.lognormal_size_distribution.get_lognormal_pmf_distribution

*function*

```python
def get_lognormal_pmf_distribution(x_values: NDArray[np.float64], mode: NDArray[np.float64], geometric_standard_deviation: NDArray[np.float64], number_of_particles: NDArray[np.float64]) -> NDArray[np.float64]
```

Compute a lognormal probability mass function (PMF) for given modes.

This function first calculates the lognormal PDF using
get_lognormal_pdf_distribution(), then converts it to a PMF by
integrating (or summing) over x_values. The result reflects discrete mass
(probability) distribution across the given size points.

**Parameters**

- `x_values` (`NDArray[np.float64]`) — 1D array of size points at which the PMF is evaluated.
- `mode` (`NDArray[np.float64]`) — Array of lognormal mode (scale) values for each mode.
- `geometric_standard_deviation` (`NDArray[np.float64]`) — Array of GSD values for each mode.
- `number_of_particles` (`NDArray[np.float64]`) — Number of particles in each mode.

**Returns**

- (`NDArray[np.float64]`) — 1D array of the total PMF values summed across all modes.

**Examples**

```py title="Example"
import numpy as np
import particula as par
x_vals = np.linspace(1e-9, 1e-6, 100)
par.particles.get_lognormal_pmf_distribution(
    x_values=x_vals,
    mode=np.array([5e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.5, 2.0]),
    number_of_particles=np.array([1e9, 5e9])
)
# Output: [...]
```

> **References**
>
> - [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
> - [Probability Mass Function Wikipedia](https://en.wikipedia.org/wiki/Probability_mass_function)
> - [Scipy Lognorm Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py#L93-L163)

## particula.particles.properties.lognormal_size_distribution.get_lognormal_sample_distribution

*function*

```python
def get_lognormal_sample_distribution(mode: NDArray[np.float64], geometric_standard_deviation: NDArray[np.float64], number_of_particles: NDArray[np.float64], number_of_samples: int, upper_bound: float = np.inf, lower_bound: float = 0) -> NDArray[np.float64]
```

Generate random samples from a lognormal distribution for given modes.

This function uses scipy.stats.lognorm.rvs() to draw samples for each mode,
with a specified scale (mode) and shape (GSD). The total samples are then
combined according to the relative number of particles in each mode.

**Parameters**

- `mode` (`NDArray[np.float64]`) — Array of lognormal mode (scale) values for each mode.
- `geometric_standard_deviation` (`NDArray[np.float64]`) — Array of GSD values for each mode.
- `number_of_particles` (`NDArray[np.float64]`) — Number of particles for each mode.
- `number_of_samples` (`int`) — Total number of random samples to generate.

**Returns**

- (`NDArray[np.float64]`) — 1D array of sampled particle sizes, combining all modes.

**Examples**

```py title="Example"
import numpy as np
import particula as par
par.particles.get_lognormal_sample_distribution(
    mode=np.array([5e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.5, 2.0]),
    number_of_particles=np.array([1e9, 5e9]),
    number_of_samples=10_000
)
# Output: [...]
```

> **References**
>
> - [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
> - [Probability Density Function Wikipedia](https://en.wikipedia.org/wiki/Probability_density_function)
> - [Scipy Lognorm Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py#L166-L249)
