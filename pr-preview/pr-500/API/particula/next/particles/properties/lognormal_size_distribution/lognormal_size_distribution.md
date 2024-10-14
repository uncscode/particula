# Lognormal Size Distribution

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Lognormal Size Distribution

> Auto-generated documentation for [particula.next.particles.properties.lognormal_size_distribution](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/lognormal_size_distribution.py) module.

## lognormal_pdf_distribution

[Show source in lognormal_size_distribution.py:12](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/lognormal_size_distribution.py#L12)

Probability Density Function for the lognormal distribution of particles
for varying modes, geometric standard deviations, and numbers of particles,
across a range of x_values.

#### Arguments

- `x_values` - The size interval of the distribution.
- `mode` - Scales corresponding to the mode in lognormal for different
    modes.
- `geometric_standard_deviation` - Geometric standard deviations of the
    distribution for different modes.
- `number_of_particles` - Number of particles for each mode.

#### Returns

The normalized lognormal distribution of the particles, summed
across all modes.

#### References

- [Log-normal Distribution Wikipedia](
    https://en.wikipedia.org/wiki/Log-normal_distribution)
 - [Probability Density Function Wikipedia](
    https://en.wikipedia.org/wiki/Probability_density_function)

#### Signature

```python
def lognormal_pdf_distribution(
    x_values: NDArray[np.float64],
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## lognormal_pmf_distribution

[Show source in lognormal_size_distribution.py:67](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/lognormal_size_distribution.py#L67)

Probability Mass function for lognormal distribution of particles for
varying modes, geometric standard deviations, and numbers of particles,
across a range of x_values.

#### Arguments

- `x_values` - The size interval of the distribution.
- `mode` - Scales corresponding to the mode in lognormal for different
    modes.
- `geometric_standard_deviation` - Geometric standard deviations of the
    distribution for different modes.
- `number_of_particles` - Number of particles for each mode.

#### Returns

The normalized lognormal distribution of the particles, summed
across all modes.

#### References

- [Log-normal Distribution Wikipedia](
    https://en.wikipedia.org/wiki/Log-normal_distribution)
- [Probability Mass Function Wikipedia](
    https://en.wikipedia.org/wiki/Probability_mass_function)

#### Signature

```python
def lognormal_pmf_distribution(
    x_values: NDArray[np.float64],
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## lognormal_sample_distribution

[Show source in lognormal_size_distribution.py:117](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/lognormal_size_distribution.py#L117)

Sample a Probability Density Function for the lognormal distribution.

Samples a set of samples (particle) to represent the lognormal distribution
for varying modes, geometric standard deviations, and numbers of particles,
across a range of x_values.

#### Arguments

- `mode` - Scales corresponding to the mode in lognormal for different
    modes.
- `geometric_standard_deviation` - Geometric standard deviations of the
    distribution for different modes.
- `number_of_particles` - Number of particles for each mode.
- `number_of_samples` - Number of samples to generate.

#### Returns

The normalized lognormal distribution of the particles, summed
across all modes.

#### References

- [Log-normal Distribution Wikipedia](
    https://en.wikipedia.org/wiki/Log-normal_distribution)
 - [Probability Density Function Wikipedia](
    https://en.wikipedia.org/wiki/Probability_density_function)

#### Signature

```python
def lognormal_sample_distribution(
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
    number_of_samples: int,
) -> NDArray[np.float64]: ...
```
