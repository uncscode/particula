# Particle Property

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Lagrangian](./index.md#lagrangian) / Particle Property

> Auto-generated documentation for [particula.lagrangian.particle_property](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py) module.

## friction_factor_wrapper

[Show source in particle_property.py:70](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L70)

Calculate the friction factor for a given radius, temperature, and
pressure.

This function wraps several underlying calculations related to
dynamic viscosity, mean free path, Knudsen number, and slip correction
factor to compute the particle friction factor.

#### Arguments

- `radius_meter` - A tensor representing the radius of the
sphere(s) in meters. Can be a scalar or a vector.
- `temperature_kelvin` - A tensor of the temperature in Kelvin.
- `pressure_pascal` - A tensor of the pressure in Pascals.

#### Returns

- `torch.Tensor` - A tensor of the same shape as `radius_meter`,
representing the particle friction factor.

#### Signature

```python
def friction_factor_wrapper(
    radius_meter: torch.Tensor, temperature_kelvin: float, pressure_pascal: float
) -> torch.Tensor: ...
```



## generate_particle_masses

[Show source in particle_property.py:122](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L122)

Generate an array of particle masses based on a log-normal distribution of
particle radii and a given density.

#### Arguments

- `mean_radius` *float* - Mean radius of the particles. The units are
    specified by `radius_input_units`.
- `std_dev_radius` *float* - Standard deviation of the particle radii. The
    units are specified by `radius_input_units`.
- `density` *torch.Tensor* - Density of the particles in kg/m^3.
- `num_particles` *int* - Number of particles to generate.
- `radius_input_units` *str, optional* - Units of `mean_radius` and
    `std_dev_radius`. Defaults to 'nm' (nanometers).

#### Returns

- `torch.Tensor` - A tensor of particle masses in kg, corresponding to each
    particle.

#### Raises

- `ValueError` - If `mean_radius` or `std_dev_radius` are non-positive.

#### Signature

```python
def generate_particle_masses(
    mean_radius: float,
    std_dev_radius: float,
    density: torch.Tensor,
    num_particles: int,
    radius_input_units: str = "nm",
) -> torch.Tensor: ...
```



## mass_calculation

[Show source in particle_property.py:44](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L44)

Calculate the mass of a sphere given its radius and density using the
formula for the volume of a sphere.

This function assumes a uniform density and spherical shape to compute the
mass based on the mass-density relationship:
Volume = Mass / Density. The volume of a sphere is given by
(4/3) * pi * radius^3.

#### Arguments

- radius (torch.Tensor): A tensor containing the radius of the sphere(s).
    Can be a scalar or a vector.
- density (torch.Tensor): A tensor containing the density of the sphere(s).
    Can be a scalar or a vector.

#### Returns

- `torch.Tensor` - A tensor of the same shape as `radius` and `density`
    representing the mass of the sphere(s).

#### Signature

```python
def mass_calculation(radius: torch.Tensor, density: torch.Tensor) -> torch.Tensor: ...
```



## nearest_match

[Show source in particle_property.py:254](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L254)

Perform nearest neighbor interpolation (on torch objects) to find y-values
corresponding to new x-values. The function identifies the nearest x-value
for each value in x_new and returns the corresponding y-value.

#### Arguments

- `x_values` *torch.Tensor* - The original x-values of shape (n,).
- `y_values` *torch.Tensor* - The original y-values of shape (n,).
    Each y-value corresponds to an x-value.
- `x_new` *torch.Tensor* - The new x-values for which y-values are to be
    interpolated, of shape (m,).

#### Returns

- `torch.Tensor` - The interpolated y-values of shape (m,). Each value
    corresponds to the nearest match from x_values.

#### Signature

```python
def nearest_match(
    x_values: torch.Tensor, y_values: torch.Tensor, x_new: torch.Tensor
) -> torch.Tensor: ...
```



## radius_calculation

[Show source in particle_property.py:12](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L12)

Calculate the radius of a sphere given its mass and density using the
formula for the volume of a sphere.

This function assumes a uniform density and spherical shape to compute the
radius based on the mass-density relationship:
Volume = Mass / Density. The volume of a sphere is given by
(4/3) * pi * radius^3.

#### Arguments

- mass (torch.Tensor): A tensor containing the mass of the sphere(s). Can
    be a scalar or a vector.
- density (torch.Tensor): A tensor containing the density of the sphere(s).
    Can be a scalar or a vector.

#### Returns

- `torch.Tensor` - A tensor of the same shape as `mass` and `density`
    representing the radius of the sphere(s).

#### Notes

- The function supports broadcasting, so `mass` and `density` can be of
    different shapes, as long as they are broadcastable to a common shape.
- Units of mass and density should be consistent to obtain a radius in
    meaningful units.

#### Signature

```python
def radius_calculation(mass: torch.Tensor, density: torch.Tensor) -> torch.Tensor: ...
```



## random_thermal_velocity

[Show source in particle_property.py:221](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L221)

Generate a random thermal velocity for each particle.

#### Arguments

- `temperature_kelvin` *torch.Tensor* - Temperature of the fluid in Kelvin.
- `mass_kg` *torch.Tensor* - Mass of the particle in kilograms.
- `number_of_particles` *int* - Number of particles.

#### Returns

- `torch.Tensor` - Thermal speed of the particle in meters per second.

#### Signature

```python
def random_thermal_velocity(
    temperature_kelvin: float,
    mass_kg: torch.Tensor,
    number_of_particles: int,
    t_type=torch.float,
    random_seed: int = 0,
) -> torch.Tensor: ...
```



## speed

[Show source in particle_property.py:206](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L206)

Calculate the speed of a particle.

#### Arguments

- `velocity` *torch.Tensor* - Velocity of the particle.

#### Returns

- `torch.Tensor` - Speed of the particle.

#### Signature

```python
def speed(velocity: torch.Tensor) -> torch.Tensor: ...
```



## thermal_speed

[Show source in particle_property.py:173](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_property.py#L173)

Calculate the thermal speed of a particle based on its temperature and
mass.

The thermal speed is computed using the formula: sqrt(8 * k * T / (pi * m))
where k is the Boltzmann constant, T is the temperature in Kelvin, and m is
the particle mass in kilograms.

#### Arguments

- `temperature_kelvin` *float* - Temperature of the environment in Kelvin.
- `mass_kg` *torch.Tensor* - Mass of the particle(s) in kilograms.
    Can be a scalar or a vector.

#### Returns

- `torch.Tensor` - The thermal speed of the particle(s) in meters per second

#### Raises

- `ValueError` - If `temperature_kelvin` is less than or equal to zero or
if any element of `mass_kg` is non-positive.

#### Signature

```python
def thermal_speed(temperature_kelvin: float, mass_kg: torch.Tensor) -> torch.Tensor: ...
```
