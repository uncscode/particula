# Wall Loss

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Wall Loss

> Auto-generated documentation for [particula.util.wall_loss](https://github.com/uncscode/particula/blob/main/particula/util/wall_loss.py) module.

## rectangle_wall_loss

[Show source in wall_loss.py:39](https://github.com/uncscode/particula/blob/main/particula/util/wall_loss.py#L39)

Calculate the wall loss coefficient, β₀, for a rectangular chamber.

Given the rate of wall diffusivity parameter (ktp_value), the particle
diffusion coefficient (diffusion_coefficient_value), and the terminal
settling velocity (settling_velocity_value), this function computes the
wall loss coefficient for a rectangular-prism chamber with specified
dimensions.

The wall loss coefficient is calculated based on the diffusion and
gravitational sedimentation in a rectangular chamber. It accounts for the
effect of chamber geometry on particle loss by considering the length (L),
width (W), and height (H) of the chamber.

#### Arguments

- `ktp_value` *float* - Rate of wall diffusivity parameter in units of
    inverse seconds (s^-1).
- `diffusion_coefficient_value` *float* - The particle diffusion
    coefficient in units of square meters per second (m^2/s).
- `settling_velocity_value` *float* - The terminal settling velocity of the
    particles, in units of meters per second (m/s).
- `dimension` *tuple* - A tuple of three floats representing the length (L)
    width (W), and height (H) of the rectangular chamber,
    in units of meters (m).

#### Returns

- `float` - The calculated wall loss coefficient (B0) for the rectangular
chamber.

Reference:
    The wall loss coefficient, β₀, is calculated using the following
    formula:
    $$
    eta_0 = (LWH)^{-1} (4H(L+W) \sqrt{k_t D}/\pi +
    v_g LW \coth{[(\pi v_g)/(4\sqrt{k_t D}})])
    $$

#### Signature

```python
def rectangle_wall_loss(
    ktp_value, diffusion_coefficient_value, settling_velocity_value, dimension
): ...
```



## spherical_wall_loss_coefficient

[Show source in wall_loss.py:12](https://github.com/uncscode/particula/blob/main/particula/util/wall_loss.py#L12)

Calculate the wall loss coefficient for a spherical chamber
approximation.

#### Arguments

- `ktp_value` - rate of the wall eddy diffusivity
- `diffusion_coefficient_value` - Particle diffusion coefficient.
- `settling_velocity_value` - Settling velocity of the particle.
- `chamber_radius` - Radius of the chamber.

#### Returns

The calculated wall loss coefficient for simple case.

#### Signature

```python
def spherical_wall_loss_coefficient(
    ktp_value, diffusion_coefficient_value, settling_velocity_value, chamber_radius
): ...
```



## wlc

[Show source in wall_loss.py:94](https://github.com/uncscode/particula/blob/main/particula/util/wall_loss.py#L94)

Calculate the wall loss coefficient.

#### Arguments

- `approximation` - The approximation method to use, e.g., "none",
"spherical", "rectangle"
- `ktp_value` - rate of the wall eddy diffusivity
- `diffusion_coefficient_value` - Particle diffusion coefficient.
- `settling_velocity_value` - Settling velocity of the particle.
- `dimension` - Radius of the chamber or tuple of rectangular dimensions.

#### Returns

The calculated wall loss coefficient.

#### Signature

```python
def wlc(
    approx="none",
    ktp_value=0.1 * u.s**-1,
    diffusion_coefficient_value=None,
    dimension=1 * u.m,
    settling_velocity_value=None,
    **kwargs
): ...
```
