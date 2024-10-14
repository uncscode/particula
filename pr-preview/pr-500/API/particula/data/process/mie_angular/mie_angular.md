# Mie Angular

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Mie Angular

> Auto-generated documentation for [particula.data.process.mie_angular](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_angular.py) module.

## assign_scattering_thetas

[Show source in mie_angular.py:101](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_angular.py#L101)

Assign scattering angles and efficiencies based on the z-axis position
within the CAPS instrument.

#### Arguments

- `alpha` - The forward scattering angle in radians.
- `beta` - The backward scattering angle in radians.
- `q_mie` - The Mie scattering efficiency.
- `z_position` - The position along the z-axis in centimeters (cm).
- `integrate_sphere_diameter_cm` - The diameter of the integrating sphere
    in centimeters (cm).

#### Returns

Tuple:
- The forward scattering angle (theta1) in radians.
- The backward scattering angle (theta2) in radians.
- The ideal scattering efficiency (qsca_ideal) for the given z-axis
    position.

#### Signature

```python
def assign_scattering_thetas(
    alpha: float,
    beta: float,
    q_mie: float,
    z_position: Union[float, np.float64],
    integrate_sphere_diameter_cm: float,
) -> Tuple[float, float, float]: ...
```



## calculate_scattering_angles

[Show source in mie_angular.py:63](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_angular.py#L63)

Calculate forward and backward scattering angles for a given position
along the z-axis within the CAPS instrument geometry.

#### Arguments

- `z_position` - The position along the z-axis in centimeters (cm).
- `integrate_sphere_diameter_cm` - The diameter of the integrating sphere
    in centimeters (cm).
- `tube_diameter_cm` - The diameter of the sample tube in centimeters (cm).

#### Returns

Tuple:
- The forward scattering angle (alpha) in radians.
- The backward scattering angle (beta) in radians.

#### Signature

```python
def calculate_scattering_angles(
    z_position: Union[float, np.float64],
    integrate_sphere_diameter_cm: float,
    tube_diameter_cm: float,
) -> Tuple[float, float]: ...
```



## discretize_scattering_angles

[Show source in mie_angular.py:16](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_angular.py#L16)

Discretize and cache the scattering function for a spherical particle
with specified material properties and size.

This function optimizes the performance of scattering calculations by
caching results for frequently used parameters, thereby reducing the
need for repeated calculations.

#### Arguments

- `m_sphere` - The complex or real refractive index of the particle.
- `wavelength` - The wavelength of the incident light in nanometers (nm).
- `diameter` - The diameter of the particle in nanometers (nm).
- `min_angle` - The minimum scattering angle in degrees to be considered in
    the calculation. Defaults to 0.
- `max_angle` - The maximum scattering angle in degrees to be considered in
    the calculation. Defaults to 180.
- `angular_resolution` - The resolution in degrees between calculated
    scattering angles. Defaults to 1.

#### Returns

Tuple:
- `-` *measure* - The scattering intensity as a function of angle.
- `-` *parallel* - The scattering intensity for parallel polarization.
- `-` *perpendicular* - The scattering intensity for perpendicular
    polarization.
- `-` *unpolarized* - The unpolarized scattering intensity.

#### Signature

```python
@lru_cache(maxsize=100000)
def discretize_scattering_angles(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: Union[float, np.float64],
    min_angle: int = 0,
    max_angle: int = 180,
    angular_resolution: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
```
