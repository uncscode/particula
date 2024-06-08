# Mie Angular

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Mie Angular

> Auto-generated documentation for [particula.data.process.mie_angular](https://github.com/Gorkowski/particula/blob/main/particula/data/process/mie_angular.py) module.

## assign_scattering_thetas

[Show source in mie_angular.py:110](https://github.com/Gorkowski/particula/blob/main/particula/data/process/mie_angular.py#L110)

Assigns scattering angles and efficiencies based on the z-axis position
within the CAPS instrument.

Parameters
----------
alpha : float
    Forward scattering angle in radians.
beta : float
    Backward scattering angle in radians.
q_mie : float
    The Mie scattering efficiency.
z_position : Union[float, np.float64]
    The position along the z-axis (in cm).
integrate_sphere_diameter_cm : float
    The diameter of the integrating sphere (in cm).

Returns
-------
Tuple[float, float, float]
    A tuple containing the forward scattering angle (theta1), backward
    scattering angle (theta2), and the ideal scattering efficiency
    (qsca_ideal) for the given z-axis position.

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

[Show source in mie_angular.py:68](https://github.com/Gorkowski/particula/blob/main/particula/data/process/mie_angular.py#L68)

Calculates forward and backward scattering angles for a given position
along the z-axis within the CAPS instrument geometry.

Parameters
----------
z_position : Union[float, np.float64]
    The position along the z-axis (in cm).
integrate_sphere_diameter_cm : float
    The diameter of the integrating sphere (in cm).
tube_diameter_cm : float
    The diameter of the sample tube (in cm).

Returns
-------
Tuple[float, float]
    A tuple containing the forward (alpha) and backward (beta)
    scattering angles in radians.

#### Signature

```python
def calculate_scattering_angles(
    z_position: Union[float, np.float64],
    integrate_sphere_diameter_cm: float,
    tube_diameter_cm: float,
) -> Tuple[float, float]: ...
```



## discretize_scattering_angles

[Show source in mie_angular.py:14](https://github.com/Gorkowski/particula/blob/main/particula/data/process/mie_angular.py#L14)

Discretizes and caches the scattering function for a spherical particle
with specified material properties and size. This function aims to optimize
the performance of scattering calculations by caching results for
frequently used parameters, reducing the need for repeated calculations.

Parameters
----------
m_sphere : Union[complex, float]
    The complex or real refractive index of the particle.
wavelength : float
    The wavelength of the incident light in nanometers (nm).
diameter : Union[float, np.float64]
    The diameter of the particle in nanometers (nm).
min_angle : float, optional
    The minimum scattering angle in degrees to be considered in the
    calculation. Defaults to 0.
max_angle : float, optional
    The maximum scattering angle in degrees to be considered in the
    calculation. Defaults to 180.
angular_resolution : float, optional
    The resolution in degrees between calculated scattering angles.
    Defaults to 1.

Returns
-------
Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
A tuple containing numpy arrays for the scattering function measurements:
    - measure: The scattering intensity as a function of angle.
    - parallel: The scattering intensity for parallel polarization.
    - perpendicular: The scattering intensity for
        perpendicular polarization.
    - unpolarized: The unpolarized scattering intensity.

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
