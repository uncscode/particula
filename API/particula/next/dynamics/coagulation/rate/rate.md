# Rate

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Rate

> Auto-generated documentation for [particula.next.dynamics.coagulation.rate](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/rate.py) module.

## continuous_gain

[Show source in rate.py:120](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/rate.py#L120)

Calculate the coagulation gain rate, via the integration method.

#### Arguments

radius : The radius of the particles.
concentration : The distribution of particles.
kernel : The coagulation kernel.

#### Returns

The coagulation gain rate.

#### References

----------
- This equation necessitates the use of a for-loop due to the
convoluted use of different radii at different stages. This is the
most expensive step of all coagulation calculations. Using
`RectBivariateSpline` accelerates this significantly.
- Note, to estimate the kernel and distribution at
(other_radius**3 - some_radius**3)*(1/3)
we use interporlation techniques.
- Seinfeld, J. H., & Pandis, S. (2016). Atmospheric chemistry and
physics, Chapter 13 Equations 13.61

#### Signature

```python
def continuous_gain(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]: ...
```



## continuous_loss

[Show source in rate.py:96](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/rate.py#L96)

Calculate the coagulation loss rate, via the integration method.

#### Arguments

radius : The radius of the particles.
concentration : The distribution of particles.
kernel : The coagulation kernel.

#### Returns

The coagulation loss rate.

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61

#### Signature

```python
def continuous_loss(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]: ...
```



## discrete_gain

[Show source in rate.py:40](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/rate.py#L40)

Calculate the coagulation gain rate, via the integration method, by
converting to a continuous distribution.

#### Arguments

radius : The radius of the particles.
concentration : The distribution of particles.
kernel : The coagulation kernel.

#### Returns

The coagulation gain rate.

#### References

----------
- This equation necessitates the use of a for-loop due to the
convoluted use of different radii at different stages. This is the
most expensive step of all coagulation calculations. Using
`RectBivariateSpline` accelerates this significantly.
- Note, to estimate the kernel and distribution at
(other_radius**3 - some_radius**3)*(1/3) we use interporlation techniques.
- Seinfeld, J. H., & Pandis, S. (2016). Atmospheric chemistry and
physics, Chapter 13 Equations 13.61

#### Signature

```python
def discrete_gain(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]: ...
```



## discrete_loss

[Show source in rate.py:19](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/rate.py#L19)

Calculate the coagulation loss rate, via the summation method.

#### Arguments

concentraiton : The distribution of particles.
kernel : The coagulation kernel.

#### Returns

The coagulation loss rate.

#### References

Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
physics, Chapter 13 Equations 13.61

#### Signature

```python
def discrete_loss(
    concentration: Union[float, NDArray[np.float64]], kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```
