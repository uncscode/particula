# CoagulationRate

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / CoagulationRate

> Auto-generated documentation for [particula.util.coagulation_rate](https://github.com/uncscode/particula/blob/main/particula/util/coagulation_rate.py) module.

## CoagulationRate

[Show source in coagulation_rate.py:8](https://github.com/uncscode/particula/blob/main/particula/util/coagulation_rate.py#L8)

A class to calculate the coagulation rate.

#### Arguments

- `distribution` *float* - The distribution of particles.
- `radius` *float* - The radius of the particles.
- `kernel` *float* - The kernel of the particles.

#### Signature

```python
class CoagulationRate:
    def __init__(self, distribution, radius, kernel, lazy=True): ...
```

### CoagulationRate().coag_gain

[Show source in coagulation_rate.py:76](https://github.com/uncscode/particula/blob/main/particula/util/coagulation_rate.py#L76)

Returns the coagulation gain rate

Equation:

gain_rate(other_radius) = (
    other_radius**2 *
    integral( # from some_radius=0 to other_radius/2**(1/3)
        kernel(some_radius, (other_radius**3-some_radius**3)*(1/3)*
        dist(some_radius) *
        dist((other_radius**3 - some_radius**3)*(1/3)) /
        (other_radius**3 - some_radius**3)*(2/3),
        some_radius
    )
)

Units:
    m**-4/s

This equation necessitates the use of a for-loop due to the
convoluted use of different radii at different stages.
This is the costliest step of all coagulation calculations.
Note, to estimate the kernel and distribution at
(other_radius**3 - some_radius**3)*(1/3)
we use interporlation techniques.

Using `RectBivariateSpline` accelerates this significantly.

#### Signature

```python
def coag_gain(self): ...
```

### CoagulationRate().coag_loss

[Show source in coagulation_rate.py:54](https://github.com/uncscode/particula/blob/main/particula/util/coagulation_rate.py#L54)

Returns the coagulation loss rate

Equation:

loss_rate(other_radius) = (
    dist(other_radius) *
    integral( # over all space
        kernel(radius, other_radius) *
        dist(radius),
        radius
    )

Units:

m**-4/s

#### Signature

```python
def coag_loss(self): ...
```

### CoagulationRate().coag_prep

[Show source in coagulation_rate.py:44](https://github.com/uncscode/particula/blob/main/particula/util/coagulation_rate.py#L44)

Repackage the parameters

#### Signature

```python
def coag_prep(self): ...
```
