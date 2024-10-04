# DimensionlessCoagulation

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / DimensionlessCoagulation

> Auto-generated documentation for [particula.util.dimensionless_coagulation](https://github.com/uncscode/particula/blob/main/particula/util/dimensionless_coagulation.py) module.

## DimensionlessCoagulation

[Show source in dimensionless_coagulation.py:12](https://github.com/uncscode/particula/blob/main/particula/util/dimensionless_coagulation.py#L12)

dimensionless coagulation

#### Signature

```python
class DimensionlessCoagulation(DKn):
    def __init__(self, dkn_val=None, coag_approx="hardsphere", **kwargs): ...
```

#### See also

- [DiffusiveKnudsen](./diffusive_knudsen.md#diffusiveknudsen)

### DimensionlessCoagulation().coag_full

[Show source in dimensionless_coagulation.py:58](https://github.com/uncscode/particula/blob/main/particula/util/dimensionless_coagulation.py#L58)

Retrun the dimensioned coagulation kernel

#### Signature

```python
def coag_full(self): ...
```

### DimensionlessCoagulation().coag_less

[Show source in dimensionless_coagulation.py:43](https://github.com/uncscode/particula/blob/main/particula/util/dimensionless_coagulation.py#L43)

Return the dimensionless coagulation kernel.

#### Signature

```python
def coag_less(self): ...
```



## full_coag

[Show source in dimensionless_coagulation.py:102](https://github.com/uncscode/particula/blob/main/particula/util/dimensionless_coagulation.py#L102)

Return the dimensioned coagulation kernel

#### Signature

```python
def full_coag(**kwargs): ...
```



## less_coag

[Show source in dimensionless_coagulation.py:73](https://github.com/uncscode/particula/blob/main/particula/util/dimensionless_coagulation.py#L73)

Return the dimensionless coagulation kernel.

The dimensionless coagulation kernel is defined as
a function of the diffusive knudsen number; for more info,
please see the documentation of the respective function:
    - particula.util.diffusive_knudsen.diff_knu(**kwargs)

#### Examples

```
>>> from particula import u
>>> from particula.util.dimensionless_coagulation import less_coag
>>> # only for hardsphere coagulation for now
>>> # with only one radius
>>> less_coag(radius=1e-9)
<Quantity(147.877572, 'dimensionless')>
>>> # with two radii
>>> less_coag(radius=1e-9, other_radius=1e-8)
<Quantity(18.4245966, 'dimensionless')>
>>> # with two radii and charges
>>> less_coag(
... radius=1e-9, other_radius=1e-8, charge=1, other_charge=-1
... )
<Quantity(22.0727435, 'dimensionless')>

#### Signature

```python
def less_coag(**kwargs): ...
```
