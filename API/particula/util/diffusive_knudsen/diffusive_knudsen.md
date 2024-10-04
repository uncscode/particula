# DiffusiveKnudsen

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / DiffusiveKnudsen

> Auto-generated documentation for [particula.util.diffusive_knudsen](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py) module.

## DiffusiveKnudsen

[Show source in diffusive_knudsen.py:14](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L14)

A class for Diff..Knu

#### Signature

```python
class DiffusiveKnudsen(CoulombEnhancement):
    def __init__(self, density=1000, other_density=None, **kwargs): ...
```

#### See also

- [CoulombEnhancement](./coulomb_enhancement.md#coulombenhancement)

### DiffusiveKnudsen().get_celimits

[Show source in diffusive_knudsen.py:73](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L73)

get coag enh limits

#### Signature

```python
def get_celimits(self): ...
```

### DiffusiveKnudsen().get_ces

[Show source in diffusive_knudsen.py:67](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L67)

get coulomb enhancement parameters

#### Signature

```python
def get_ces(self): ...
```

### DiffusiveKnudsen().get_diff_knu

[Show source in diffusive_knudsen.py:80](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L80)

calculate it

#### Signature

```python
def get_diff_knu(self): ...
```

### DiffusiveKnudsen().get_red_frifac

[Show source in diffusive_knudsen.py:51](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L51)

get the reduced friction factor

#### Signature

```python
def get_red_frifac(self): ...
```

### DiffusiveKnudsen().get_red_mass

[Show source in diffusive_knudsen.py:34](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L34)

get the reduced mass

#### Signature

```python
def get_red_mass(self): ...
```

### DiffusiveKnudsen().get_rxr

[Show source in diffusive_knudsen.py:45](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L45)

add two radii

#### Signature

```python
def get_rxr(self): ...
```



## celimits

[Show source in diffusive_knudsen.py:163](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L163)

get coag enh limits

#### Signature

```python
def celimits(**kwargs): ...
```



## ces

[Show source in diffusive_knudsen.py:156](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L156)

get the coulomb enhancement limits

#### Signature

```python
def ces(**kwargs): ...
```



## diff_knu

[Show source in diffusive_knudsen.py:94](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L94)

Diffusive Knudsen number.

The *diffusive* Knudsen number is different from Knudsen number.
Ratio of:
    - numerator: mean persistence of one particle
    - denominator: effective length scale of
        particle--particle Coulombic interaction

#### Examples

```
>>> from particula import u
>>> from particula.util.diffusive_knudsen import diff_knu
>>> # with only one radius
>>> diff_knu(radius=1e-9)
<Quantity(29.6799, 'dimensionless')>
>>> # with two radii
>>> diff_knu(radius=1e-9, other_radius=1e-8)
<Quantity(3.85387845, 'dimensionless')>
>>> # with radii and charges
>>> diff_knu(radius=1e-9, other_radius=1e-8, charge=-1, other_charge=1)
<Quantity(4.58204028, 'dimensionless')>
```

#### Arguments

radius          (float) [m]
other_radius    (float) [m]             (default: radius)
density         (float) [kg/m^3]        (default: 1000)
other_density   (float) [kg/m^3]        (default: density)
charge          (int)   [dimensionless] (default: 0)
other_charge    (int)   [dimensionless] (default: 0)
temperature     (float) [K]             (default: 298)

#### Returns

(float) [dimensionless]

#### Notes

this function uses the friction factor and
the coulomb enhancement calculations; for more information,
please see the documentation of the respective functions:
    - particula.util.friction_factor.frifac(**kwargs)
    - particula.util.coulomb_enhancement.cekl(**kwargs)
    - particula.util.coulomb_enhancement.cecl(**kwargs)

#### Signature

```python
def diff_knu(**kwargs): ...
```



## red_frifac

[Show source in diffusive_knudsen.py:149](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L149)

get the reduced friction factor

#### Signature

```python
def red_frifac(**kwargs): ...
```



## red_mass

[Show source in diffusive_knudsen.py:142](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L142)

get the reduced mass

#### Signature

```python
def red_mass(**kwargs): ...
```



## rxr

[Show source in diffusive_knudsen.py:170](https://github.com/uncscode/particula/blob/main/particula/util/diffusive_knudsen.py#L170)

add two radii

#### Signature

```python
def rxr(**kwargs): ...
```
