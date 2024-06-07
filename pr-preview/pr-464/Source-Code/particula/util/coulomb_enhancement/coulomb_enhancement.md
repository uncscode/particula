# CoulombEnhancement

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / CoulombEnhancement

> Auto-generated documentation for [particula.util.coulomb_enhancement](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py) module.

## CoulombEnhancement

[Show source in coulomb_enhancement.py:17](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L17)

Calculate coulomb-related enhancements

#### Attributes

radius          (float) [m]
other_radius    (float) [m]             (default: radius)
charge          (float) [dimensionless] (default: 0)
other_charge    (float) [dimensionless] (default: 0)
temperature     (float) [K]             (default: 298)

#### Signature

```python
class CoulombEnhancement:
    def __init__(
        self,
        radius=None,
        other_radius=None,
        charge=0,
        other_charge=0,
        temperature=298,
        elementary_charge_value=ELEMENTARY_CHARGE_VALUE,
        electric_permittivity=ELECTRIC_PERMITTIVITY,
        boltzmann_constant=BOLTZMANN_CONSTANT,
        **kwargs
    ): ...
```

#### See also

- [BOLTZMANN_CONSTANT](../constants.md#boltzmann_constant)
- [ELECTRIC_PERMITTIVITY](../constants.md#electric_permittivity)
- [ELEMENTARY_CHARGE_VALUE](../constants.md#elementary_charge_value)

### CoulombEnhancement().coulomb_enhancement_continuum_limit

[Show source in coulomb_enhancement.py:124](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L124)

Coulombic coagulation enhancement continuum limit.

#### Arguments

radius          (float) [m]
other_radius    (float) [m]             (default: radius)
charge          (float) [dimensionless] (default: 0)
other_charge    (float) [dimensionless] (default: 0)
temperature     (float) [K]             (default: 298)

#### Returns

(float) [dimensionless]

#### Signature

```python
def coulomb_enhancement_continuum_limit(self): ...
```

### CoulombEnhancement().coulomb_enhancement_kinetic_limit

[Show source in coulomb_enhancement.py:103](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L103)

Coulombic coagulation enhancement kinetic limit.

#### Arguments

radius          (float) [m]
other_radius    (float) [m]             (default: radius)
charge          (float) [dimensionless] (default: 0)
other_charge    (float) [dimensionless] (default: 0)
temperature     (float) [K]             (default: 298)

#### Returns

(float) [dimensionless]

#### Signature

```python
def coulomb_enhancement_kinetic_limit(self): ...
```

### CoulombEnhancement().coulomb_potential_ratio

[Show source in coulomb_enhancement.py:77](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L77)

Calculates the Coulomb potential ratio.

#### Arguments

radius          (float) [m]
other_radius    (float) [m]             (default: radius)
charge          (int)   [dimensionless] (default: 0)
other_charge    (int)   [dimensionless] (default: 0)
temperature     (float) [K]             (default: 298)

#### Returns

(float) [dimensionless]

#### Signature

```python
def coulomb_potential_ratio(self): ...
```



## cecl

[Show source in coulomb_enhancement.py:159](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L159)

Calculate coulombic enhancement continuum limit

#### Signature

```python
def cecl(**kwargs): ...
```



## cekl

[Show source in coulomb_enhancement.py:153](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L153)

Calculate coulombic enhancement kinetic limit

#### Signature

```python
def cekl(**kwargs): ...
```



## coulomb_enhancement_all

[Show source in coulomb_enhancement.py:165](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L165)

Return all the values above in one call

#### Signature

```python
def coulomb_enhancement_all(**kwargs): ...
```



## cpr

[Show source in coulomb_enhancement.py:147](https://github.com/Gorkowski/particula/blob/main/particula/util/coulomb_enhancement.py#L147)

Calculate coulomb potential ratio

#### Signature

```python
def cpr(**kwargs): ...
```
