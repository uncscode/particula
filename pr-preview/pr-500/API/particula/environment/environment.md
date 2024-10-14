# Environment

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Environment

> Auto-generated documentation for [particula.environment](https://github.com/uncscode/particula/blob/main/particula/environment.py) module.

## Environment

[Show source in environment.py:67](https://github.com/uncscode/particula/blob/main/particula/environment.py#L67)

creating the environment class

For now, the environment class takes properties such as
temperature and pressure to calculate derived properties
such as viscosity and mean free path.

#### Signature

```python
class Environment(SharedProperties):
    def __init__(self, **kwargs): ...
```

#### See also

- [SharedProperties](#sharedproperties)

### Environment().dynamic_viscosity

[Show source in environment.py:110](https://github.com/uncscode/particula/blob/main/particula/environment.py#L110)

Returns the dynamic viscosity in Pa*s.

#### Signature

```python
def dynamic_viscosity(self): ...
```

### Environment().mean_free_path

[Show source in environment.py:120](https://github.com/uncscode/particula/blob/main/particula/environment.py#L120)

Returns the mean free path in m.

#### Signature

```python
def mean_free_path(self): ...
```

### Environment().water_vapor_concentration

[Show source in environment.py:131](https://github.com/uncscode/particula/blob/main/particula/environment.py#L131)

Returns the water vapor concentration in kg/m^3.

#### Signature

```python
def water_vapor_concentration(self): ...
```



## SharedProperties

[Show source in environment.py:41](https://github.com/uncscode/particula/blob/main/particula/environment.py#L41)

 a hidden class for sharing properties like
coagulation_approximation

#### Signature

```python
class SharedProperties:
    def __init__(self, **kwargs): ...
```

### SharedProperties().dilution_rate_coefficient

[Show source in environment.py:58](https://github.com/uncscode/particula/blob/main/particula/environment.py#L58)

get the dilution rate coefficient

#### Signature

```python
def dilution_rate_coefficient(self): ...
```
