# Input Handling

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Input Handling

> Auto-generated documentation for [particula.util.input_handling](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py) module.

## convert_units

[Show source in input_handling.py:77](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L77)

generic pint function to convert units

#### Arguments

old     [str | u.Quantity]
new     [str | u.Quantity]
value   (float) [optional]

#### Returns

multiplier     (float)

#### Notes

* If unit is correct, take to base units
* Throws ValueError if unit is wrong
* Assigning default base units to scalar input

#### Signature

```python
def convert_units(
    old: Union[str, u.Quantity],
    new: Union[str, u.Quantity],
    value: Optional[float] = None,
) -> float: ...
```



## in_acceleration

[Show source in input_handling.py:128](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L128)

#### Signature

```python
def in_acceleration(acc): ...
```



## in_area

[Show source in input_handling.py:133](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L133)

#### Signature

```python
def in_area(area): ...
```



## in_concentration

[Show source in input_handling.py:135](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L135)

#### Signature

```python
def in_concentration(conc): ...
```



## in_density

[Show source in input_handling.py:130](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L130)

#### Signature

```python
def in_density(density): ...
```



## in_gas_constant

[Show source in input_handling.py:134](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L134)

#### Signature

```python
def in_gas_constant(con): ...
```



## in_handling

[Show source in input_handling.py:44](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L44)

generic function to handle inputs

#### Arguments

value     (float)       [u.Quantity | dimensionless]
units     (u.Quantity)

#### Returns

value     (float)       [u.Quantity]

#### Notes

* If unit is correct, take to base units
* Throws ValueError if unit is wrong
* Assigning default base units to scalar input

#### Signature

```python
def in_handling(value, units: u.Quantity): ...
```



## in_latent_heat

[Show source in input_handling.py:137](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L137)

#### Signature

```python
def in_latent_heat(latheat): ...
```



## in_length

[Show source in input_handling.py:132](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L132)

#### Signature

```python
def in_length(length): ...
```



## in_mass

[Show source in input_handling.py:124](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L124)

#### Signature

```python
def in_mass(mass): ...
```



## in_molecular_weight

[Show source in input_handling.py:129](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L129)

#### Signature

```python
def in_molecular_weight(molw): ...
```



## in_pressure

[Show source in input_handling.py:123](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L123)

#### Signature

```python
def in_pressure(pres): ...
```



## in_radius

[Show source in input_handling.py:8](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L8)

Handles radius input

#### Arguments

radius    (float) [m | dimensionless]

#### Returns

radius    (float) [m]

#### Notes

* If unit is correct, take to base units in m
* Throws ValueError if unit is wrong
* Assigning m units to scalar input

#### Signature

```python
def in_radius(radius): ...
```



## in_scalar

[Show source in input_handling.py:131](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L131)

#### Signature

```python
def in_scalar(scalar): ...
```



## in_surface_tension

[Show source in input_handling.py:136](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L136)

#### Signature

```python
def in_surface_tension(surften): ...
```



## in_temperature

[Show source in input_handling.py:121](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L121)

#### Signature

```python
def in_temperature(temp): ...
```



## in_time

[Show source in input_handling.py:126](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L126)

#### Signature

```python
def in_time(time): ...
```



## in_velocity

[Show source in input_handling.py:127](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L127)

#### Signature

```python
def in_velocity(vel): ...
```



## in_viscosity

[Show source in input_handling.py:122](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L122)

#### Signature

```python
def in_viscosity(vis): ...
```



## in_volume

[Show source in input_handling.py:125](https://github.com/uncscode/particula/blob/main/particula/util/input_handling.py#L125)

#### Signature

```python
def in_volume(vol): ...
```
