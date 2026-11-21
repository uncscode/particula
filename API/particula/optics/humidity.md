# particula.optics.humidity

*module*

Subsaturated homogeneous-particle water uptake for optical closure.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/humidity.py#L1-L1)

## particula.optics.humidity._real_arrays

*function*

```python
def _real_arrays(*values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/humidity.py#L8-L13)

## particula.optics.humidity.hydrate_index

*function*

```python
def hydrate_index(dry_index, water_index, water_to_dry_volume)
```

Mix the dry and water indices at a prescribed water/dry volume ratio.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/humidity.py#L74-L90)

## particula.optics.humidity.subsaturated_growth

*function*

```python
def subsaturated_growth(dry_radius, relative_humidity, kappa, kelvin_length)
```

Return the wet radius and water/dry volume ratio at equilibrium.

**Parameters**

- `dry_radius` — Positive dry radius [m].
- `relative_humidity` — Water saturation ratio in [0, 1).
- `kappa` — Nonnegative effective hygroscopicity.
- `kelvin_length` — Nonnegative radius-form Kelvin coefficient [m], determined by temperature, water density and surface tension. A zero length selects the flat-interface model.

**Returns**

- Wet radius [m] and dimensionless water/dry volume ratio.
- Inputs use NumPy broadcasting.

The model follows the liquid branch of the kappa-Köhler relation.
A prescribed dry solid population can use kappa zero. Phase selection and
uptake kinetics are outside this equilibrium calculation.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/humidity.py#L16-L71)
