# particula.activity.ratio

*module*

Molar mass conversions with validation and array support.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L1-L1)

## particula.activity.ratio.FloatArray

*attribute* · *module attribute*

```python
FloatArray = Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L10-L10)

## particula.activity.ratio.from_molar_mass_ratio

*function*

```python
def from_molar_mass_ratio(molar_mass_ratio: Union[float, Iterable[float], NDArray[np.float64]], other_molar_mass: float = 18.01528) -> FloatArray
```

Convert molar mass ratio to organic molar mass.

**Parameters**

- `molar_mass_ratio` (`Union[float, Iterable[float], NDArray[np.float64]]`) — Molar mass ratio (MW water / MW organic). Must be positive.
- `other_molar_mass` (`float`) (default: `18.01528`) — Reference molar mass (default water, 18.01528). Must be positive.

**Returns**

- (`FloatArray`) — Organic molar mass. Float for scalar input; ndarray for list/array
- (`FloatArray`) — input.

**Raises**

- `ValueError` — If molar_mass_ratio or other_molar_mass is not positive.

**Examples**

```pycon
>>> from_molar_mass_ratio(0.5)
9.00764
>>> from_molar_mass_ratio([0.5, 1.0])
array([ 9.00764, 18.01528])
>>> from_molar_mass_ratio(np.array([0.25]))
array([4.50382])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L46-L85)

## particula.activity.ratio.to_molar_mass_ratio

*function*

```python
def to_molar_mass_ratio(molar_mass: Union[float, Iterable[float], NDArray[np.float64]], other_molar_mass: float = 18.01528) -> FloatArray
```

Convert to molar mass ratio (MW water / MW organic).

**Parameters**

- `molar_mass` (`Union[float, Iterable[float], NDArray[np.float64]]`) — Molar mass of the organic compound. Must be positive.
- `other_molar_mass` (`float`) (default: `18.01528`) — Reference molar mass (default water, 18.01528). Must be positive.

**Returns**

- (`FloatArray`) — Molar mass ratio. Float for scalar input; ndarray for list/array input.

**Raises**

- `ValueError` — If molar_mass or other_molar_mass is not positive.

**Examples**

```pycon
>>> to_molar_mass_ratio(36.03)
0.5
>>> to_molar_mass_ratio([36.03, 18.01528])
array([0.5, 1. ])
>>> to_molar_mass_ratio(np.array([72.06]))
array([0.25])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L13-L43)
