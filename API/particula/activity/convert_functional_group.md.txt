# particula.activity.convert_functional_group

*module*

OH equivalent for the oxygen to carbon ratio and molar mass ratio.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L1-L1)

## particula.activity.convert_functional_group.FloatArray

*attribute* · *module attribute*

```python
FloatArray = Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L16-L16)

## particula.activity.convert_functional_group.SUPPORTED_FUNCTIONAL_GROUPS

*attribute* · *module attribute*

```python
SUPPORTED_FUNCTIONAL_GROUPS = (None, *tuple(_FUNCTIONAL_GROUP_ADJUSTMENTS.keys()))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L25-L28)

## particula.activity.convert_functional_group._FUNCTIONAL_GROUP_ADJUSTMENTS

*attribute* · *module attribute*

```python
_FUNCTIONAL_GROUP_ADJUSTMENTS = {'alcohol': (1.0, 16.0), 'ether': (1.0, 16.0), 'carboxylic_acid': (2.0, 45.0)}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L18-L22)

## particula.activity.convert_functional_group._normalize_functional_group

*function*

```python
def _normalize_functional_group(functional_group: Optional[Union[str, Sequence[str]]]) -> Optional[str]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L31-L47)

## particula.activity.convert_functional_group.convert_to_oh_equivalent

*function*

```python
def convert_to_oh_equivalent(oxygen2carbon: FloatArray, molar_mass_ratio: FloatArray, functional_group: Optional[Union[str, Sequence[str]]] = None) -> Tuple[FloatArray, FloatArray]
```

Convert oxygen and molar mass ratios for BAT functional groups.

**Parameters**

- `oxygen2carbon` (`FloatArray`) — The oxygen to carbon ratio.
- `molar_mass_ratio` (`FloatArray`) — The molar mass ratio of water to organic matter.
- `functional_group` (`Optional[Union[str, Sequence[str]]]`) (default: `None`) — Optional functional group requiring adjustments. Supported values are ``None``, ``"alcohol"``, ``"ether"``, and ``"carboxylic_acid"``.

**Returns**

- (`FloatArray`) — Tuple containing the converted oxygen2carbon and molar_mass_ratio.
- (`FloatArray`) — Scalar inputs return scalars, array inputs return ``np.ndarray``.

**Raises**

- `ValueError` — If ``functional_group`` is not one of the supported values.

**Examples**

```pycon
>>> from particula.activity.convert_functional_group import (
...     convert_to_oh_equivalent,
... )
>>> convert_to_oh_equivalent(
...     oxygen2carbon=0.3,
...     molar_mass_ratio=0.1,
...     functional_group="carboxylic_acid",
... )
(2.3, 45.1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L50-L97)
