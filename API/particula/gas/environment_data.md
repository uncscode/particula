# particula.gas.environment_data

*module*

Provide per-box thermodynamic state for gas-phase simulations.

This module defines :class:`EnvironmentData`, a mutable container for the
temperature, pressure, and saturation-ratio fields associated with one or
more simulation boxes. Separating this state from gas-species data lets
multi-box workflows manage shared thermodynamic conditions independently.

`EnvironmentData` is a constructor-validated CPU-side container in
``particula.gas.environment_data`` and is also exported from
``particula.gas`` for package-level imports. Warp transfer helpers such as
``to_warp_environment_data`` and ``from_warp_environment_data`` live in
``particula.gpu.conversion`` and enable CPU↔GPU environment round trips.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L1-L1)

## particula.gas.environment_data.EnvironmentData

*class* · *dataclass*

```python
class EnvironmentData
```

Mutable thermodynamic environment data for one or more boxes.

This class stores box-level temperature and pressure together with a
per-box, per-species saturation-ratio array. The temperature array is
the source of truth for the box axis, so derived properties such as
``n_boxes`` follow ``temperature.shape[0]``.

**Attributes**

- `temperature` (`NDArray[np.float64]`) — Box temperatures in kelvin. Shape: ``(n_boxes,)``.
- `pressure` (`NDArray[np.float64]`) — Box pressures in pascals. Shape: ``(n_boxes,)``.
- `saturation_ratio` (`NDArray[np.float64]`) — Per-box, per-species saturation ratios. Shape: ``(n_boxes, n_species)``.

**Raises**

- `ValueError` — If any field has invalid dimensionality, shape,
finiteness, or physical bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L21-L215)

### particula.gas.environment_data.EnvironmentData.pressure

*attribute* · *instance attribute*

```python
pressure: NDArray[np.float64] = self._coerce_float64_array(pressure, field_name='pressure')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L63-L66)

### particula.gas.environment_data.EnvironmentData.saturation_ratio

*attribute* · *instance attribute*

```python
saturation_ratio: NDArray[np.float64] = self._coerce_float64_array(saturation_ratio, field_name='saturation_ratio')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L67-L70)

### particula.gas.environment_data.EnvironmentData.temperature

*attribute* · *instance attribute*

```python
temperature: NDArray[np.float64] = self._coerce_float64_array(temperature, field_name='temperature')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L59-L62)

### particula.gas.environment_data.EnvironmentData.n_boxes

*property*

```python
n_boxes: int
```

Return the number of simulation boxes.

**Returns**

- (`int`) — Number of boxes represented by ``temperature.shape[0]``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L86-L92)

### particula.gas.environment_data.EnvironmentData.__init__

*method*

```python
def __init__(temperature: ArrayLike, pressure: ArrayLike, saturation_ratio: ArrayLike) -> None
```

Construct and normalize per-box environment data.

**Parameters**

- `temperature` (`ArrayLike`) — Box temperatures in kelvin with shape ``(n_boxes,)``.
- `pressure` (`ArrayLike`) — Box pressures in pascals with shape ``(n_boxes,)``.
- `saturation_ratio` (`ArrayLike`) — Per-box, per-species saturation ratios with shape ``(n_boxes, n_species)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L45-L71)

### particula.gas.environment_data.EnvironmentData.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the normalized field contract.

**Raises**

- `ValueError` — If any field fails dimensionality, shape, finiteness,
or physical-bound validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L73-L83)

### particula.gas.environment_data.EnvironmentData._coerce_float64_array

*method* · *staticmethod*

```python
def _coerce_float64_array(values: object, *, field_name: str) -> NDArray[np.float64]
```

Convert array-like input to a ``np.float64`` NumPy array.

**Parameters**

- `values` (`object`) — Array-like input supplied for a dataclass field.
- `field_name` (`str`) — Name of the field being coerced for error reporting.

**Returns**

- (`NDArray[np.float64]`) — NumPy array with ``np.float64`` dtype.

**Raises**

- `ValueError` — If ``values`` cannot be converted to a float64 array.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L107-L130)

### particula.gas.environment_data.EnvironmentData._validate_dimensionality

*method*

```python
def _validate_dimensionality() -> None
```

Validate per-field dimensionality requirements.

**Raises**

- `ValueError` — If ``temperature`` or ``pressure`` is not 1D, or if
``saturation_ratio`` is not 2D.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L132-L154)

### particula.gas.environment_data.EnvironmentData._validate_finite_values

*method*

```python
def _validate_finite_values() -> None
```

Require all numeric fields to be finite.

**Raises**

- `ValueError` — If any field contains ``NaN`` or infinite values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L186-L199)

### particula.gas.environment_data.EnvironmentData._validate_physical_bounds

*method*

```python
def _validate_physical_bounds() -> None
```

Require positive thermodynamic state and nonnegative saturation.

**Raises**

- `ValueError` — If temperature or pressure is nonpositive, or if any
saturation ratio is negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L201-L215)

### particula.gas.environment_data.EnvironmentData._validate_shapes

*method*

```python
def _validate_shapes() -> None
```

Validate shared box-count consistency after ndim checks.

**Raises**

- `ValueError` — If no boxes are provided, or if ``pressure`` or
``saturation_ratio`` does not share the same leading box
dimension as ``temperature``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L156-L184)

### particula.gas.environment_data.EnvironmentData.copy

*method*

```python
def copy() -> EnvironmentData
```

Create an independent copy of the environment state.

**Returns**

- `New` (`EnvironmentData`) — class:`EnvironmentData` instance with copied temperature,
- (`EnvironmentData`) — pressure, and saturation-ratio arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L94-L105)
