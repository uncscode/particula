# particula.gas.gas_data_builder

*module*

Builder for creating validated :class:`GasData` instances.

Provides a fluent interface to set gas species fields with optional unit
conversion and automatic batch dimension handling.

**Examples**

Single-box with direct values::

    from particula.gas import GasDataBuilder
    import numpy as np

    gas = (
        GasDataBuilder()
        .set_names(["Water", "Ammonia"])
        .set_molar_mass([18, 17], units="g/mol")
        .set_concentration([1e-6, 5e-9], units="kg/m^3")
        .set_partitioning([True, True])
        .build()
    )

Multi-box with broadcasting::

    gas = (
        GasDataBuilder()
        .set_n_boxes(100)
        .set_names(["Water", "Ammonia"])
        .set_molar_mass([0.018, 0.017], units="kg/mol")
        .set_concentration([1e-6, 5e-9])  # Broadcast to 100 boxes
        .set_partitioning([True, True])
        .build()
    )

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L1-L1)

## particula.gas.gas_data_builder.GasDataBuilder

*class*

```python
class GasDataBuilder
```

Fluent builder that prepares arrays for ``GasData``.

Each setter performs unit conversion and ensures correct dtypes.
Batch dimensions are inserted automatically when 1D concentration is
provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L45-L217)

### particula.gas.gas_data_builder.GasDataBuilder._concentration

*attribute* · *instance attribute*

```python
_concentration: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L57-L57)

### particula.gas.gas_data_builder.GasDataBuilder._molar_mass

*attribute* · *instance attribute*

```python
_molar_mass: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L56-L56)

### particula.gas.gas_data_builder.GasDataBuilder._n_boxes

*attribute* · *instance attribute*

```python
_n_boxes: Optional[int] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L59-L59)

### particula.gas.gas_data_builder.GasDataBuilder._names

*attribute* · *instance attribute*

```python
_names: Optional[list[str]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L55-L55)

### particula.gas.gas_data_builder.GasDataBuilder._partitioning

*attribute* · *instance attribute*

```python
_partitioning: Optional[NDArray[np.bool_]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L58-L58)

### particula.gas.gas_data_builder.GasDataBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize empty builder state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L53-L59)

### particula.gas.gas_data_builder.GasDataBuilder.build

*method*

```python
def build() -> GasData
```

Build and return validated GasData.

**Returns**

- (`GasData`) — A validated GasData instance.

**Raises**

- `ValueError` — If required fields are missing or invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L182-L217)

### particula.gas.gas_data_builder.GasDataBuilder.set_concentration

*method*

```python
def set_concentration(concentration: Union[list[float], NDArray[np.float64]], units: str = 'kg/m^3') -> 'GasDataBuilder'
```

Set concentrations with unit conversion and auto batch dimension.

**Parameters**

- `concentration` (`Union[list[float], NDArray[np.float64]]`) — Concentration values. If 1D (n_species,), batch dimension added. If 2D (n_boxes, n_species), used as-is.
- `units` (`str`) (default: `'kg/m^3'`) — Units of the provided concentration. Supported: kg/m^3, g/m^3.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When any concentration is negative or wrong dimension.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L105-L140)

### particula.gas.gas_data_builder.GasDataBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[list[float], NDArray[np.float64]], units: str = 'kg/mol') -> 'GasDataBuilder'
```

Set molar masses with optional unit conversion.

**Parameters**

- `molar_mass` (`Union[list[float], NDArray[np.float64]]`) — Molar mass values shaped (n_species,).
- `units` (`str`) (default: `'kg/mol'`) — Units of the provided values. Supported: kg/mol, g/mol.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When molar_mass is not 1D or any value is non-positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L73-L103)

### particula.gas.gas_data_builder.GasDataBuilder.set_n_boxes

*method*

```python
def set_n_boxes(n_boxes: int) -> 'GasDataBuilder'
```

Set number of boxes for broadcasting 1D concentration.

**Parameters**

- `n_boxes` (`int`) — Number of simulation boxes.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L164-L180)

### particula.gas.gas_data_builder.GasDataBuilder.set_names

*method*

```python
def set_names(names: list[str]) -> 'GasDataBuilder'
```

Set species names.

**Parameters**

- `names` (`list[str]`) — List of species names.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L61-L71)

### particula.gas.gas_data_builder.GasDataBuilder.set_partitioning

*method*

```python
def set_partitioning(partitioning: Union[list[bool], NDArray[np.bool_]]) -> 'GasDataBuilder'
```

Set whether each species can partition.

**Parameters**

- `partitioning` (`Union[list[bool], NDArray[np.bool_]]`) — Boolean array indicating which species partition. Shape: (n_species,).

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When partitioning is not 1D.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L142-L162)
