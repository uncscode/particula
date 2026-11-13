# particula.gas.gas_data

*module*

Provide a batched CPU gas data container for multi-box CFD simulations.

GasData isolates gas species arrays from behavior (vapor pressure strategies)
while embedding the batch dimension required for CFD experiments spanning
multiple boxes. It is the CPU owner of ordered gas names, molar masses,
concentrations, and boolean partitioning flags only. GPU helper state such as
vapor pressure lives on ``WarpGasData`` with shape ``(n_boxes, n_species)``
and does not round-trip back into this container automatically. Restoring from
GPU data therefore depends on caller-managed ordered names for semantic name
reconstruction.

> **Example**
>
> Single-box simulation (n_boxes=1)::
> 
>     from particula.gas.gas_data import GasData
>     import numpy as np
> 
>     data = GasData(
>         name=["Water", "Ammonia", "H2SO4"],
>         molar_mass=np.array([0.018, 0.017, 0.098]),
>         concentration=np.array([[1e-6, 5e-9, 2e-10]]),  # (1, 3)
>         partitioning=np.array([True, True, True]),
>     )
> 
> Multi-box CFD simulation (100 boxes)::
> 
>     cfd_data = GasData(
>         name=["Water", "Ammonia", "H2SO4"],
>         molar_mass=np.array([0.018, 0.017, 0.098]),
>         concentration=np.zeros((100, 3)),  # kg/m^3, (100, 3)
>         partitioning=np.array([True, True, True]),
>     )

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L1-L1)

## particula.gas.gas_data.GasData

*class* · *dataclass*

```python
class GasData
```

Batched gas species data container for multi-box simulations.

Simple data container with batch dimension built-in. Concentration
arrays have shape (n_boxes, n_species) to support multi-box CFD.
Single-box simulations use n_boxes=1.

``GasData`` is also the CPU-side owner of ordered species names. GPU
transfer helpers exclude ``name``, convert ``partitioning`` as
``bool → int32 → bool`` with binary ``0``/``1`` restore requirements, and
intentionally exclude GPU-only helper state such as vapor pressure from
restored CPU data. ``from_warp_gas_data()`` can rebuild names only from
caller-managed ordered names or placeholder names; it cannot recover
semantic species identity from GPU data alone.

This is NOT a frozen dataclass - concentrations can be updated in place
for performance. Use copy() if immutability is needed.

**Attributes**

- `name` (`list[str]`) — Species names. List of strings, length n_species.
- `molar_mass` (`NDArray[np.float64]`) — Molar masses in kg/mol. Shape: (n_species,)
- `concentration` (`NDArray[np.float64]`) — Mass concentrations in kg/m^3. Shape: (n_boxes, n_species)
- `partitioning` (`NDArray[np.bool_]`) — Whether each species can partition to particles. Shape: (n_species,) - shared across boxes

**Raises**

- `ValueError` — If array shapes are inconsistent or species list is empty.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L41-L142)

### particula.gas.gas_data.GasData.concentration

*attribute* · *instance attribute*

```python
concentration: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L74-L74)

### particula.gas.gas_data.GasData.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L73-L73)

### particula.gas.gas_data.GasData.name

*attribute* · *instance attribute*

```python
name: list[str]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L72-L72)

### particula.gas.gas_data.GasData.partitioning

*attribute* · *instance attribute*

```python
partitioning: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L75-L75)

### particula.gas.gas_data.GasData.n_boxes

*property*

```python
n_boxes: int
```

Number of simulation boxes (batch dimension).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L126-L128)

### particula.gas.gas_data.GasData.n_species

*property*

```python
n_species: int
```

Number of gas species.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L131-L133)

### particula.gas.gas_data.GasData.__init__

*method*

```python
def __init__(name: list[str], molar_mass: NDArray[np.float64], concentration: NDArray[np.float64], partitioning: NDArray[np.bool_]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L0-L0)

### particula.gas.gas_data.GasData.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate array shapes are consistent and enforce boolean mask.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L77-L123)

### particula.gas.gas_data.GasData.copy

*method*

```python
def copy() -> GasData
```

Create a deep copy of this GasData.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L135-L142)
