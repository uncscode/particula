# particula.coupling.cavity_replay

*module*

Map a stationary Cartesian OpenFOAM cavity export to SI forcing.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L1-L1)

## particula.coupling.cavity_replay._UNITS

*attribute* · *module attribute*

```python
_UNITS = {'time': 's', 'centres': 'm', 'volumes': 'm3', 'face_centres': 'm', 'area_vectors': 'm2', 'phi': 'm3/s', 'velocity': 'm/s', 'scalar': 'kg/m3', 'kinematic_pressure': 'm2/s2'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L14-L24)

## particula.coupling.cavity_replay.CavityReference

*class* · *dataclass*

```python
class CavityReference
```

Forcing and independent scalar histories (kg/m³) in grid row order.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L27-L35)

### particula.coupling.cavity_replay.CavityReference.concentration

*attribute* · *instance attribute*

```python
concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L32-L32)

### particula.coupling.cavity_replay.CavityReference.fine_restricted

*attribute* · *instance attribute*

```python
fine_restricted: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L34-L34)

### particula.coupling.cavity_replay.CavityReference.forcing

*attribute* · *instance attribute*

```python
forcing: GridReplay
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L31-L31)

### particula.coupling.cavity_replay.CavityReference.half_step

*attribute* · *instance attribute*

```python
half_step: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L33-L33)

### particula.coupling.cavity_replay.CavityReference.metadata

*attribute* · *instance attribute*

```python
metadata: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L35-L35)

### particula.coupling.cavity_replay.CavityReference.__init__

*method*

```python
def __init__(forcing: GridReplay, concentration: np.ndarray, half_step: np.ndarray, fine_restricted: np.ndarray, metadata: dict) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L0-L0)

## particula.coupling.cavity_replay._array

*function*

```python
def _array(data, key, shape, *, integer=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L49-L58)

## particula.coupling.cavity_replay._cell_mapping

*function*

```python
def _cell_mapping(grid, data)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L61-L81)

## particula.coupling.cavity_replay._face_rates

*function*

```python
def _face_rates(grid, data, lookup, diffusivity)
```

Require a single owner-outward face for every logical cell side.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L84-L141)

## particula.coupling.cavity_replay.map_cavity

*function*

```python
def map_cavity(metadata, data)
```

Map every native scalar step in a stationary cavity export.

Cell IDs may be permuted. Internal face owner/neighbour may be exchanged
if the area vector and signed flux are reversed together. This adapter
uses volume-based concentrations and frozen pressure-corrected face rates.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L144-L214)

## particula.coupling.cavity_replay.read_cavity

*function*

```python
def read_cavity(directory)
```

Verify the exported archive SHA-256 before mapping native cell IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/cavity_replay.py#L38-L46)
