# particula.dynamics.parcel_cluster

*module*

Parcel geometry and directed, conservative exchange under prescribed flow.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L1-L1)

## particula.dynamics.parcel_cluster.Connection

*class* · *dataclass*

```python
class Connection
```

Directed flow (m³/s) of organic gas, particle bins or water.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L13-L35)

### particula.dynamics.parcel_cluster.Connection.active_intervals

*attribute* · *class attribute* · *instance attribute*

```python
active_intervals: tuple[tuple[float, float], ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L23-L23)

### particula.dynamics.parcel_cluster.Connection.donor

*attribute* · *instance attribute*

```python
donor: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L17-L17)

### particula.dynamics.parcel_cluster.Connection.flow

*attribute* · *instance attribute*

```python
flow: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L19-L19)

### particula.dynamics.parcel_cluster.Connection.gas

*attribute* · *class attribute* · *instance attribute*

```python
gas: bool = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L20-L20)

### particula.dynamics.parcel_cluster.Connection.particles

*attribute* · *class attribute* · *instance attribute*

```python
particles: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L21-L21)

### particula.dynamics.parcel_cluster.Connection.receiver

*attribute* · *instance attribute*

```python
receiver: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L18-L18)

### particula.dynamics.parcel_cluster.Connection.water

*attribute* · *class attribute* · *instance attribute*

```python
water: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L22-L22)

### particula.dynamics.parcel_cluster.Connection.__init__

*method*

```python
def __init__(donor: str, receiver: str, flow: float, gas: bool = True, particles: bool = False, water: bool = False, active_intervals: tuple[tuple[float, float], ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L0-L0)

### particula.dynamics.parcel_cluster.Connection.__post_init__

*method*

```python
def __post_init__()
```

Validate half-open contact intervals in absolute seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L25-L35)

## particula.dynamics.parcel_cluster.ParcelCluster

*class* · *dataclass*

```python
class ParcelCluster
```

Traces keyed by stable ID, with centers and cell bounds in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L38-L108)

### particula.dynamics.parcel_cluster.ParcelCluster.bounds

*attribute* · *instance attribute*

```python
bounds: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L45-L45)

### particula.dynamics.parcel_cluster.ParcelCluster.centers

*attribute* · *instance attribute*

```python
centers: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L43-L43)

### particula.dynamics.parcel_cluster.ParcelCluster.connections

*attribute* · *instance attribute*

```python
connections: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L44-L44)

### particula.dynamics.parcel_cluster.ParcelCluster.faces

*attribute* · *instance attribute*

```python
faces: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L46-L46)

### particula.dynamics.parcel_cluster.ParcelCluster.traces

*attribute* · *instance attribute*

```python
traces: dict[str, ParcelTrace]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L42-L42)

### particula.dynamics.parcel_cluster.ParcelCluster.geometric_volumes

*property*

```python
geometric_volumes
```

Return bounding-cell volumes in m³ for particle location.

Prescribed gas exchange uses the thermodynamic volumes in the traces.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L49-L57)

### particula.dynamics.parcel_cluster.ParcelCluster.__init__

*method*

```python
def __init__(traces: dict[str, ParcelTrace], centers: dict, connections: tuple, bounds: dict, faces: tuple) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L0-L0)

### particula.dynamics.parcel_cluster.ParcelCluster.custom

*method* · *classmethod*

```python
def custom(traces, *, centers, connections=(), bounds=None)
```

Declare exchange and optional boxes for geometric crossings.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L59-L80)

### particula.dynamics.parcel_cluster.ParcelCluster.regular

*method* · *classmethod*

```python
def regular(traces, *, shape, widths, connections=())
```

Build a regular grid from counts, widths and ordered traces.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L82-L108)

## particula.dynamics.parcel_cluster._connections

*function*

```python
def _connections(traces, connections)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L111-L139)

## particula.dynamics.parcel_cluster._geometry

*function*

```python
def _geometry(ids, xyz, bounds)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L142-L177)

## particula.dynamics.parcel_cluster._independent_groups

*function*

```python
def _independent_groups(cluster, max_step, device, amplitude, humidity_cell)
```

Batch common forcing schedules and preserve independent step meshes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L219-L263)

## particula.dynamics.parcel_cluster.run_cluster

*function*

```python
def run_cluster(cluster, *, max_step, device='cpu', gradient=None, humidity_amplitude=0.0)
```

Advance all parcels across the union of their forcing records.

A gradient tuple (donor, receiver, observation_cell[, 'gas']) selects
a flow rate and terminal organic particle mass (or gas mass).
("humidity", parcel_id) observes terminal liquid in that parcel as all
histories undergo the same fractional vapor perturbation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_cluster.py#L180-L216)
