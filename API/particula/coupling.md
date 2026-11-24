# particula.coupling

*module*

Strategies, builders and fixed-grid geometry for prescribed air flows.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/__init__.py#L1-L1)

## particula.coupling.FlowFactory

*class*

```python
class FlowFactory(StrategyFactoryABC)
```

Bases: `particula.abc_factory.StrategyFactoryABC`

Re-exported from: `particula.coupling.flow_factory`

Construct an independent builder for each supported flow provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L173-L185)

### particula.coupling.FlowFactory.get_builders

*method*

```python
def get_builders()
```

Map flow names to fresh builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L176-L185)

### particula.coupling.FlowFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: Optional[Dict[str, Any]] = None) -> StrategyT
```

Inherited from: `particula.abc_factory.StrategyFactoryABC`

Create a strategy instance using its corresponding builder.

**Parameters**

- `strategy_type` (`str`) — Name of the strategy to build.
- `parameters` (`Dict[str, Any]`) (default: `None`) — Dictionary of parameters to configure the chosen builder.

**Returns**

- `StrategyT` (`StrategyT`) — The built strategy object corresponding to the specified type.

**Raises**

- `ValueError` — If the `strategy_type` is unknown, or if any required
parameter is invalid/missing for the chosen builder.

**Examples**

```py title="Strategy Creation Example"
my_factory = SomeStrategyFactory()
my_strategy = my_factory.get_strategy(
    "desired_strategy", {"param_x": 42}
)
# my_strategy is now an instance configured with param_x=42
```

> **References**
>
> - "Factory Method Pattern,"
> [Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L69-L110)

## particula.coupling.FlowGrid

*class* · *dataclass*

```python
class FlowGrid
```

Re-exported from: `particula.coupling.grid`

Uniform reference edges and optional piecewise-linear bed elevations.

A singleton axis has the physical thickness in the supplied bounds. For a
terrain grid the third coordinate spans zero to the level lid in metres.
Boundaries are ordered west/east, south/north, bottom/top. Open faces
accept prescribed inlet concentrations; closed faces have zero transfer.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L9-L163)

### particula.coupling.FlowGrid.bed

*attribute* · *class attribute* · *instance attribute*

```python
bed: tuple = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L22-L22)

### particula.coupling.FlowGrid.boundaries

*attribute* · *class attribute* · *instance attribute*

```python
boundaries: tuple = ('closed') * 6
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L21-L21)

### particula.coupling.FlowGrid.bounds

*attribute* · *instance attribute*

```python
bounds: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L20-L20)

### particula.coupling.FlowGrid.shape

*attribute* · *instance attribute*

```python
shape: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L19-L19)

### particula.coupling.FlowGrid.centers

*property* · *cached*

```python
centers
```

Physical center coordinates (m), one row per cell.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L116-L118)

### particula.coupling.FlowGrid.coordinates

*property* · *cached*

```python
coordinates
```

Reference cell centers (ncell,3) in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L85-L92)

### particula.coupling.FlowGrid.edges

*property* · *cached*

```python
edges
```

Reference edge coordinates in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L67-L72)

### particula.coupling.FlowGrid.neighbors

*property* · *cached*

```python
neighbors
```

Six neighboring row IDs; exterior faces use -1.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L126-L139)

### particula.coupling.FlowGrid.spacing

*property* · *cached*

```python
spacing
```

Reference cell widths in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L75-L82)

### particula.coupling.FlowGrid.volume

*property* · *cached*

```python
volume
```

Exact physical volume (m³) of each mapped cell.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L121-L123)

### particula.coupling.FlowGrid.__init__

*method*

```python
def __init__(shape: tuple, bounds: tuple, boundaries: tuple = ('closed') * 6, bed: tuple = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L0-L0)

### particula.coupling.FlowGrid.__post_init__

*method*

```python
def __post_init__()
```

Validate geometry and opposite periodic faces before allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L24-L64)

### particula.coupling.FlowGrid.face_geometry

*method*

```python
def face_geometry(side)
```

Return reference centers and oriented physical area vectors (m²).

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L141-L163)

### particula.coupling.FlowGrid.ground

*method*

```python
def ground(x)
```

Piecewise-linear physical bed ordinate in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L94-L98)

### particula.coupling.FlowGrid.jacobian

*method*

```python
def jacobian(x)
```

Ratio of physical to reference column thickness.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L100-L104)

### particula.coupling.FlowGrid.world

*method*

```python
def world(coord)
```

Convert (...,3) reference metres into world xyz.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L106-L113)

## particula.coupling.FlowStrategy

*class*

```python
class FlowStrategy
```

Re-exported from: `particula.coupling.flows`

A forcing provider exposing the operations its data support.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L29-L38)

### particula.coupling.FlowStrategy.interval

*method*

```python
def interval(grid, start, end, mixing, *, device='cpu')
```

Offer a grid interval or reject an incompatible provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L32-L34)

### particula.coupling.FlowStrategy.trajectory

*method*

```python
def trajectory(parcel_id)
```

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

**Modules**

- `particula.coupling.cavity_replay`
- `particula.coupling.dispersion`
- `particula.coupling.flow_factory`
- `particula.coupling.flow_replay`
- `particula.coupling.flows`
- `particula.coupling.grid`
- `particula.coupling.grid_inventory`
- `particula.coupling.grid_run`
- `particula.coupling.grid_warp`
- `particula.coupling.particle_dispersion`
- `particula.coupling.prescribed`
- `particula.coupling.sounding_sequence`
- `particula.coupling.surface_emission`
