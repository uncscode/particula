# particula.coupling.grid

*module*

Stationary structured finite volumes. SI lengths; C-order (x,y,layer) IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L1-L1)

## particula.coupling.grid.FlowGrid

*class* · *dataclass*

```python
class FlowGrid
```

Uniform reference edges and optional piecewise-linear bed elevations.

A singleton axis has the physical thickness in the supplied bounds. For a
terrain grid the third coordinate spans zero to the level lid in metres.
Boundaries are ordered west/east, south/north, bottom/top. Open faces
accept prescribed inlet concentrations; closed faces have zero transfer.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L9-L163)

### particula.coupling.grid.FlowGrid.bed

*attribute* · *class attribute* · *instance attribute*

```python
bed: tuple = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L22-L22)

### particula.coupling.grid.FlowGrid.boundaries

*attribute* · *class attribute* · *instance attribute*

```python
boundaries: tuple = ('closed') * 6
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L21-L21)

### particula.coupling.grid.FlowGrid.bounds

*attribute* · *instance attribute*

```python
bounds: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L20-L20)

### particula.coupling.grid.FlowGrid.shape

*attribute* · *instance attribute*

```python
shape: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L19-L19)

### particula.coupling.grid.FlowGrid.centers

*property* · *cached*

```python
centers
```

Physical center coordinates (m), one row per cell.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L116-L118)

### particula.coupling.grid.FlowGrid.coordinates

*property* · *cached*

```python
coordinates
```

Reference cell centers (ncell,3) in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L85-L92)

### particula.coupling.grid.FlowGrid.edges

*property* · *cached*

```python
edges
```

Reference edge coordinates in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L67-L72)

### particula.coupling.grid.FlowGrid.neighbors

*property* · *cached*

```python
neighbors
```

Six neighboring row IDs; exterior faces use -1.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L126-L139)

### particula.coupling.grid.FlowGrid.spacing

*property* · *cached*

```python
spacing
```

Reference cell widths in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L75-L82)

### particula.coupling.grid.FlowGrid.volume

*property* · *cached*

```python
volume
```

Exact physical volume (m³) of each mapped cell.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L121-L123)

### particula.coupling.grid.FlowGrid.__init__

*method*

```python
def __init__(shape: tuple, bounds: tuple, boundaries: tuple = ('closed') * 6, bed: tuple = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L0-L0)

### particula.coupling.grid.FlowGrid.__post_init__

*method*

```python
def __post_init__()
```

Validate geometry and opposite periodic faces before allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L24-L64)

### particula.coupling.grid.FlowGrid.face_geometry

*method*

```python
def face_geometry(side)
```

Return reference centers and oriented physical area vectors (m²).

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L141-L163)

### particula.coupling.grid.FlowGrid.ground

*method*

```python
def ground(x)
```

Piecewise-linear physical bed ordinate in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L94-L98)

### particula.coupling.grid.FlowGrid.jacobian

*method*

```python
def jacobian(x)
```

Ratio of physical to reference column thickness.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L100-L104)

### particula.coupling.grid.FlowGrid.world

*method*

```python
def world(coord)
```

Convert (...,3) reference metres into world xyz.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid.py#L106-L113)
