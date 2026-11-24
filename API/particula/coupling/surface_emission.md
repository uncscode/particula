# particula.coupling.surface_emission

*module*

Finite-bed dust emission under a prescribed piecewise-linear wind.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L1-L1)

## particula.coupling.surface_emission.BedPatch

*class* · *dataclass*

```python
class BedPatch
```

A bottom cell's erodible slope area (m²) and initial kg per class.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L58-L80)

### particula.coupling.surface_emission.BedPatch.area

*attribute* · *instance attribute*

```python
area: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L64-L64)

### particula.coupling.surface_emission.BedPatch.available

*attribute* · *instance attribute*

```python
available: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L65-L65)

### particula.coupling.surface_emission.BedPatch.cell

*attribute* · *instance attribute*

```python
cell: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L63-L63)

### particula.coupling.surface_emission.BedPatch.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L62-L62)

### particula.coupling.surface_emission.BedPatch.__init__

*method*

```python
def __init__(name: str, cell: int, area: float, available: tuple) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L0-L0)

### particula.coupling.surface_emission.BedPatch.__post_init__

*method*

```python
def __post_init__()
```

Check patch identity, physical surface area and inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L67-L80)

## particula.coupling.surface_emission.DustClass

*class* · *dataclass*

```python
class DustClass
```

Properties of an emitted dry spherical class in SI units.

threshold is friction velocity (m/s), coefficient is kg/m²/s above
normalized excess shear, diameter is m, density is kg/m³, slip is
prescribed downward terminal speed (m/s), fractions are dry mass fractions.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L13-L55)

### particula.coupling.surface_emission.DustClass.coefficient

*attribute* · *instance attribute*

```python
coefficient: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L24-L24)

### particula.coupling.surface_emission.DustClass.density

*attribute* · *instance attribute*

```python
density: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L26-L26)

### particula.coupling.surface_emission.DustClass.diameter

*attribute* · *instance attribute*

```python
diameter: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L25-L25)

### particula.coupling.surface_emission.DustClass.fractions

*attribute* · *instance attribute*

```python
fractions: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L27-L27)

### particula.coupling.surface_emission.DustClass.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L22-L22)

### particula.coupling.surface_emission.DustClass.slip

*attribute* · *class attribute* · *instance attribute*

```python
slip: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L28-L28)

### particula.coupling.surface_emission.DustClass.threshold

*attribute* · *instance attribute*

```python
threshold: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L23-L23)

### particula.coupling.surface_emission.DustClass.mass

*property*

```python
mass
```

Return kg per physical grain of the emitted diameter.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L53-L55)

### particula.coupling.surface_emission.DustClass.__init__

*method*

```python
def __init__(name: str, threshold: float, coefficient: float, diameter: float, density: float, fractions: tuple, slip: float = 0.0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L0-L0)

### particula.coupling.surface_emission.DustClass.__post_init__

*method*

```python
def __post_init__()
```

Validate the dry mass and material partition.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L30-L50)

## particula.coupling.surface_emission.SurfaceEmission

*class*

```python
class SurfaceEmission
```

Prescribed bed exchange integrated over accepted replay intervals.

Each interval's integrated release enters at its left endpoint. Refining
accepted intervals refines the resulting transport timing. Forcing and
emission have independent outputs and share the same fixed grid and clock.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L107-L321)

### particula.coupling.surface_emission.SurfaceEmission.forcing_id

*attribute* · *instance attribute*

```python
forcing_id = forcing.identity
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L174-L174)

### particula.coupling.surface_emission.SurfaceEmission.friction_velocity

*attribute* · *instance attribute*

```python
friction_velocity = np.sqrt(drag_coefficient) * np.linalg.norm(wind, axis=2)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L188-L190)

### particula.coupling.surface_emission.SurfaceEmission.identity

*attribute* · *instance attribute*

```python
identity = digest.hexdigest()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L223-L223)

### particula.coupling.surface_emission.SurfaceEmission.materials

*attribute* · *instance attribute*

```python
materials = tuple(materials)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L142-L142)

### particula.coupling.surface_emission.SurfaceEmission.records

*attribute* · *instance attribute*

```python
records = np.argwhere(self.emitted > 0) if representation == 'resolved' else np.empty((0, 3), dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L192-L196)

### particula.coupling.surface_emission.SurfaceEmission.representation

*attribute* · *instance attribute*

```python
representation = representation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L173-L173)

### particula.coupling.surface_emission.SurfaceEmission.wind_times

*attribute* · *instance attribute*

```python
wind_times = knots.copy()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L187-L187)

### particula.coupling.surface_emission.SurfaceEmission.__init__

*method*

```python
def __init__(forcing, flow, patches, classes, materials, *, drag_coefficient, reference_height, ejection_height, representation, capacity=None, device='cpu')
```

Sample wind above the bed, integrate gusts and reserve record slots.

reference_height and ejection_height are physical metres above bed.
capacity limits all resolved source records across the recording.
Per-class rates use piecewise linear horizontal winds between knots.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L115-L223)

### particula.coupling.surface_emission.SurfaceEmission._integrate

*method*

```python
def _integrate(forcing, knots, wind, drag_coefficient)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L252-L279)

### particula.coupling.surface_emission.SurfaceEmission._reference_points

*method*

```python
def _reference_points(grid, ejection_height, reference_height)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L225-L250)

### particula.coupling.surface_emission.SurfaceEmission.bind

*method*

```python
def bind(forcing, layout)
```

Check compatibility and construct stable weighted resolved slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L281-L307)

### particula.coupling.surface_emission.SurfaceEmission.field_increment

*method*

```python
def field_increment(index, layout, n)
```

Convert emitted kg to count and dry-material lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L309-L321)

## particula.coupling.surface_emission._excess_integral

*function*

```python
def _excess_integral(u0, u1, threshold, cd)
```

Integrate positive normalized excess shear over unit linear-wind time.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/surface_emission.py#L83-L104)
