# particula.dynamics.parcel_motion

*module*

Resolved Stokes settling with event-time migration in a translating mesh.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L1-L1)

## particula.dynamics.parcel_motion.Crossing

*class* · *dataclass*

```python
class Crossing
```

Face event at absolute seconds with its represented inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L66-L76)

### particula.dynamics.parcel_motion.Crossing.axis

*attribute* · *instance attribute*

```python
axis: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L74-L74)

### particula.dynamics.parcel_motion.Crossing.destination

*attribute* · *instance attribute*

```python
destination: str | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L73-L73)

### particula.dynamics.parcel_motion.Crossing.kind

*attribute* · *instance attribute*

```python
kind: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L75-L75)

### particula.dynamics.parcel_motion.Crossing.particle

*attribute* · *instance attribute*

```python
particle: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L71-L71)

### particula.dynamics.parcel_motion.Crossing.record

*attribute* · *instance attribute*

```python
record: Particle
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L76-L76)

### particula.dynamics.parcel_motion.Crossing.source

*attribute* · *instance attribute*

```python
source: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L72-L72)

### particula.dynamics.parcel_motion.Crossing.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L70-L70)

### particula.dynamics.parcel_motion.Crossing.__init__

*method*

```python
def __init__(time: float, particle: str, source: str, destination: str | None, axis: int, kind: str, record: Particle) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L0-L0)

## particula.dynamics.parcel_motion.MotionState

*class* · *dataclass*

```python
class MotionState
```

Restart boundary with world coordinates and cumulative events.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L79-L130)

### particula.dynamics.parcel_motion.MotionState.events

*attribute* · *class attribute* · *instance attribute*

```python
events: tuple[Crossing, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L85-L85)

### particula.dynamics.parcel_motion.MotionState.geometry

*attribute* · *class attribute* · *instance attribute*

```python
geometry: str = ''
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L86-L86)

### particula.dynamics.parcel_motion.MotionState.particles

*attribute* · *instance attribute*

```python
particles: tuple[Particle, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L84-L84)

### particula.dynamics.parcel_motion.MotionState.retained

*attribute* · *class attribute* · *instance attribute*

```python
retained: tuple = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L87-L87)

### particula.dynamics.parcel_motion.MotionState.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L83-L83)

### particula.dynamics.parcel_motion.MotionState.__init__

*method*

```python
def __init__(time: float, particles: tuple[Particle, ...], events: tuple[Crossing, ...] = (), geometry: str = '', retained: tuple = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L0-L0)

### particula.dynamics.parcel_motion.MotionState.load

*method* · *classmethod*

```python
def load(path)
```

Read a checkpoint for validation at continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L93-L108)

### particula.dynamics.parcel_motion.MotionState.rebind_geometry

*method*

```python
def rebind_geometry(cluster, *, mesh_velocity=(0.0, 0.0, 0.0), boundaries=None)
```

Adopt explicitly updated bounds at a completed motion boundary.

Each live particle must still lie in its declared cell at this time.
Subsequent crossings use the new shared faces and boundary policies.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L110-L130)

### particula.dynamics.parcel_motion.MotionState.save

*method*

```python
def save(path)
```

Write the completed checkpoint to JSON.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L89-L91)

## particula.dynamics.parcel_motion.Particle

*class* · *dataclass*

```python
class Particle
```

Stable ID, world position (m), radius (m), density (kg/m³) and inventory.

Count is represented number, masses are kg per particle, and charge is
elementary charges per particle. Cell IDs are independent of storage slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L22-L63)

### particula.dynamics.parcel_motion.Particle.cell

*attribute* · *instance attribute*

```python
cell: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L31-L31)

### particula.dynamics.parcel_motion.Particle.charge

*attribute* · *instance attribute*

```python
charge: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L37-L37)

### particula.dynamics.parcel_motion.Particle.count

*attribute* · *instance attribute*

```python
count: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L35-L35)

### particula.dynamics.parcel_motion.Particle.density

*attribute* · *instance attribute*

```python
density: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L34-L34)

### particula.dynamics.parcel_motion.Particle.id

*attribute* · *instance attribute*

```python
id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L30-L30)

### particula.dynamics.parcel_motion.Particle.masses

*attribute* · *instance attribute*

```python
masses: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L36-L36)

### particula.dynamics.parcel_motion.Particle.position

*attribute* · *instance attribute*

```python
position: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L32-L32)

### particula.dynamics.parcel_motion.Particle.radius

*attribute* · *instance attribute*

```python
radius: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L33-L33)

### particula.dynamics.parcel_motion.Particle.__init__

*method*

```python
def __init__(id: str, cell: str, position: tuple, radius: float, density: float, count: float, masses: tuple, charge: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L0-L0)

### particula.dynamics.parcel_motion.Particle.__post_init__

*method*

```python
def __post_init__()
```

Validate the physical inputs and retain owned values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L39-L63)

## particula.dynamics.parcel_motion._advance_to_face

*function*

```python
def _advance_to_face(keys, live, positions, boxes, v, now, finish, opts, device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L215-L248)

## particula.dynamics.parcel_motion._boundary_rules

*function*

```python
def _boundary_rules(boxes, mesh, boundaries)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L351-L369)

## particula.dynamics.parcel_motion._cross_faces

*function*

```python
def _cross_faces(cluster, hits, keys, live, v, positions, boxes, policies, blocked, occupancy, mesh, now, limits, events, velocity, speed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L283-L348)

## particula.dynamics.parcel_motion._destination

*function*

```python
def _destination(cluster, cell, axis, sign, x, boxes, periodic=False)
```

Find an oriented face patch containing the trajectory intersection.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L140-L163)

## particula.dynamics.parcel_motion._face_rule

*function*

```python
def _face_rule(cluster, cell, a, sign, x, boxes, policies)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L251-L260)

## particula.dynamics.parcel_motion._initial_motion

*function*

```python
def _initial_motion(state, boxes, air_velocity, mesh, capacity, temperature, pressure)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L166-L212)

## particula.dynamics.parcel_motion._relative_velocity

*function*

```python
def _relative_velocity(air, mesh, settling)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L372-L373)

## particula.dynamics.parcel_motion._retained_contacts

*function*

```python
def _retained_contacts(retained, live, velocity, mesh, speed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L376-L384)

## particula.dynamics.parcel_motion._update_wall_contacts

*function*

```python
def _update_wall_contacts(blocked, record, dst, relative, position, cluster, boxes, policies)
```

Keep a sliding contact where the destination continues its wall.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L263-L280)

## particula.dynamics.parcel_motion._vector

*function*

```python
def _vector(v)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L133-L137)

## particula.dynamics.parcel_motion.advance_motion

*function*

```python
def advance_motion(cluster, state, *, duration, air_velocity, capacity, device='cpu', mesh_velocity=(0.0, 0.0, 0.0), boundaries=None, temperature=288.15, pressure=101325.0, gradient=False)
```

Advance constant air velocity per cell in a uniformly translating mesh.

Bounds are at time zero. Boundary keys are (cell, axis, sign), with
axis 0/1/2 and sign −1/+1. Modes are retain, periodic, open and deposit.
Coincident arrivals are resolved by axis then stable ID. Capacity is per
destination cell. A rejected call leaves its input checkpoint intact.
CPU or native CUDA computes arrival times and displacements; the host
reserves slots and records events at each crossing boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_motion.py#L387-L483)
