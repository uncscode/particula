# particula.coupling.grid_run

*module*

Conservative fixed-grid execution and completed-interval checkpoints.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L1-L1)

## particula.coupling.grid_run.GridRun

*class*

```python
class GridRun
```

Carry named inventory and stable particles across accepted intervals.

CPU uses NumPy for conservative transport. Particle face traversal uses
the same compiled Warp operation on CPU and CUDA. Observation, checkpoint
and host-owned input are explicit synchronization boundaries.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L47-L666)

### particula.coupling.grid_run.GridRun._faulted

*attribute* · *instance attribute*

```python
_faulted = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L80-L80)

### particula.coupling.grid_run.GridRun.device

*attribute* · *instance attribute*

```python
device = wp.get_device(device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L74-L74)

### particula.coupling.grid_run.GridRun.dispersion

*attribute* · *instance attribute*

```python
dispersion = dispersion
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L81-L81)

### particula.coupling.grid_run.GridRun.exports

*attribute* · *instance attribute*

```python
exports = np.zeros_like(a)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L104-L104)

### particula.coupling.grid_run.GridRun.forcing

*attribute* · *instance attribute*

```python
forcing = forcing
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L71-L71)

### particula.coupling.grid_run.GridRun.gpu

*attribute* · *instance attribute*

```python
gpu = self.device.is_cuda
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L75-L75)

### particula.coupling.grid_run.GridRun.grid

*attribute* · *instance attribute*

```python
grid = forcing.grid
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L72-L72)

### particula.coupling.grid_run.GridRun.imports

*attribute* · *instance attribute*

```python
imports = np.zeros_like(a)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L103-L103)

### particula.coupling.grid_run.GridRun.index

*attribute* · *instance attribute*

```python
index = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L79-L79)

### particula.coupling.grid_run.GridRun.inlet

*attribute* · *instance attribute*

```python
inlet = inlet
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L99-L99)

### particula.coupling.grid_run.GridRun.jump_neighbors

*attribute* · *instance attribute*

```python
jump_neighbors = self.neighbor if self.gpu else self._device_array(self.neighbor, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L129-L133)

### particula.coupling.grid_run.GridRun.lanes

*attribute* · *instance attribute*

```python
lanes = uptake.bind(layout) if uptake else ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L83-L83)

### particula.coupling.grid_run.GridRun.layout

*attribute* · *instance attribute*

```python
layout = layout
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L73-L73)

### particula.coupling.grid_run.GridRun.manual_records

*attribute* · *instance attribute*

```python
manual_records = len(self.releases)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L86-L86)

### particula.coupling.grid_run.GridRun.neighbor

*attribute* · *instance attribute*

```python
neighbor = self.grid.neighbors
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L100-L100)

### particula.coupling.grid_run.GridRun.owner

*attribute* · *instance attribute*

```python
owner = owner
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L78-L78)

### particula.coupling.grid_run.GridRun.releases

*attribute* · *instance attribute*

```python
releases = tuple(releases)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L84-L84)

### particula.coupling.grid_run.GridRun.scratch

*attribute* · *instance attribute*

```python
scratch = wp.empty_like(self.state)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L127-L127)

### particula.coupling.grid_run.GridRun.signature

*attribute* · *instance attribute*

```python
signature = digest.hexdigest()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L122-L122)

### particula.coupling.grid_run.GridRun.state

*attribute* · *instance attribute*

```python
state = a.copy()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L102-L102)

### particula.coupling.grid_run.GridRun.surface

*attribute* · *instance attribute*

```python
surface = surface
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L85-L85)

### particula.coupling.grid_run.GridRun.uptake

*attribute* · *instance attribute*

```python
uptake = uptake
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L82-L82)

### particula.coupling.grid_run.GridRun.volume

*attribute* · *instance attribute*

```python
volume = self.grid.volume
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L101-L101)

### particula.coupling.grid_run.GridRun.__init__

*method*

```python
def __init__(forcing, layout, inventory, *, releases=(), bed=None, surface=None, inlet=None, uptake=None, dispersion=None, owner='particula', device='cpu')
```

Bind the forcing, native inventory layout and execution device.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L55-L133)

### particula.coupling.grid_run.GridRun._advance_frame

*method*

```python
def _advance_frame(f, host)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L257-L298)

### particula.coupling.grid_run.GridRun._advance_particles

*method*

```python
def _advance_particles(i, start, step_dt, speed, k, jumps)
```

Traverse physical boundaries, then sample mapped mixing faces.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L322-L372)

### particula.coupling.grid_run.GridRun._apply_uptake

*method*

```python
def _apply_uptake(dt)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L389-L402)

### particula.coupling.grid_run.GridRun._check_checkpoint_particles

*method*

```python
def _check_checkpoint_particles(values, bed_mass, index)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L632-L666)

### particula.coupling.grid_run.GridRun._device_array

*method*

```python
def _device_array(a, dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L135-L138)

### particula.coupling.grid_run.GridRun._emit_binned

*method*

```python
def _emit_binned()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L374-L387)

### particula.coupling.grid_run.GridRun._field_total

*method*

```python
def _field_total(a)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L493-L500)

### particula.coupling.grid_run.GridRun._history_id

*method*

```python
def _history_id(index)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L481-L488)

### particula.coupling.grid_run.GridRun._init_particles

*method*

```python
def _init_particles(bed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L140-L232)

### particula.coupling.grid_run.GridRun._install_host

*method*

```python
def _install_host(a, imp, exp)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L425-L438)

### particula.coupling.grid_run.GridRun._read

*method*

```python
def _read(a)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L490-L491)

### particula.coupling.grid_run.GridRun._require_consistent

*method*

```python
def _require_consistent()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L475-L479)

### particula.coupling.grid_run.GridRun._step_schedule

*method*

```python
def _step_schedule(f, dt, cfl)
```

Bound the outgoing fraction and stochastic first-contact step.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L300-L320)

### particula.coupling.grid_run.GridRun._transport

*method*

```python
def _transport(outgoing, incoming)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L440-L473)

### particula.coupling.grid_run.GridRun._validate_host

*method*

```python
def _validate_host(f, h)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L404-L423)

### particula.coupling.grid_run.GridRun.advance

*method*

```python
def advance(host=None)
```

Advance one accepted interval, then apply the local uptake once.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L234-L239)

### particula.coupling.grid_run.GridRun.advance_interval

*method*

```python
def advance_interval(frame, host=None)
```

Accept one live host interval and retain forcing for exact replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L241-L255)

### particula.coupling.grid_run.GridRun.budget

*method*

```python
def budget()
```

Reduce fields and resolved count, charge and kg by fate.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L502-L539)

### particula.coupling.grid_run.GridRun.checkpoint

*method*

```python
def checkpoint(path)
```

Write the state at a completed coupling boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L559-L582)

### particula.coupling.grid_run.GridRun.restore

*method*

```python
def restore(path)
```

Restore matching state at an accepted interval boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L584-L630)

### particula.coupling.grid_run.GridRun.snapshot

*method*

```python
def snapshot()
```

Download fields, stable IDs, xyz and event times.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L541-L557)

## particula.coupling.grid_run.HostTransport

*class* · *dataclass*

```python
class HostTransport
```

Already transported extensive inventories at a completed host interval.

imports and exports are nonnegative per-cell boundary totals over the
interval. Internal redistribution belongs to the host.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L32-L44)

### particula.coupling.grid_run.HostTransport.end

*attribute* · *instance attribute*

```python
end: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L41-L41)

### particula.coupling.grid_run.HostTransport.exports

*attribute* · *instance attribute*

```python
exports: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L44-L44)

### particula.coupling.grid_run.HostTransport.imports

*attribute* · *instance attribute*

```python
imports: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L43-L43)

### particula.coupling.grid_run.HostTransport.inventories

*attribute* · *instance attribute*

```python
inventories: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L42-L42)

### particula.coupling.grid_run.HostTransport.start

*attribute* · *instance attribute*

```python
start: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L40-L40)

### particula.coupling.grid_run.HostTransport.__init__

*method*

```python
def __init__(start: float, end: float, inventories: np.ndarray, imports: np.ndarray, exports: np.ndarray) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L0-L0)

## particula.coupling.grid_run.Release

*class* · *dataclass*

```python
class Release
```

One stable particle record, slip (m/s) and launch interval.

A launch interval of -1 means initially airborne. Other records reserve
the specified particle inventory in the bed until that interval begins.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L19-L29)

### particula.coupling.grid_run.Release.interval

*attribute* · *class attribute* · *instance attribute*

```python
interval: int = -1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L29-L29)

### particula.coupling.grid_run.Release.particle

*attribute* · *instance attribute*

```python
particle: Particle
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L27-L27)

### particula.coupling.grid_run.Release.slip

*attribute* · *instance attribute*

```python
slip: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L28-L28)

### particula.coupling.grid_run.Release.__init__

*method*

```python
def __init__(particle: Particle, slip: float, interval: int = -1) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_run.py#L0-L0)
