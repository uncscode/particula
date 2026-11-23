# particula.execution.multivapor_parcel

*module*

CUDA conditional-graph solver for the finite multivapor parcel.

The resident coordinates are pressure, temperature and analytical material
masses. Each RHS reconstructs its finite gas inventories by conservation.
This formulation and embedded solvers are independent of the Radau oracle,
which integrates the gas reservoirs explicitly.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L1-L1)

## particula.execution.multivapor_parcel.ParcelReplayCheckpoint

*class* · *dataclass*

```python
class ParcelReplayCheckpoint
```

Accepted SI restart boundary with finite reservoirs and counters.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L30-L81)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.completed_segments

*attribute* · *instance attribute*

```python
completed_segments: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L44-L44)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.coordinates

*attribute* · *instance attribute*

```python
coordinates: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L41-L41)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.counters

*attribute* · *instance attribute*

```python
counters: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L43-L43)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.height

*attribute* · *instance attribute*

```python
height: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L36-L36)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.identity

*attribute* · *instance attribute*

```python
identity: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L34-L34)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.masses

*attribute* · *instance attribute*

```python
masses: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L39-L39)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.next_step

*attribute* · *instance attribute*

```python
next_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L42-L42)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.pressure

*attribute* · *instance attribute*

```python
pressure: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L37-L37)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.temperature

*attribute* · *instance attribute*

```python
temperature: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L38-L38)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L35-L35)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.vapor

*attribute* · *instance attribute*

```python
vapor: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L40-L40)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.__init__

*method*

```python
def __init__(identity: dict, time: float, height: float, pressure: float, temperature: float, masses: np.ndarray, vapor: np.ndarray, coordinates: np.ndarray, next_step: float, counters: tuple[int, ...], completed_segments: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L0-L0)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.load

*method* · *classmethod*

```python
def load(path)
```

Read detached numeric arrays without initializing a device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L64-L81)

### particula.execution.multivapor_parcel.ParcelReplayCheckpoint.save

*method*

```python
def save(path)
```

Save an accepted parcel to an atomic numeric restart archive.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L46-L62)

## particula.execution.multivapor_parcel.ResidentMultiVaporParcel

*class*

```python
class ResidentMultiVaporParcel
```

Prepare one isolated device workspace for a constant-updraft parcel.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L84-L598)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._arrays

*attribute* · *instance attribute*

```python
_arrays = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L153-L153)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._counts

*attribute* · *instance attribute*

```python
_counts = upload(initial.number.reshape(1, -1))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L169-L169)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._density

*attribute* · *instance attribute*

```python
_density = upload(initial.density)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L172-L172)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._lanes

*attribute* · *instance attribute*

```python
_lanes = upload(self.lanes, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L171-L171)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._m0

*attribute* · *instance attribute*

```python
_m0 = upload(initial.masses)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L167-L167)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._props

*attribute* · *instance attribute*

```python
_props = upload(np.column_stack((self.rhs.gas_r, self.rhs.latent, self.rhs.p0, self.rhs.t0, self.rhs.diffusivity * self.rhs.enabled)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L173-L183)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._q0

*attribute* · *instance attribute*

```python
_q0 = upload(initial.vapor)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L168-L168)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._scale

*attribute* · *instance attribute*

```python
_scale = upload(self.scale)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L170-L170)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.binding

*attribute* · *instance attribute*

```python
binding = initial.activity_strategy.bind_device((1, nclass, nmat), self.lanes, self.device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L144-L146)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.boundary_readbacks

*attribute* · *instance attribute*

```python
boundary_readbacks = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L124-L124)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.clock

*attribute* · *instance attribute*

```python
clock = upload([0.0, self.timestep, self.timestep, self.timestep])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L185-L185)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.cond

*attribute* · *instance attribute*

```python
cond = zeros(1, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L187-L187)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.controls

*attribute* · *instance attribute*

```python
controls = adaptive
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L119-L119)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.counters

*attribute* · *instance attribute*

```python
counters = zeros(5, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L186-L186)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.delta

*attribute* · *instance attribute*

```python
delta = zeros(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L201-L201)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.device

*attribute* · *instance attribute*

```python
device = wp.get_device(device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L108-L108)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.graph

*attribute* · *instance attribute*

```python
graph = capture.graph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L215-L215)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.initial

*attribute* · *instance attribute*

```python
initial = initial
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L118-L118)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.initial_y

*attribute* · *instance attribute*

```python
initial_y = np.r_[1.0, 1.0, initial.masses[:, self.lanes] / self.scale.ravel()]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L141-L143)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.invalid

*attribute* · *instance attribute*

```python
invalid = zeros(1, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L188-L188)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.jac

*attribute* · *instance attribute*

```python
jac = zeros((self.n, self.n))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L199-L199)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.jac_base

*attribute* · *instance attribute*

```python
jac_base = zeros(self.n) if getattr(self.binding, 'hybrid', False) else self.f0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L203-L207)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.lanes

*attribute* · *instance attribute*

```python
lanes = self.rhs.targets
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L127-L127)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.masses

*attribute* · *instance attribute*

```python
masses = zeros((1, nclass, nmat))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L193-L193)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.n

*attribute* · *instance attribute*

```python
n = 2 + nclass * nv
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L131-L131)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.nclass

*attribute* · *instance attribute*

```python
nclass = nclass
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L130-L130)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.phase_exchange

*attribute* · *instance attribute*

```python
phase_exchange = bool(phase_exchange)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L122-L122)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.pivots

*attribute* · *instance attribute*

```python
pivots = zeros(self.n, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L200-L200)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.replay_seconds

*attribute* · *instance attribute*

```python
replay_seconds = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L125-L125)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.rhs

*attribute* · *instance attribute*

```python
rhs = ParcelTendencies(initial, updraft, phase_exchange)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L126-L126)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.ros

*attribute* · *instance attribute*

```python
ros = isinstance(adaptive, ParcelROS3)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L132-L132)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.scale

*attribute* · *instance attribute*

```python
scale = np.maximum(initial.masses[:, self.lanes], np.maximum(initial.masses.sum(axis=1, keepdims=True) * 1e-06, 1e-30))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L137-L140)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.scratch_bytes

*attribute* · *instance attribute*

```python
scratch_bytes = sum((a.size * wp.types.type_size_in_bytes(a.dtype) for a in self._arrays))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L216-L218)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.segments

*attribute* · *instance attribute*

```python
segments = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L123-L123)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.time

*attribute* · *instance attribute*

```python
time = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L117-L117)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.timestep

*attribute* · *instance attribute*

```python
timestep = float(timestep)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L120-L120)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.updraft

*attribute* · *instance attribute*

```python
updraft = float(updraft)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L121-L121)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.vapor

*attribute* · *instance attribute*

```python
vapor = zeros(nv)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L192-L192)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.y

*attribute* · *instance attribute*

```python
y = upload(self.initial_y)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L184-L184)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.__init__

*method*

```python
def __init__(initial, *, device, adaptive, timestep, updraft, phase_exchange=True)
```

Allocate bounded scratch and capture the non-advancing trial.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L87-L218)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._decode

*method*

```python
def _decode(y)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L487-L499)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._fluxes

*method*

```python
def _fluxes()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L246-L264)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._iteration

*method*

```python
def _iteration()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L437-L452)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._launch

*method*

```python
def _launch(kernel, n, *inputs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L220-L221)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._rate

*method*

```python
def _rate(state, out, stage, *, linearize=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L266-L333)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._trial

*method*

```python
def _trial()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L335-L435)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel._unpack

*method*

```python
def _unpack(state, stage)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L223-L244)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.advance

*method*

```python
def advance(stop)
```

Replay to an absolute time, then read observation and status.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L501-L523)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.checkpoint

*method*

```python
def checkpoint()
```

Copy an accepted boundary and next step for recapture.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L525-L547)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.identity

*method*

```python
def identity()
```

Record the material, thermophysical, solver and device identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L454-L485)

### particula.execution.multivapor_parcel.ResidentMultiVaporParcel.restore

*method*

```python
def restore(checkpoint)
```

Validate restart coordinates and identity before device writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L549-L598)

## particula.execution.multivapor_parcel.run_multivapor_cuda

*function*

```python
def run_multivapor_cuda(initial, time, updraft, phase_exchange, device, adaptive)
```

Run resident RK23 or ROS3 with observations at the requested times.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/multivapor_parcel.py#L601-L619)
