# particula.coupling.flows

*module*

Flow strategies for point observations and accepted grid intervals.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L1-L1)

## particula.coupling.flows.DirectionalMixing

*class* · *dataclass*

```python
class DirectionalMixing
```

Prescribed directional K in m²/s along x, y and layer coordinates.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L282-L297)

### particula.coupling.flows.DirectionalMixing.diffusivity

*attribute* · *class attribute* · *instance attribute*

```python
diffusivity: tuple = (0.0, 0.0, 0.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L286-L286)

### particula.coupling.flows.DirectionalMixing.__init__

*method*

```python
def __init__(diffusivity: tuple = (0.0, 0.0, 0.0)) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L0-L0)

### particula.coupling.flows.DirectionalMixing.__post_init__

*method*

```python
def __post_init__()
```

Require three finite nonnegative coefficients.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L288-L293)

### particula.coupling.flows.DirectionalMixing.coefficient

*method*

```python
def coefficient(flow, grid, ref, axis)
```

Return K for one face direction.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L295-L297)

## particula.coupling.flows.DuneFlowStrategy

*class* · *dataclass*

```python
class DuneFlowStrategy(SpatialFlow)
```

Bases: `particula.coupling.flows.SpatialFlow`

Adapt the moving-parcel dune to point and terrain-grid sampling.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L226-L279)

### particula.coupling.flows.DuneFlowStrategy.dune

*attribute* · *instance attribute*

```python
dune: DuneFlow
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L230-L230)

### particula.coupling.flows.DuneFlowStrategy.__init__

*method*

```python
def __init__(dune: DuneFlow) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L0-L0)

### particula.coupling.flows.DuneFlowStrategy.parcel_trace

*method*

```python
def parcel_trace(parcel_id, xyz, time, aerosol, *, provenance)
```

Use the terrain-coordinate integrator on continuous streamlines.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L232-L236)

### particula.coupling.flows.DuneFlowStrategy.reference_air

*method*

```python
def reference_air(grid, coord, time, device)
```

Evaluate the same streamlines on the piecewise-linear grid bed.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L250-L279)

### particula.coupling.flows.DuneFlowStrategy.sample

*method*

```python
def sample(xyz, time, *, device='cpu')
```

Sample the continuous Gaussian dune at physical xyz.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L238-L248)

### particula.coupling.flows.DuneFlowStrategy.interval

*method*

```python
def interval(grid, start, end, mixing, *, device='cpu')
```

Inherited from: `particula.coupling.flows.SpatialFlow`

Integrate steady face volume flux and mixing across one interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L89-L133)

### particula.coupling.flows.DuneFlowStrategy.trajectory

*method*

```python
def trajectory(parcel_id)
```

Inherited from: `particula.coupling.flows.FlowStrategy`

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

## particula.coupling.flows.FlowStrategy

*class*

```python
class FlowStrategy
```

A forcing provider exposing the operations its data support.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L29-L38)

### particula.coupling.flows.FlowStrategy.interval

*method*

```python
def interval(grid, start, end, mixing, *, device='cpu')
```

Offer a grid interval or reject an incompatible provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L32-L34)

### particula.coupling.flows.FlowStrategy.trajectory

*method*

```python
def trajectory(parcel_id)
```

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

## particula.coupling.flows.GridInterval

*class* · *dataclass*

```python
class GridInterval
```

Accepted forcing. Face values are outward signed m³ and mixing m³.

air columns are u,v,w (m/s), T (K), p (Pa), vapor (kg/kg dry air).
speed contains cell-centered reference-coordinate velocities (m/s).

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L12-L26)

### particula.coupling.flows.GridInterval.air

*attribute* · *instance attribute*

```python
air: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L22-L22)

### particula.coupling.flows.GridInterval.diffusivity

*attribute* · *instance attribute*

```python
diffusivity: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L26-L26)

### particula.coupling.flows.GridInterval.end

*attribute* · *instance attribute*

```python
end: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L21-L21)

### particula.coupling.flows.GridInterval.mixing

*attribute* · *instance attribute*

```python
mixing: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L24-L24)

### particula.coupling.flows.GridInterval.speed

*attribute* · *instance attribute*

```python
speed: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L25-L25)

### particula.coupling.flows.GridInterval.start

*attribute* · *instance attribute*

```python
start: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L20-L20)

### particula.coupling.flows.GridInterval.transfer

*attribute* · *instance attribute*

```python
transfer: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L23-L23)

### particula.coupling.flows.GridInterval.__init__

*method*

```python
def __init__(start: float, end: float, air: np.ndarray, transfer: np.ndarray, mixing: np.ndarray, speed: np.ndarray, diffusivity: np.ndarray) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L0-L0)

## particula.coupling.flows.SoundingFlow

*class* · *dataclass*

```python
class SoundingFlow(SpatialFlow)
```

Bases: `particula.coupling.flows.SpatialFlow`

Steady horizontally uniform measured or synthetic vertical profile.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L183-L195)

### particula.coupling.flows.SoundingFlow.sounding

*attribute* · *instance attribute*

```python
sounding: Sounding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L187-L187)

### particula.coupling.flows.SoundingFlow.__init__

*method*

```python
def __init__(sounding: Sounding) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L0-L0)

### particula.coupling.flows.SoundingFlow.sample

*method*

```python
def sample(xyz, time, *, device='cpu')
```

Interpolate wind and thermodynamics against geometric height.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L189-L195)

### particula.coupling.flows.SoundingFlow.interval

*method*

```python
def interval(grid, start, end, mixing, *, device='cpu')
```

Inherited from: `particula.coupling.flows.SpatialFlow`

Integrate steady face volume flux and mixing across one interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L89-L133)

### particula.coupling.flows.SoundingFlow.parcel_trace

*method*

```python
def parcel_trace(parcel_id, xyz, time, aerosol, *, provenance)
```

Inherited from: `particula.coupling.flows.SpatialFlow`

Follow air with the point sampler and build a native replay trace.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L48-L81)

### particula.coupling.flows.SoundingFlow.reference_air

*method*

```python
def reference_air(grid, coord, time, device)
```

Inherited from: `particula.coupling.flows.SpatialFlow`

Sample physical fields on a flat grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L83-L87)

### particula.coupling.flows.SoundingFlow.trajectory

*method*

```python
def trajectory(parcel_id)
```

Inherited from: `particula.coupling.flows.FlowStrategy`

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

## particula.coupling.flows.SpatialFlow

*class*

```python
class SpatialFlow(FlowStrategy, ABC)
```

Bases: `particula.coupling.flows.FlowStrategy`, `ABC`

Spatial air sampling and fixed-grid face quadrature.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L41-L133)

### particula.coupling.flows.SpatialFlow.interval

*method*

```python
def interval(grid, start, end, mixing, *, device='cpu')
```

Integrate steady face volume flux and mixing across one interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L89-L133)

### particula.coupling.flows.SpatialFlow.parcel_trace

*method*

```python
def parcel_trace(parcel_id, xyz, time, aerosol, *, provenance)
```

Follow air with the point sampler and build a native replay trace.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L48-L81)

### particula.coupling.flows.SpatialFlow.reference_air

*method*

```python
def reference_air(grid, coord, time, device)
```

Sample physical fields on a flat grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L83-L87)

### particula.coupling.flows.SpatialFlow.sample

*method* · *abstract*

```python
def sample(xyz, time, *, device='cpu')
```

Sample the six air fields at (n,3) points in metres and seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L44-L46)

### particula.coupling.flows.SpatialFlow.trajectory

*method*

```python
def trajectory(parcel_id)
```

Inherited from: `particula.coupling.flows.FlowStrategy`

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

## particula.coupling.flows.StabilityMixing

*class* · *dataclass*

```python
class StabilityMixing
```

Effective transverse K = U/2 d(sigma²)/dx for a flat rural plume.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L300-L339)

### particula.coupling.flows.StabilityMixing.source_x

*attribute* · *instance attribute*

```python
source_x: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L305-L305)

### particula.coupling.flows.StabilityMixing.stability

*attribute* · *instance attribute*

```python
stability: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L304-L304)

### particula.coupling.flows.StabilityMixing.__init__

*method*

```python
def __init__(stability: str, source_x: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L0-L0)

### particula.coupling.flows.StabilityMixing.coefficient

*method*

```python
def coefficient(flow, grid, ref, axis)
```

Differentiate the Briggs open-country variance at positive fetch.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L307-L339)

## particula.coupling.flows.UniformFlow

*class* · *dataclass*

```python
class UniformFlow(SpatialFlow)
```

Bases: `particula.coupling.flows.SpatialFlow`

Uniform velocity (m/s), T (K), p (Pa) and dry-air vapor ratio.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L148-L180)

### particula.coupling.flows.UniformFlow.pressure

*attribute* · *instance attribute*

```python
pressure: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L154-L154)

### particula.coupling.flows.UniformFlow.temperature

*attribute* · *instance attribute*

```python
temperature: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L153-L153)

### particula.coupling.flows.UniformFlow.vapor

*attribute* · *instance attribute*

```python
vapor: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L155-L155)

### particula.coupling.flows.UniformFlow.velocity

*attribute* · *instance attribute*

```python
velocity: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L152-L152)

### particula.coupling.flows.UniformFlow.__init__

*method*

```python
def __init__(velocity: tuple, temperature: float, pressure: float, vapor: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L0-L0)

### particula.coupling.flows.UniformFlow.__post_init__

*method*

```python
def __post_init__()
```

Check velocity and positive thermodynamic inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L157-L170)

### particula.coupling.flows.UniformFlow.sample

*method*

```python
def sample(xyz, time, *, device='cpu')
```

Broadcast uniform air to each world point.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L172-L180)

### particula.coupling.flows.UniformFlow.interval

*method*

```python
def interval(grid, start, end, mixing, *, device='cpu')
```

Inherited from: `particula.coupling.flows.SpatialFlow`

Integrate steady face volume flux and mixing across one interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L89-L133)

### particula.coupling.flows.UniformFlow.parcel_trace

*method*

```python
def parcel_trace(parcel_id, xyz, time, aerosol, *, provenance)
```

Inherited from: `particula.coupling.flows.SpatialFlow`

Follow air with the point sampler and build a native replay trace.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L48-L81)

### particula.coupling.flows.UniformFlow.reference_air

*method*

```python
def reference_air(grid, coord, time, device)
```

Inherited from: `particula.coupling.flows.SpatialFlow`

Sample physical fields on a flat grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L83-L87)

### particula.coupling.flows.UniformFlow.trajectory

*method*

```python
def trajectory(parcel_id)
```

Inherited from: `particula.coupling.flows.FlowStrategy`

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

## particula.coupling.flows._points

*function*

```python
def _points(xyz, time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L136-L145)

## particula.coupling.flows._profile

*function*

```python
def _profile(sounding, z, eta, jac, gradient, device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L198-L223)
