# particula.optics.gpu.shortwave_resident

*module*

Shortwave observations of live particles on the simulation stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L1-L1)

## particula.optics.gpu.shortwave_resident.F

*attribute* · *module attribute*

```python
F = wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L21-L21)

## particula.optics.gpu.shortwave_resident._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(F(pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L22-L22)

## particula.optics.gpu.shortwave_resident.ResidentShortwave

*class*

```python
class ResidentShortwave
```

Observe spectral radiation between steps of a prepared Warp runnable.

**Attributes**

- `workspace` (`ShortwaveWorkspace`) — Flux and broadband arrays on the simulation stream, available to subsequent device calculations.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L95-L306)

### particula.optics.gpu.shortwave_resident.ResidentShortwave._cases

*attribute* · *instance attribute*

```python
_cases = cases
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L256-L256)

### particula.optics.gpu.shortwave_resident.ResidentShortwave._combine

*attribute* · *instance attribute*

```python
_combine = wp.launch(_combine, active.depth.shape, inputs=[base, active, indices, dz, bulk, hg_order], device=dev, stream=stream, record_cmd=True)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L248-L255)

### particula.optics.gpu.shortwave_resident.ResidentShortwave._mie

*attribute* · *instance attribute*

```python
_mie = mie
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L257-L257)

### particula.optics.gpu.shortwave_resident.ResidentShortwave._owner

*attribute* · *instance attribute*

```python
_owner = simulation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L188-L188)

### particula.optics.gpu.shortwave_resident.ResidentShortwave._particles

*attribute* · *instance attribute*

```python
_particles = particles
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L259-L259)

### particula.optics.gpu.shortwave_resident.ResidentShortwave._reduce

*attribute* · *instance attribute*

```python
_reduce = wp.launch(_particle_depth_inputs, bulk.size, inputs=[cases.radius, cases.weight, mie.efficiencies, mie.status, bulk], device=dev, stream=stream, record_cmd=True)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L234-L247)

### particula.optics.gpu.shortwave_resident.ResidentShortwave._resources

*attribute* · *instance attribute*

```python
_resources = (base, indices, dz, bulk)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L258-L258)

### particula.optics.gpu.shortwave_resident.ResidentShortwave.workspace

*attribute* · *instance attribute*

```python
workspace = prepare_shortwave(backgrounds, sun, surface, device=dev, stream=stream, streams=streams, **transport_options)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L204-L212)

### particula.optics.gpu.shortwave_resident.ResidentShortwave.__init__

*method*

```python
def __init__(simulation, config, columns, sun, surface, *, box_indices, streams=8, hg_order=32, coefficient_capacity=512, particle_tile=256, **transport_options)
```

Bind live particle buffers and freeze atmospheric boundary inputs.

**Parameters**

- `simulation` (`PreparedSimulation`) — Active Warp PreparedSimulation with discrete or particle-resolved counts and a positive fixed timestep.
- `config` (`OpticalConfig`) — OpticalConfig for native particle Mie cross sections.
- `columns` (`SpectralColumn | Sequence[SpectralColumn]`) — Background SpectralColumn(s) excluding the live aerosol.
- `sun` (`SolarInput | Sequence[SolarInput]`) — Shared SolarInput or one per column.
- `surface` (`Surface | Sequence[Surface]`) — Shared surface boundary or one per column.
- `box_indices` (`ArrayLike`) — Integer (case, layer) map from top-first layers to simulation boxes. A value of -1 leaves that layer unchanged.
- `streams` (`int`) (default: `8`) — Angular ordinates per hemisphere.
- `hg_order` (`int`) (default: `32`) — Highest aerosol Henyey–Greenstein Legendre degree.
- `coefficient_capacity` (`int`) (default: `512`) — Maximum Mie series order in device scratch.
- `particle_tile` (`int`) (default: `256`) — Maximum particle/wavelength cases in Mie scratch.
- `**transport_options` (`dict`) (default: `{}`) — Additional prepare_shortwave options such as view_zenith_deg, case_tile and sensor_response.

**Raises**

- `ValueError` — If particle layout, indices or optical inputs conflict.
- `RuntimeError` — If the owner is closed or inside CUDA capture.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L103-L259)

### particula.optics.gpu.shortwave_resident.ResidentShortwave.check

*method*

```python
def check()
```

Synchronize the simulation and inspect optical status on its stream.

**Raises**

- `RuntimeError` — If no observation exists or physics or optics failed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L280-L290)

### particula.optics.gpu.shortwave_resident.ResidentShortwave.download

*method*

```python
def download()
```

Synchronize physics and optics and return detached ShortwaveFluxes.

**Returns**

- (`tuple[ShortwaveFlux, ...]`) — One result per column for the latest enqueue call.

**Raises**

- `RuntimeError` — An owner, Mie or radiation calculation failed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L292-L306)

### particula.optics.gpu.shortwave_resident.ResidentShortwave.enqueue

*method*

```python
def enqueue()
```

Sample current particle state without advancing or downloading it.

**Raises**

- `RuntimeError` — The simulation is closed, replaced or capturing.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L261-L278)

## particula.optics.gpu.shortwave_resident._combine

*function*

```python
def _combine(base: LayerData, active: LayerData, mapping: wp.array2d(dtype=int), thickness: wp.array2d(dtype=F), bulk: wp.array(dtype=wp.vec3d), order: int)
```

Add live aerosol to a fixed column using scattering-depth weights.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L51-L92)

## particula.optics.gpu.shortwave_resident._particle_depth_inputs

*function*

```python
def _particle_depth_inputs(radius: wp.array2d(dtype=F), weight: wp.array2d(dtype=F), efficiencies: wp.array2d(dtype=F), status: wp.array(dtype=int), bulk: wp.array(dtype=wp.vec3d))
```

Reduce extinction, scattering and its first moment per box/wavelength.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave_resident.py#L25-L48)
