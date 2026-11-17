# particula.gpu.kernels._brownian_parallel

*module*

Bounded-memory parallel properties and exact collision pair maxima.

Pair evaluation remains quadratic in the active cohort size. A fixed maximum
of 4096 reduction lanes per box bounds scratch independently of pair count.
Selection, acceptance, merge admission and random draws use the original
sequential device function.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L1-L1)

## particula.gpu.kernels._brownian_parallel._PreparedBrownianWork

*class* · *dataclass*

```python
class _PreparedBrownianWork
```

Retain all parallel scratch and exact launch inputs during setup.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L409-L512)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.active

*attribute* · *instance attribute*

```python
active: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L415-L415)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.counts

*attribute* · *instance attribute*

```python
counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L416-L416)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L419-L419)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.ion_data

*attribute* · *class attribute* · *instance attribute*

```python
ion_data: Any = field(default_factory=IonPairData)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L420-L420)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.ion_setup

*attribute* · *class attribute* · *instance attribute*

```python
ion_setup: Callable | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L421-L421)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.majorants

*attribute* · *instance attribute*

```python
majorants: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L418-L418)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.mean_density

*attribute* · *class attribute* · *instance attribute*

```python
mean_density: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L422-L422)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.partials

*attribute* · *instance attribute*

```python
partials: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L417-L417)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.property_inputs

*attribute* · *instance attribute*

```python
property_inputs: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L414-L414)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.selector_inputs

*attribute* · *instance attribute*

```python
selector_inputs: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L413-L413)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.__init__

*method*

```python
def __init__(selector_inputs: tuple[Any, ...], property_inputs: tuple[Any, ...], active: Any, counts: Any, partials: Any, majorants: Any, device: Any, ion_data: Any = IonPairData(), ion_setup: Callable | None = None, mean_density: Any = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L0-L0)

### particula.gpu.kernels._brownian_parallel._PreparedBrownianWork.enqueue

*method*

```python
def enqueue(failure)
```

Enqueue retained work without allocation or observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L424-L512)

## particula.gpu.kernels._brownian_parallel._charged_pair_maxima

*function*

```python
def _charged_pair_maxima(radii: wp.array2d(dtype=wp.float64), masses: wp.array2d(dtype=wp.float64), charges: wp.array2d(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), gas_constant: wp.float64, boltzmann_constant: wp.float64, molecular_weight_air: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64, elementary_charge_value: wp.float64, electric_permittivity: wp.float64, mechanism_mask: wp.int32, indices: wp.array2d(dtype=wp.int32), counts: wp.array(dtype=wp.int32), partials: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32), ion_data: IonPairData)
```

Tile the upper pair matrix across bounded lanes without storing it.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L242-L306)

## particula.gpu.kernels._brownian_parallel._compact_active

*function*

```python
def _compact_active(active: wp.array2d(dtype=wp.int32), indices: wp.array2d(dtype=wp.int32), counts: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32))
```

Preserve ascending source indices, independent of GPU scheduling.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L156-L176)

## particula.gpu.kernels._brownian_parallel._finish_maxima

*function*

```python
def _finish_maxima(partials: wp.array2d(dtype=wp.float64), majorants: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Reduce finite maxima without floating-point sum reordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L309-L324)

## particula.gpu.kernels._brownian_parallel._mean_material_density

*function*

```python
def _mean_material_density(masses: wp.array3d(dtype=wp.float64), density: wp.array(dtype=wp.float64), result: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Retain the native strategy's mean over positive-mass material slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L127-L153)

## particula.gpu.kernels._brownian_parallel._pair_maxima

*function*

```python
def _pair_maxima(radii: wp.array2d(dtype=wp.float64), diffusivities: wp.array2d(dtype=wp.float64), g_terms: wp.array2d(dtype=wp.float64), speeds: wp.array2d(dtype=wp.float64), settling_velocities: wp.array2d(dtype=wp.float64), mechanism_mask: wp.int32, indices: wp.array2d(dtype=wp.int32), counts: wp.array(dtype=wp.int32), partials: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Tile the upper pair matrix across bounded lanes without storing it.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L179-L239)

## particula.gpu.kernels._brownian_parallel._properties

*function*

```python
def _properties(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), gas_constant: wp.float64, boltzmann_constant: wp.float64, molecular_weight_air: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64, radii: wp.array2d(dtype=wp.float64), diffusivities: wp.array2d(dtype=wp.float64), g_terms: wp.array2d(dtype=wp.float64), speeds: wp.array2d(dtype=wp.float64), total_masses: wp.array2d(dtype=wp.float64), settling_velocities: wp.array2d(dtype=wp.float64), mechanism_mask: wp.int32, mean_density: wp.array(dtype=wp.float64), active: wp.array2d(dtype=wp.int32), failure: wp.array(dtype=wp.int32))
```

Evaluate the original transport formulas independently per slot.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L39-L124)

## particula.gpu.kernels._brownian_parallel._select_brownian

*function*

```python
def _select_brownian(masses: Any, concentration: Any, density: Any, volume: Any, temperature: Any, pressure: Any, gas_constant: Any, boltzmann_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, time_step: Any, radii: Any, diffusivities: Any, g_terms: Any, speeds: Any, settling_velocities: Any, turbulent_dissipation: Any, fluid_density: Any, total_masses: Any, charge: Any, active_indices: Any, collision_pairs: Any, n_collisions: Any, rng_states: Any, mechanism_mask: Any, collision_capacity: Any, active_counts: wp.array(dtype=wp.int32), majorants: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32), ion_data: IonPairData) -> None
```

Select from parallel properties and maxima with the original RNG code.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L327-L406)

## particula.gpu.kernels._brownian_parallel.prepare_brownian_work

*function*

```python
def prepare_brownian_work(selector_inputs, device)
```

Allocate O(BN) flags and at most 4096 fp64 maxima per box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_brownian_parallel.py#L515-L532)
