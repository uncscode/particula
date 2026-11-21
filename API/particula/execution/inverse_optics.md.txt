# particula.execution.inverse_optics

*module*

Bulk optical losses for the smooth closed-condensation inverse model.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L1-L1)

## particula.execution.inverse_optics._OpticalGrowth

*class*

```python
class _OpticalGrowth
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L74-L146)

### particula.execution.inverse_optics._OpticalGrowth.count

*attribute* · *instance attribute*

```python
count = self.upload(numbers)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L86-L86)

### particula.execution.inverse_optics._OpticalGrowth.density

*attribute* · *instance attribute*

```python
density = self.upload(np.repeat(numbers, wave.size, axis=0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L97-L97)

### particula.execution.inverse_optics._OpticalGrowth.device

*attribute* · *instance attribute*

```python
device = device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L78-L78)

### particula.execution.inverse_optics._OpticalGrowth.grad

*attribute* · *instance attribute*

```python
grad = grad
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L79-L79)

### particula.execution.inverse_optics._OpticalGrowth.grid

*attribute* · *instance attribute*

```python
grid = (len(radii) * wave.size, radii.shape[1])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L87-L87)

### particula.execution.inverse_optics._OpticalGrowth.index

*attribute* · *instance attribute*

```python
index = wp.array(np.ascontiguousarray(np.stack((indices.real, indices.imag), axis=-1).reshape((*self.grid, 2))), dtype=wp.vec2d, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L88-L96)

### particula.execution.inverse_optics._OpticalGrowth.initial

*attribute* · *instance attribute*

```python
initial = self.empty(radii.shape)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L84-L84)

### particula.execution.inverse_optics._OpticalGrowth.mie

*attribute* · *instance attribute*

```python
mie = BulkOpticsAdjoint(self.grid, np.tile(wave, len(radii)), device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L98-L100)

### particula.execution.inverse_optics._OpticalGrowth.output_shape

*attribute* · *instance attribute*

```python
output_shape = (radii.shape[0], wave.size, 3)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L82-L82)

### particula.execution.inverse_optics._OpticalGrowth.radius

*attribute* · *instance attribute*

```python
radius = self.upload(radii, grad=grad)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L83-L83)

### particula.execution.inverse_optics._OpticalGrowth.transport

*attribute* · *instance attribute*

```python
transport = transport
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L80-L80)

### particula.execution.inverse_optics._OpticalGrowth.water

*attribute* · *instance attribute*

```python
water = self.upload(vapor)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L85-L85)

### particula.execution.inverse_optics._OpticalGrowth.waves

*attribute* · *instance attribute*

```python
waves = wave.size
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L81-L81)

### particula.execution.inverse_optics._OpticalGrowth.__init__

*method*

```python
def __init__(radii, numbers, vapor, wave, indices, transport, grad, device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L75-L100)

### particula.execution.inverse_optics._OpticalGrowth.empty

*method*

```python
def empty(shape)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L107-L113)

### particula.execution.inverse_optics._OpticalGrowth.initialize

*method*

```python
def initialize()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L118-L123)

### particula.execution.inverse_optics._OpticalGrowth.launch

*method*

```python
def launch(kernel, dim, args)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L115-L116)

### particula.execution.inverse_optics._OpticalGrowth.observe

*method*

```python
def observe(mass, tape=None)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L125-L132)

### particula.execution.inverse_optics._OpticalGrowth.segment

*method*

```python
def segment(mass, start, end, visit)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L134-L146)

### particula.execution.inverse_optics._OpticalGrowth.upload

*method*

```python
def upload(data, *, grad=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L102-L105)

## particula.execution.inverse_optics._observation_inputs

*function*

```python
def _observation_inputs(observation_steps, wavelength, index, steps, shape, adjoint)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L31-L71)

## particula.execution.inverse_optics._optical_radii

*function*

```python
def _optical_radii(mass: FloatMatrix, coefficient: wp.float64, waves: int, output: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L17-L28)

## particula.execution.inverse_optics._reverse

*function*

```python
def _reverse(model, adjoint, initial_tape, first, checkpoints, selection)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L238-L262)

## particula.execution.inverse_optics.condensation_optics

*function*

```python
def condensation_optics(initial_um, number_density, vapor, *, wavelength, refractive_index, observation_steps, density, mean_path, diffusivity, duration, steps, device='cuda:0', adjoint=None, segment_steps=None)
```

Evaluate a closed-box bulk optical trajectory and optional initial VJP.

Physics inputs and fixed-step restrictions match terminal_radius().
Wavelengths [m] are shared across boxes. Absolute refractive indices
broadcast to (box, wavelength, slot) and stay prescribed. Outputs
are in m⁻¹ on (observation, box, wavelength, channel), with channels
extinction, scattering and absorption in that order. Supply optical
adjoint weights in the same layout for a gradient per initial radius
measured in micrometres. The reverse pass replays checkpoint segments
and uses bounded Mie scratch at each optical observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_optics.py#L149-L235)
