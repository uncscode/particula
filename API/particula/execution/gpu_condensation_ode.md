# particula.execution.gpu_condensation_ode

*module*

Native Warp condensation with device-resident RK3(2) trial state.

Construction explicitly uploads detached native state/configuration. Each
step uses a host-controlled shared maximum-error clock and downloads only
scalar validation/error reductions; snapshots explicitly download accepted
state. This is not CUDA-capturable or a CPU fallback for requested CUDA.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L1-L1)

## particula.execution.gpu_condensation_ode.GPUCondensationODE

*class*

```python
class GPUCondensationODE
```

Own native condensation on an explicitly requested Warp device.

particle_atol is kg per particle; gas_atol is kg/m3. Scalar or full native
shapes are accepted. rtol is a dimensionless scalar. All boxes share one
accepted time and a maximum error, including small species. The physical
profile is the existing mapped constant/Buck GPU rate with one volatile
material; zero constant vapor pressure also represents involatile uptake.
Native discrete and resolved measures are supported; PDF and latent heat
are rejected. Full active-slot evaporation remains a domain boundary.

Optional forcing is a pure time -> (temperature[K], pressure[Pa]) callback.
Every stage explicitly uploads those box values. Discontinuities require
explicit step boundaries. No arbitrary/stochastic runnable is adapted.
Capture is unsupported because each trial reads control reductions on host.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L29-L464)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._atol

*attribute* · *instance attribute*

```python
_atol = self._array(absolute)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L143-L143)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._base

*attribute* · *instance attribute*

```python
_base = _cpu_batch.detached(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L82-L82)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._candidate

*attribute* · *instance attribute*

```python
_candidate = self._array(np.zeros(self._size))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L145-L145)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._faulted

*attribute* · *instance attribute*

```python
_faulted = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L101-L101)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._forcing

*attribute* · *instance attribute*

```python
_forcing = forcing
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L84-L84)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._gas_shape

*attribute* · *instance attribute*

```python
_gas_shape = aerosol.gas.concentration.shape
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L105-L105)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._host_scale

*attribute* · *instance attribute*

```python
_host_scale = np.concatenate((np.broadcast_to(np.maximum(aerosol.particles.masses.sum(axis=2, keepdims=True), 1e-30), self._shape).ravel(), np.maximum(aerosol.gas.concentration, 1e-12).ravel()))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L108-L119)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._invalid

*attribute* · *instance attribute*

```python
_invalid = self._array(np.zeros(1, dtype=np.int32), wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L148-L148)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._mass_size

*attribute* · *instance attribute*

```python
_mass_size = aerosol.particles.masses.size
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L106-L106)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._norm

*attribute* · *instance attribute*

```python
_norm = self._array(np.zeros(1))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L149-L149)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._plan

*attribute* · *instance attribute*

```python
_plan = plan_condensation(MassCondensation(self._strategy), self._base, allow_zero_vapor=True)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L85-L87)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._rates

*attribute* · *instance attribute*

```python
_rates = tuple((self._array(np.zeros(self._size)) for _ in range(4)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L147-L147)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._rtol

*attribute* · *instance attribute*

```python
_rtol = self._array(np.full(self._size, relative))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L144-L144)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._scale

*attribute* · *instance attribute*

```python
_scale = self._array(self._host_scale)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L142-L142)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._shape

*attribute* · *instance attribute*

```python
_shape = aerosol.particles.masses.shape
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L104-L104)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._size

*attribute* · *instance attribute*

```python
_size = self._mass_size + aerosol.gas.concentration.size
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L107-L107)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._stage

*attribute* · *instance attribute*

```python
_stage = self._array(np.zeros(self._size))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L146-L146)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._state

*attribute* · *instance attribute*

```python
_state = self._array(physical / self._host_scale)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L141-L141)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._strategy

*attribute* · *instance attribute*

```python
_strategy = deepcopy(strategy)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L83-L83)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._time

*attribute* · *instance attribute*

```python
_time = AdaptiveSolver._scalar(time, 'time', positive=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L88-L88)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.control_readbacks

*attribute* · *instance attribute*

```python
control_readbacks = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L154-L154)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.device

*attribute* · *instance attribute*

```python
device = wp.get_device(device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L102-L102)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.device_storage_bytes

*attribute* · *instance attribute*

```python
device_storage_bytes = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L140-L140)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.last_error

*attribute* · *instance attribute*

```python
last_error = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L99-L99)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.max_step

*attribute* · *instance attribute*

```python
max_step = AdaptiveSolver._scalar(max_step, 'max_step')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L90-L90)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.max_trials

*attribute* · *instance attribute*

```python
max_trials = max_trials
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L96-L96)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.method

*attribute* · *instance attribute*

```python
method = method
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L97-L97)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.min_step

*attribute* · *instance attribute*

```python
min_step = AdaptiveSolver._scalar(min_step, 'min_step')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L89-L89)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.next_step

*attribute* · *instance attribute*

```python
next_step = AdaptiveSolver._scalar(first_step, 'first_step')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L91-L91)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.statistics

*attribute* · *instance attribute*

```python
statistics = StepStatistics()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L153-L153)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.time

*property*

```python
time
```

Return the accepted shared physical time [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L399-L401)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.__init__

*method*

```python
def __init__(aerosol, strategy, *, device, method='rk23', time=0.0, particle_atol=1e-24, gas_atol=1e-12, rtol=1e-05, first_step=0.001, min_step=1e-14, max_step=1.0, max_trials=100, forcing=None)
```

Validate physics and explicitly upload fixed buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L46-L154)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._allocate_physics

*method*

```python
def _allocate_physics()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L176-L210)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._array

*method*

```python
def _array(values, dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L169-L174)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._check_ready

*method*

```python
def _check_ready()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L212-L220)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._combine

*method*

```python
def _combine(h, coefficients, output)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L356-L367)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._environment

*method*

```python
def _environment(time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L232-L244)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._launch

*method*

```python
def _launch(kernel, dim, inputs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L222-L223)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._rate

*method*

```python
def _rate(time, state, result)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L246-L354)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._status

*method*

```python
def _status()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L225-L230)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._tolerance

*method* · *staticmethod*

```python
def _tolerance(value, shape, name)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L156-L167)

### particula.execution.gpu_condensation_ode.GPUCondensationODE._trial

*method*

```python
def _trial(h)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L369-L396)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.snapshot

*method*

```python
def snapshot()
```

Download detached accepted native state, never trial buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L450-L464)

### particula.execution.gpu_condensation_ode.GPUCondensationODE.step

*method*

```python
def step(bound)
```

Accept one bounded device trial with scalar control readbacks.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L403-L448)
