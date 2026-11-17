# particula.execution.gpu_resident_condensation

*module*

Adaptive condensation with native CUDA graph control and resident state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L1-L1)

## particula.execution.gpu_resident_condensation.GPUResidentCondensation

*class*

```python
class GPUResidentCondensation(GPUCondensationODE)
```

Bases: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

Run RK23 or material-coupled ROS3 with all adaptive decisions on CUDA.

Construction uploads native state once and captures a conditional trial
loop without advancing it. ``advance(bound)`` launches that loop; accepted
state, clock, retry count and step size remain on device. Only completed
observation-boundary control metadata is downloaded. ``snapshot()`` is an
explicit full-state download. No Warp CPU or host-control fallback exists.

Environment is constant between explicit bounds. Arbitrary host forcing
callbacks are rejected; callers may update environment at segment bounds.
Tolerances and physical units match GPUCondensationODE. CUDA conditional
graph support (Toolkit and driver >=12.4) is required.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L14-L335)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._accepted

*attribute* · *instance attribute*

```python
_accepted = self._array(np.zeros(1, dtype=np.int32), wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L78-L78)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._condition

*attribute* · *instance attribute*

```python
_condition = self._array(np.zeros(1, dtype=np.int32), wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L73-L73)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._control

*attribute* · *instance attribute*

```python
_control = self._array(np.array([self.next_step, self.time, self.next_step, self.time, self.min_step, self.max_step, 0.0]))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L44-L56)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._counters

*attribute* · *instance attribute*

```python
_counters = self._array(np.zeros(5, dtype=np.int32), wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L74-L74)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._device_control

*attribute* · *instance attribute*

```python
_device_control = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L87-L87)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._graph

*attribute* · *instance attribute*

```python
_graph = capture.graph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L94-L94)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._physical_atol

*attribute* · *instance attribute*

```python
_physical_atol = self._array(np.concatenate((self._tolerance(kwargs.get('particle_atol', 1e-24), self._shape, 'particle_atol').ravel(), self._tolerance(kwargs.get('gas_atol', 1e-12), self._gas_shape, 'gas_atol').ravel())))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L57-L72)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._ros3

*attribute* · *instance attribute*

```python
_ros3 = BlockROS3Stages(self, pairs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L85-L85)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._sticky_invalid

*attribute* · *instance attribute*

```python
_sticky_invalid = self._array(np.zeros(1, dtype=np.int32), wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L75-L77)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.control_readbacks

*attribute* · *instance attribute*

```python
control_readbacks = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L95-L95)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._atol

*attribute* · *instance attribute*

```python
_atol = self._array(absolute)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L154-L154)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._base

*attribute* · *instance attribute*

```python
_base = _cpu_batch.detached(aerosol)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L86-L86)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._candidate

*attribute* · *instance attribute*

```python
_candidate = self._array(np.zeros(self._size))
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L156-L156)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._faulted

*attribute* · *instance attribute*

```python
_faulted = False
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L108-L108)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._forcing

*attribute* · *instance attribute*

```python
_forcing = forcing
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L88-L88)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._gas_shape

*attribute* · *instance attribute*

```python
_gas_shape = aerosol.gas.concentration.shape
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L112-L112)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._host_initial_physical

*attribute* · *instance attribute*

```python
_host_initial_physical = physical.copy()
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L148-L148)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._host_initial_scaled

*attribute* · *instance attribute*

```python
_host_initial_scaled = physical / self._host_scale
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L149-L149)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._host_scale

*attribute* · *instance attribute*

```python
_host_scale = np.concatenate((np.broadcast_to(np.maximum(aerosol.particles.masses.sum(axis=2, keepdims=True), 1e-30), self._shape).ravel(), np.maximum(aerosol.gas.concentration, 1e-12).ravel()))
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L115-L126)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._initial_physical

*attribute* · *instance attribute*

```python
_initial_physical = self._array(self._host_initial_physical)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L150-L150)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._initial_scaled

*attribute* · *instance attribute*

```python
_initial_scaled = self._array(self._host_initial_scaled)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L151-L151)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._invalid

*attribute* · *instance attribute*

```python
_invalid = self._array(np.zeros(1, dtype=np.int32), wp.int32)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L159-L159)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._mass_size

*attribute* · *instance attribute*

```python
_mass_size = aerosol.particles.masses.size
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L113-L113)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._norm

*attribute* · *instance attribute*

```python
_norm = self._array(np.zeros(1))
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L160-L160)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._plan

*attribute* · *instance attribute*

```python
_plan = plan_adaptive_condensation(self._strategy, self._base)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L89-L89)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._rates

*attribute* · *instance attribute*

```python
_rates = tuple((self._array(np.zeros(self._size)) for _ in range(4)))
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L158-L158)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._rtol

*attribute* · *instance attribute*

```python
_rtol = self._array(np.full(self._size, relative))
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L155-L155)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._scale

*attribute* · *instance attribute*

```python
_scale = self._array(self._host_scale)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L153-L153)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._shape

*attribute* · *instance attribute*

```python
_shape = aerosol.particles.masses.shape
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L111-L111)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._size

*attribute* · *instance attribute*

```python
_size = self._mass_size + aerosol.gas.concentration.size
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L114-L114)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._stage

*attribute* · *instance attribute*

```python
_stage = self._array(np.zeros(self._size))
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L157-L157)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._state

*attribute* · *instance attribute*

```python
_state = self._array(self._host_initial_scaled)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L152-L152)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._strategy

*attribute* · *instance attribute*

```python
_strategy = deepcopy(strategy)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L87-L87)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._time

*attribute* · *instance attribute*

```python
_time = AdaptiveSolver._scalar(time, 'time', positive=False)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L95-L95)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.device

*attribute* · *instance attribute*

```python
device = wp.get_device(device)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L109-L109)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.device_storage_bytes

*attribute* · *instance attribute*

```python
device_storage_bytes = 0
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L147-L147)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.last_error

*attribute* · *instance attribute*

```python
last_error = None
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L106-L106)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.max_step

*attribute* · *instance attribute*

```python
max_step = AdaptiveSolver._scalar(max_step, 'max_step')
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L97-L97)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.max_trials

*attribute* · *instance attribute*

```python
max_trials = max_trials
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L103-L103)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.method

*attribute* · *instance attribute*

```python
method = method
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L104-L104)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.min_step

*attribute* · *instance attribute*

```python
min_step = AdaptiveSolver._scalar(min_step, 'min_step')
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L96-L96)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.next_step

*attribute* · *instance attribute*

```python
next_step = AdaptiveSolver._scalar(first_step, 'first_step')
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L98-L98)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.statistics

*attribute* · *instance attribute*

```python
statistics = StepStatistics()
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L176-L176)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.time

*property*

```python
time
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

Return the accepted shared physical time [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L452-L454)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.__init__

*method*

```python
def __init__(aerosol, strategy, *, device, method='ros3', **kwargs)
```

Upload fixed-shape buffers and capture the adaptive trial loop.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L29-L95)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._device_trial

*method*

```python
def _device_trial()
```

Build a trial without acceptance, host scalar reads or uploads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L104-L181)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._export_to_session

*method*

```python
def _export_to_session(session)
```

Commit accepted physical masses/gas to the same event buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L318-L322)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._iteration

*method*

```python
def _iteration()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L183-L207)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._reset_coordinates

*method*

```python
def _reset_coordinates(time, first_step)
```

Reset scaled coordinates and control using device state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L287-L316)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._reset_from_session

*method*

```python
def _reset_from_session(session, time, first_step)
```

Import event buffers on device at a complete split boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L256-L268)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._status

*method*

```python
def _status()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L97-L102)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.advance

*method*

```python
def advance(bound)
```

Reach an explicit observation/forcing bound [s] entirely on CUDA.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L209-L250)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.restart_controller

*method*

```python
def restart_controller(first_step)
```

Reset scaling and step history on CUDA at an accepted boundary.

``first_step`` is in seconds, within this owner's timestep limits.
Physical state and accepted time are retained; per-interval statistics
reset. This matches a split flow's fresh-controller policy without
downloading or re-uploading physical state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L270-L285)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.snapshot

*method*

```python
def snapshot()
```

Explicitly download accepted native state and current environment.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L324-L335)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation.step

*method*

```python
def step(bound)
```

Advance to the bound, including all internal accepted steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_resident_condensation.py#L252-L254)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._allocate_physics

*method*

```python
def _allocate_physics()
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L199-L240)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._array

*method*

```python
def _array(values, dtype=wp.float64)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L192-L197)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._check_ready

*method*

```python
def _check_ready()
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L242-L250)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._combine

*method*

```python
def _combine(h, coefficients, output)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L407-L418)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._environment

*method*

```python
def _environment(time)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L262-L274)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._launch

*method*

```python
def _launch(kernel, dim, inputs)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L252-L253)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._rate

*method*

```python
def _rate(time, state, result)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L307-L405)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._step

*method*

```python
def _step(bound)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

Advance within one smooth forcing interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L463-L506)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._tolerance

*method* · *staticmethod*

```python
def _tolerance(value, shape, name)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L179-L190)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._trial

*method*

```python
def _trial(h)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L420-L449)

### particula.execution.gpu_resident_condensation.GPUResidentCondensation._unpack_trial

*method*

```python
def _unpack_trial(state)
```

Inherited from: `particula.execution.gpu_condensation_ode.GPUCondensationODE`

Validate and unpack device coordinates without a physical rate.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_condensation_ode.py#L276-L305)
