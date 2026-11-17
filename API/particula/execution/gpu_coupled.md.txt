# particula.execution.gpu_coupled

*module*

Adaptive GPU condensation with a fixed prepared event grid.

The default explicitly transfers state between stages. With ``resident=True``,
particle/gas state stays on CUDA and native conditional graphs own every
adaptive decision. Both modes retain prepared event buffers and RNG streams;
the host orchestrates fixed physical boundaries and requested observations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L1-L1)

## particula.execution.gpu_coupled.GPUCondensationControls

*class* · *dataclass*

```python
class GPUCondensationControls
```

Scalar adaptive controls, reset at every complete event interval.

Particle absolute tolerance is kg per representative; gas absolute
tolerance is kg/m³. Step durations are seconds. Error control covers
condensation only, not error from the enclosing sequential composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L38-L75)

### particula.execution.gpu_coupled.GPUCondensationControls.first_step

*attribute* · *class attribute* · *instance attribute*

```python
first_step: float = 1e-06
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L51-L51)

### particula.execution.gpu_coupled.GPUCondensationControls.gas_atol

*attribute* · *class attribute* · *instance attribute*

```python
gas_atol: float = 1e-12
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L50-L50)

### particula.execution.gpu_coupled.GPUCondensationControls.max_step

*attribute* · *class attribute* · *instance attribute*

```python
max_step: float = 1000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L53-L53)

### particula.execution.gpu_coupled.GPUCondensationControls.max_trials

*attribute* · *class attribute* · *instance attribute*

```python
max_trials: int = 100
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L54-L54)

### particula.execution.gpu_coupled.GPUCondensationControls.method

*attribute* · *class attribute* · *instance attribute*

```python
method: str = 'ros3'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L47-L47)

### particula.execution.gpu_coupled.GPUCondensationControls.min_step

*attribute* · *class attribute* · *instance attribute*

```python
min_step: float = 1e-14
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L52-L52)

### particula.execution.gpu_coupled.GPUCondensationControls.particle_atol

*attribute* · *class attribute* · *instance attribute*

```python
particle_atol: float = 1e-25
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L49-L49)

### particula.execution.gpu_coupled.GPUCondensationControls.rtol

*attribute* · *class attribute* · *instance attribute*

```python
rtol: float = 1e-05
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L48-L48)

### particula.execution.gpu_coupled.GPUCondensationControls.__init__

*method*

```python
def __init__(method: str = 'ros3', rtol: float = 1e-05, particle_atol: float = 1e-25, gas_atol: float = 1e-12, first_step: float = 1e-06, min_step: float = 1e-14, max_step: float = 1000.0, max_trials: int = 100) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L0-L0)

### particula.execution.gpu_coupled.GPUCondensationControls.__post_init__

*method*

```python
def __post_init__()
```

Reject invalid controls before physical state is allocated.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L56-L75)

## particula.execution.gpu_coupled.GPUCoupledSimulation

*class*

```python
class GPUCoupledSimulation
```

Advance one fixed event interval and one adaptive condensation flow.

``order`` is ``events-first`` or ``condensation-first``. Each event runs
once per cycle, regardless of rejected condensation trials. Temperature and
gas updates are explicit between-cycle boundaries. An error after any
physical work faults the owner; whole-cycle rollback is not promised.
Observations/checkpoints are available only between complete cycles.

**Parameters**

- `aerosol` — Native state, detached during setup.
- `events` — Nonempty tuple of supported native event runnables.
- `strategy` — Supported isothermal condensation strategy.
- `device` — Explicit Warp device, with no automatic fallback.
- `time_step` — Frozen event interval in seconds.
- `order` (default: `'events-first'`) — Sequential ordering of events and condensation.
- `controls` (default: `None`) — Scalar condensation error and timestep controls.
- `root_seed` (default: `0`) — Seed for persistent prepared event random streams.
- `logical_box_ids` (default: `None`) — Optional stable identifiers for those streams.
- `time` (default: `0.0`) — Initial physical time in seconds.
- `resident` (default: `False`) — Keep stage handoff and adaptive decisions on CUDA.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L121-L499)

### particula.execution.gpu_coupled.GPUCoupledSimulation._events

*attribute* · *instance attribute*

```python
_events = prepare_simulation(events, aerosol, device=device, time_step=time_step, root_seed=root_seed, logical_box_ids=logical_box_ids, time=time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L182-L190)

### particula.execution.gpu_coupled.GPUCoupledSimulation._molar_mass

*attribute* · *instance attribute*

```python
_molar_mass = aerosol.gas.molar_mass.copy()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L179-L179)

### particula.execution.gpu_coupled.GPUCoupledSimulation._vapor

*attribute* · *instance attribute*

```python
_vapor = plan.values['vapor']
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L178-L178)

### particula.execution.gpu_coupled.GPUCoupledSimulation.completed_steps

*property*

```python
completed_steps
```

Return successfully completed coupled cycles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L270-L273)

### particula.execution.gpu_coupled.GPUCoupledSimulation.controls

*property*

```python
controls
```

Return immutable condensation controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L233-L235)

### particula.execution.gpu_coupled.GPUCoupledSimulation.device

*property*

```python
device
```

Return the prepared event device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L254-L256)

### particula.execution.gpu_coupled.GPUCoupledSimulation.order

*property*

```python
order
```

Return the frozen sequential composition order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L228-L230)

### particula.execution.gpu_coupled.GPUCoupledSimulation.time

*property*

```python
time
```

Return the last complete cycle time, never an intermediate time.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L264-L267)

### particula.execution.gpu_coupled.GPUCoupledSimulation.time_step

*property*

```python
time_step
```

Return the frozen event interval [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L259-L261)

### particula.execution.gpu_coupled.GPUCoupledSimulation.__enter__

*method*

```python
def __enter__()
```

Enter the active owner's explicit resource scope.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L492-L495)

### particula.execution.gpu_coupled.GPUCoupledSimulation.__exit__

*method*

```python
def __exit__(*_exc)
```

Release event resources on normal exit and exceptions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L497-L499)

### particula.execution.gpu_coupled.GPUCoupledSimulation.__init__

*method*

```python
def __init__(aerosol, events, strategy, *, device, time_step, order='events-first', controls=None, root_seed=0, logical_box_ids=None, time=0.0, resident=False)
```

Validate both components and allocate persistent event resources.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L144-L196)

### particula.execution.gpu_coupled.GPUCoupledSimulation._accumulate_condensation_work

*method*

```python
def _accumulate_condensation_work(flow)
```

Record work without downloading physical state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L330-L336)

### particula.execution.gpu_coupled.GPUCoupledSimulation._condense

*method*

```python
def _condense(start, end)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L275-L328)

### particula.execution.gpu_coupled.GPUCoupledSimulation._configure

*method*

```python
def _configure(strategy, order, controls)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L217-L225)

### particula.execution.gpu_coupled.GPUCoupledSimulation._initialize_condensation

*method*

```python
def _initialize_condensation(aerosol, resident)
```

Allocate one resident adaptive owner when explicitly requested.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L198-L215)

### particula.execution.gpu_coupled.GPUCoupledSimulation._initialize_counters

*method*

```python
def _initialize_counters()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L237-L244)

### particula.execution.gpu_coupled.GPUCoupledSimulation._ready

*method*

```python
def _ready()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L246-L251)

### particula.execution.gpu_coupled.GPUCoupledSimulation._saturation

*method*

```python
def _saturation(gas, temperature)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L403-L416)

### particula.execution.gpu_coupled.GPUCoupledSimulation.capture_events

*method*

```python
def capture_events()
```

Capture only the fixed event sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L474-L478)

### particula.execution.gpu_coupled.GPUCoupledSimulation.checkpoint

*method*

```python
def checkpoint()
```

Explicitly download exact event/RNG continuation between cycles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L480-L486)

### particula.execution.gpu_coupled.GPUCoupledSimulation.close

*method*

```python
def close()
```

Release persistent event resources, including captured graphs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L488-L490)

### particula.execution.gpu_coupled.GPUCoupledSimulation.diagnostics

*method*

```python
def diagnostics()
```

Download box reductions and derive observed saturation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L428-L438)

### particula.execution.gpu_coupled.GPUCoupledSimulation.run

*method*

```python
def run(steps)
```

Complete a nonnegative integer number of fixed coupled cycles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L392-L401)

### particula.execution.gpu_coupled.GPUCoupledSimulation.snapshot

*method*

```python
def snapshot()
```

Explicitly download detached native state after a complete cycle.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L418-L426)

### particula.execution.gpu_coupled.GPUCoupledSimulation.step

*method*

```python
def step(bound=None)
```

Complete one fixed cycle with integer-grid bound admission.

Decimal bounds within eight local float spacings of the next nominal
grid point are admitted only when that tolerance is below a quarter
cycle. The event clock, physical interval and restart state are
unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L338-L390)

### particula.execution.gpu_coupled.GPUCoupledSimulation.update_environment

*method*

```python
def update_environment(temperature=None, pressure=None)
```

Explicitly upload between-cycle temperature [K] and pressure [Pa].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L440-L446)

### particula.execution.gpu_coupled.GPUCoupledSimulation.update_gas

*method*

```python
def update_gas(concentration)
```

Explicitly upload full-gas concentration [kg/m³] between cycles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L448-L452)

### particula.execution.gpu_coupled.GPUCoupledSimulation.update_gas_species

*method*

```python
def update_gas_species(concentrations=None, *, saturation_ratios=None)
```

Prescribe named gas targets while retaining other species on device.

``concentrations`` maps gas names to kg/m³; ``saturation_ratios`` maps
names to dimensionless ratios. Values are scalars or box vectors.
Saturation targets use the configured constant/Buck pressure law and
current device temperature. Each species may occur in only one mapping.
Validation precedes physical mutation. This explicit forcing boundary
uploads only prescribed targets and reads scalar validation status.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L454-L472)

## particula.execution.gpu_coupled._condensation_profile

*function*

```python
def _condensation_profile(aerosol, strategy, method)
```

Validate the shared GPU profile, including empty boundary populations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L107-L118)

## particula.execution.gpu_coupled._validate_events

*function*

```python
def _validate_events(processes, aerosol)
```

Exclude derived-field consumers and unsupported custom ownership.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled.py#L78-L104)
