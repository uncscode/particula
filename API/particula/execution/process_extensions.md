# particula.execution.process_extensions

*module*

Typed public preparation contracts for explicit resident process adapters.

Adapters are trusted numerical code. Declarations constrain exposed resources;
Python objects and device pointers are not a security boundary. Preparation may
allocate scratch. The retained operation must only enqueue work, check the
shared device failure flag, and preserve the fixed storage schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L1-L1)

## particula.execution.process_extensions.GPUPreparationContext

*class* · *dataclass*

```python
class GPUPreparationContext
```

Expose only declared buffers plus explicit resident execution resources.

``clock[0]`` is the absolute start time for the current timestep in seconds.
``failure[0]`` is zero while healthy. Kernels must suppress work
after any failure; extensions report their own failures with code 100 or
above. Arrays have fixed shapes/dtypes and retain their device identities.
For a stochastic plan, ``rng`` holds one uint32 Warp random-state word per
logical box. Kernels load a word, draw from it, and store the advanced word.
One thread per box owns that update; CPU/GPU bitwise parity is not implied.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L159-L185)

### particula.execution.process_extensions.GPUPreparationContext.arrays

*attribute* · *instance attribute*

```python
arrays: Mapping[str, object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L172-L172)

### particula.execution.process_extensions.GPUPreparationContext.clock

*attribute* · *instance attribute*

```python
clock: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L176-L176)

### particula.execution.process_extensions.GPUPreparationContext.device

*attribute* · *instance attribute*

```python
device: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L173-L173)

### particula.execution.process_extensions.GPUPreparationContext.failure

*attribute* · *instance attribute*

```python
failure: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L177-L177)

### particula.execution.process_extensions.GPUPreparationContext.logical_box_ids

*attribute* · *instance attribute*

```python
logical_box_ids: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L178-L178)

### particula.execution.process_extensions.GPUPreparationContext.rng

*attribute* · *class attribute* · *instance attribute*

```python
rng: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L180-L180)

### particula.execution.process_extensions.GPUPreparationContext.state

*attribute* · *instance attribute*

```python
state: Mapping[str, object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L179-L179)

### particula.execution.process_extensions.GPUPreparationContext.stream

*attribute* · *instance attribute*

```python
stream: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L174-L174)

### particula.execution.process_extensions.GPUPreparationContext.time_step

*attribute* · *instance attribute*

```python
time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L175-L175)

### particula.execution.process_extensions.GPUPreparationContext.__init__

*method*

```python
def __init__(arrays: Mapping[str, object], device: str, stream: object, time_step: float, clock: object, failure: object, logical_box_ids: tuple[str, ...], state: Mapping[str, object], rng: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L0-L0)

### particula.execution.process_extensions.GPUPreparationContext.__post_init__

*method*

```python
def __post_init__() -> None
```

Detach the map while retaining exact device array objects.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L182-L185)

## particula.execution.process_extensions.GPUProcessAdapter

*class*

```python
class GPUProcessAdapter(Protocol)
```

Bases: `Protocol`

Validate and freeze on CPU, then prepare retained resident operations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L188-L199)

### particula.execution.process_extensions.GPUProcessAdapter.plan

*method*

```python
def plan(process: object, aerosol: Aerosol) -> GPUProcessPlan
```

Validate requirements on detached CPU state without mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L191-L193)

### particula.execution.process_extensions.GPUProcessAdapter.prepare

*method*

```python
def prepare(plan: GPUProcessPlan, context: GPUPreparationContext) -> PreparedOperation
```

Allocate once and return an operation retaining its resources.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L195-L199)

## particula.execution.process_extensions.GPUProcessPlan

*class* · *dataclass*

```python
class GPUProcessPlan
```

Freeze one named implementation and its declared array requirements.

Parameters contain only finite Python scalars and nested tuples. Persistent
continuation belongs in prepared device state, never in these parameters.
Reads may include native metadata; writes must name evolving native or
registered fields. Instance names must be unique within one simulation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L81-L156)

### particula.execution.process_extensions.GPUProcessPlan.capabilities

*attribute* · *class attribute* · *instance attribute*

```python
capabilities: ProcessCapabilities = ProcessCapabilities(resident=True)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L97-L97)

### particula.execution.process_extensions.GPUProcessPlan.execution_mode

*attribute* · *class attribute* · *instance attribute*

```python
execution_mode: str = 'resident'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L99-L99)

### particula.execution.process_extensions.GPUProcessPlan.extension_id

*attribute* · *instance attribute*

```python
extension_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L92-L92)

### particula.execution.process_extensions.GPUProcessPlan.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L91-L91)

### particula.execution.process_extensions.GPUProcessPlan.parameters

*attribute* · *class attribute* · *instance attribute*

```python
parameters: tuple[tuple[str, object], ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L96-L96)

### particula.execution.process_extensions.GPUProcessPlan.reads

*attribute* · *class attribute* · *instance attribute*

```python
reads: tuple[str, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L94-L94)

### particula.execution.process_extensions.GPUProcessPlan.state

*attribute* · *class attribute* · *instance attribute*

```python
state: tuple[GPUStateSpec, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L98-L98)

### particula.execution.process_extensions.GPUProcessPlan.stochastic

*attribute* · *class attribute* · *instance attribute*

```python
stochastic: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L100-L100)

### particula.execution.process_extensions.GPUProcessPlan.version

*attribute* · *instance attribute*

```python
version: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L93-L93)

### particula.execution.process_extensions.GPUProcessPlan.writes

*attribute* · *class attribute* · *instance attribute*

```python
writes: tuple[str, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L95-L95)

### particula.execution.process_extensions.GPUProcessPlan.values

*property*

```python
values: Mapping[str, object]
```

Return an immutable setup-only parameter mapping.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L154-L156)

### particula.execution.process_extensions.GPUProcessPlan.__init__

*method*

```python
def __init__(name: str, extension_id: str, version: str, reads: tuple[str, ...] = (), writes: tuple[str, ...] = (), parameters: tuple[tuple[str, object], ...] = (), capabilities: ProcessCapabilities = ProcessCapabilities(resident=True), state: tuple[GPUStateSpec, ...] = (), execution_mode: str = 'resident', stochastic: bool = False) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L0-L0)

### particula.execution.process_extensions.GPUProcessPlan.__post_init__

*method*

```python
def __post_init__() -> None
```

Reject mutable configuration and ambiguous resource declarations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L102-L151)

## particula.execution.process_extensions.GPUStateSpec

*class* · *dataclass*

```python
class GPUStateSpec
```

Declare a fixed-shape persistent adapter array included in checkpoints.

Scratch allocated by an adapter is not continuation state. Counters, random
words and evolving parameters must use these declared arrays instead.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L35-L78)

### particula.execution.process_extensions.GPUStateSpec.dtype

*attribute* · *class attribute* · *instance attribute*

```python
dtype: str = 'float64'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L45-L45)

### particula.execution.process_extensions.GPUStateSpec.initial_value

*attribute* · *class attribute* · *instance attribute*

```python
initial_value: float | int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L46-L46)

### particula.execution.process_extensions.GPUStateSpec.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L43-L43)

### particula.execution.process_extensions.GPUStateSpec.shape

*attribute* · *instance attribute*

```python
shape: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L44-L44)

### particula.execution.process_extensions.GPUStateSpec.__init__

*method*

```python
def __init__(name: str, shape: tuple[int, ...], dtype: str = 'float64', initial_value: float | int = 0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L0-L0)

### particula.execution.process_extensions.GPUStateSpec.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate a portable primitive dtype and fixed nonempty shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L48-L78)

## particula.execution.process_extensions._RegisteredPlan

*class* · *dataclass*

```python
class _RegisteredPlan
```

Retain the implementation separately from frozen configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L202-L217)

### particula.execution.process_extensions._RegisteredPlan.adapter

*attribute* · *instance attribute*

```python
adapter: GPUProcessAdapter
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L207-L207)

### particula.execution.process_extensions._RegisteredPlan.configuration

*attribute* · *instance attribute*

```python
configuration: GPUProcessPlan
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L206-L206)

### particula.execution.process_extensions._RegisteredPlan.family

*property*

```python
family: str
```

Use a distinct scheduling key for each custom instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L210-L212)

### particula.execution.process_extensions._RegisteredPlan.values

*property*

```python
values: Mapping[str, object]
```

Expose immutable parameters to shared admission checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L215-L217)

### particula.execution.process_extensions._RegisteredPlan.__init__

*method*

```python
def __init__(configuration: GPUProcessPlan, adapter: GPUProcessAdapter) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L0-L0)

## particula.execution.process_extensions._immutable

*function*

```python
def _immutable(value: object) -> bool
```

Admit immutable scalar/tuple configuration without CPU arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/process_extensions.py#L26-L32)
