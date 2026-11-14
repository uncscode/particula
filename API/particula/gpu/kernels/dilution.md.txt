# particula.gpu.kernels.dilution

*module*

Apply fixed-shape direct GPU dilution to caller-owned concentrations.

The dilution coefficient is ``alpha = Q / V`` [s^-1], where ``Q`` is flow
rate and ``V`` is volume. This finite-step execution module applies
``c_new = c * exp(-alpha * time_step)`` in place to particle and gas
concentrations and returns the identical containers.

Before allocating scalar/factor storage, launching an update kernel, or
mutating concentrations, the entry point validates the coefficient, time step,
particle-mass schema, and particle and gas concentration schemas and values.
Device-resident validation scans may allocate or launch, but rejected calls do
not launch update kernels or mutate caller-owned state.
Valid same-device per-box coefficients retain caller ownership and identity;
scalar coefficients are broadcast into private active-device ``wp.float64``
storage only after preflight. Zero scalar coefficients and zero time steps are
write-free no-ops only after full preflight. Rollback after a successfully
launched kernel failure is not promised.

Prepared private execution preserves this validation and scalar-allocation
ordering; retained-reference enqueue performs no validation, allocation,
readback, transfer, synchronization, lookup, rebinding, or recovery. Rejection
is pre-writer; rollback is not promised after an update writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L1-L1)

## particula.gpu.kernels.dilution._PreparedDilutionCall

*class* · *dataclass*

```python
class _PreparedDilutionCall
```

Privately freeze one validated dilution launch and its references.

Enqueue uses only these pinned field references; replacing a container
attribute after preparation therefore cannot redirect a writer. Validation,
normalization, and private allocation are complete before enqueue, which
performs no readback, transfer, synchronization, lookup, rebinding, or
recovery. The no-op path is write-free; rollback is not promised after a
writer launches.

**Attributes**

- `particles` (`Any`) — Caller-owned particle container returned by the operation.
- `gas` (`Any`) — Caller-owned gas container returned by the operation.
- `coefficient` (`Any | None`) — Frozen per-box coefficient array, or ``None`` for a no-op.
- `time_step` (`float`) — Validated duration in seconds.
- `particle_concentration` (`Any`) — Frozen particle concentration target.
- `gas_concentration` (`Any`) — Frozen gas concentration target.
- `factors` (`Any | None`) — Private factor array, or ``None`` for a no-op.
- `n_boxes` (`int`) — Number of simulation boxes.
- `n_particles` (`int`) — Number of particle slots per box.
- `n_gas_species` (`int`) — Number of gas species per box.
- `device` (`Any`) — Active Warp device for the launches.
- `no_op` (`bool`) — Whether validation completed for a write-free call.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L402-L440)

### particula.gpu.kernels.dilution._PreparedDilutionCall.coefficient

*attribute* · *instance attribute*

```python
coefficient: Any | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L430-L430)

### particula.gpu.kernels.dilution._PreparedDilutionCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L438-L438)

### particula.gpu.kernels.dilution._PreparedDilutionCall.factors

*attribute* · *instance attribute*

```python
factors: Any | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L434-L434)

### particula.gpu.kernels.dilution._PreparedDilutionCall.failure

*attribute* · *class attribute* · *instance attribute*

```python
failure: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L440-L440)

### particula.gpu.kernels.dilution._PreparedDilutionCall.gas

*attribute* · *instance attribute*

```python
gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L429-L429)

### particula.gpu.kernels.dilution._PreparedDilutionCall.gas_concentration

*attribute* · *instance attribute*

```python
gas_concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L433-L433)

### particula.gpu.kernels.dilution._PreparedDilutionCall.n_boxes

*attribute* · *instance attribute*

```python
n_boxes: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L435-L435)

### particula.gpu.kernels.dilution._PreparedDilutionCall.n_gas_species

*attribute* · *instance attribute*

```python
n_gas_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L437-L437)

### particula.gpu.kernels.dilution._PreparedDilutionCall.n_particles

*attribute* · *instance attribute*

```python
n_particles: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L436-L436)

### particula.gpu.kernels.dilution._PreparedDilutionCall.no_op

*attribute* · *instance attribute*

```python
no_op: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L439-L439)

### particula.gpu.kernels.dilution._PreparedDilutionCall.particle_concentration

*attribute* · *instance attribute*

```python
particle_concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L432-L432)

### particula.gpu.kernels.dilution._PreparedDilutionCall.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L428-L428)

### particula.gpu.kernels.dilution._PreparedDilutionCall.time_step

*attribute* · *instance attribute*

```python
time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L431-L431)

### particula.gpu.kernels.dilution._PreparedDilutionCall.__init__

*method*

```python
def __init__(particles: Any, gas: Any, coefficient: Any | None, time_step: float, particle_concentration: Any, gas_concentration: Any, factors: Any | None, n_boxes: int, n_particles: int, n_gas_species: int, device: Any, no_op: bool, failure: Any = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L0-L0)

## particula.gpu.kernels.dilution._apply_gas_dilution

*function*

```python
def _apply_gas_dilution(concentration: wp.array2d(dtype=wp.float64), coefficient: wp.array(dtype=wp.float64), factors: wp.array(dtype=wp.float64)) -> None
```

Run the shared device operation through the direct kernel API.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L137-L146)

## particula.gpu.kernels.dilution._apply_gas_dilution_body

*function*

```python
def _apply_gas_dilution_body(concentration: wp.array2d(dtype=wp.float64), coefficient: wp.array(dtype=wp.float64), factors: wp.array(dtype=wp.float64), box: wp.int32, species: wp.int32) -> None
```

Apply precomputed factors to gas concentration in place.

**Parameters**

- `concentration` (`wp.array2d(dtype=wp.float64)`) — Per-box, per-species concentration array.
- `coefficient` (`wp.array(dtype=wp.float64)`) — Per-box coefficients used to skip zero-coefficient boxes.
- `factors` (`wp.array(dtype=wp.float64)`) — Precomputed per-box exponential dilution factors.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L117-L134)

## particula.gpu.kernels.dilution._apply_gas_dilution_guarded

*function*

```python
def _apply_gas_dilution_guarded(concentration: wp.array2d(dtype=wp.float64), coefficient: wp.array(dtype=wp.float64), factors: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L149-L161)

## particula.gpu.kernels.dilution._apply_particle_dilution

*function*

```python
def _apply_particle_dilution(concentration: wp.array2d(dtype=wp.float64), coefficient: wp.array(dtype=wp.float64), factors: wp.array(dtype=wp.float64)) -> None
```

Run the shared device operation through the direct kernel API.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L86-L97)

## particula.gpu.kernels.dilution._apply_particle_dilution_body

*function*

```python
def _apply_particle_dilution_body(concentration: wp.array2d(dtype=wp.float64), coefficient: wp.array(dtype=wp.float64), factors: wp.array(dtype=wp.float64), box: wp.int32, particle: wp.int32) -> None
```

Apply precomputed factors to particle concentration in place.

**Parameters**

- `concentration` (`wp.array2d(dtype=wp.float64)`) — Per-box, per-particle concentration array.
- `coefficient` (`wp.array(dtype=wp.float64)`) — Per-box coefficients used to skip zero-coefficient boxes.
- `factors` (`wp.array(dtype=wp.float64)`) — Precomputed per-box exponential dilution factors.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L64-L83)

## particula.gpu.kernels.dilution._apply_particle_dilution_guarded

*function*

```python
def _apply_particle_dilution_guarded(concentration: wp.array2d(dtype=wp.float64), coefficient: wp.array(dtype=wp.float64), factors: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L100-L114)

## particula.gpu.kernels.dilution._coerce_nonnegative_real

*function*

```python
def _coerce_nonnegative_real(value: Any, name: str) -> float
```

Return one supported finite, nonnegative real scalar.

**Parameters**

- `value` (`Any`) — Candidate Python or NumPy scalar.
- `name` (`str`) — Input label used in stable validation messages.

**Returns**

- (`float`) — The scalar as a Python float.

**Raises**

- `TypeError` — If ``value`` is not a supported real scalar form.
- `ValueError` — If ``value`` is non-finite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L186-L214)

## particula.gpu.kernels.dilution._dilution_factors

*function*

```python
def _dilution_factors(coefficient: wp.array(dtype=wp.float64), time_step: wp.float64, factors: wp.array(dtype=wp.float64)) -> None
```

Calculate one exponential dilution factor for each simulation box.

**Parameters**

- `coefficient` (`wp.array(dtype=wp.float64)`) — Per-box dilution coefficients in inverse seconds.
- `time_step` (`wp.float64`) — Duration of the dilution step in seconds.
- `factors` (`wp.array(dtype=wp.float64)`) — Output array receiving ``exp(-coefficient × time_step)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L47-L61)

## particula.gpu.kernels.dilution._enqueue_prepared_dilution_call

*function*

```python
def _enqueue_prepared_dilution_call(prepared: _PreparedDilutionCall) -> tuple[Any, Any]
```

Issue only the frozen dilution launches for one prepared call.

This private enqueue boundary performs no validation, allocation,
normalization, readback, transfer, synchronization, lookup, rebinding, or
recovery. Its no-op path is write-free; rollback is not promised after a
writer launches.

**Parameters**

- `prepared` (`_PreparedDilutionCall`) — Validated dilution record produced by ``_prepare_dilution_step_gpu``.

**Returns**

- (`tuple[Any, Any]`) — The pinned particle and gas containers, unchanged in identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L632-L689)

## particula.gpu.kernels.dilution._get_required_field

*function*

```python
def _get_required_field(container: Any, field: str, name: str) -> Any
```

Return a required container field with a stable schema failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L283-L288)

## particula.gpu.kernels.dilution._normalize_coefficient

*function*

```python
def _normalize_coefficient(coefficient: float | Any, n_boxes: int, device: Any) -> Any
```

Normalize an already validated coefficient into a Warp array.

A valid per-box Warp array is returned by identity. Scalar inputs allocate
private ``wp.float64`` storage of shape ``(n_boxes,)`` without caching.

**Parameters**

- `coefficient` (`float | Any`) — Real scalar [s^-1] or per-box Warp coefficient array.
- `n_boxes` (`int`) — Number of simulation boxes.
- `device` (`Any`) — Active particle-mass device.

**Returns**

- (`Any`) — A private scalar broadcast or the original valid Warp array.

> **Notes**
>
> This helper assumes the public entry point has completed all input
> validation. It performs allocation only for valid scalar inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L233-L264)

## particula.gpu.kernels.dilution._normalize_time_step

*function*

```python
def _normalize_time_step(time_step: float | Any) -> float
```

Validate and return a finite, nonnegative time step in seconds.

**Parameters**

- `time_step` (`float | Any`) — Candidate Python or NumPy scalar duration [s].

**Returns**

- (`float`) — The duration as a Python float.

**Raises**

- `TypeError` — If ``time_step`` is not a supported real scalar.
- `ValueError` — If ``time_step`` is non-finite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L267-L280)

## particula.gpu.kernels.dilution._prepare_dilution_step_gpu

*function*

```python
def _prepare_dilution_step_gpu(particles: Any, gas: Any, coefficient: float | Any, time_step: float, *, factors: Any | None = None) -> _PreparedDilutionCall
```

Privately prepare GPU dilution after complete entry-point preflight.

Arguments are positional in the order ``particles``, ``gas``,
``coefficient``, and ``time_step``. The coefficient ``alpha = Q / V`` has
SI units [s^-1], and ``time_step`` is in seconds. The matching private
enqueue helper applies ``c_new = c * exp(-alpha * time_step)`` in place.

Finite-step execution retains coefficient validation's coefficient-input
contract: a finite, nonnegative
Python/NumPy real scalar or a caller-owned, active-device ``wp.float64``
array shaped ``(n_boxes,)``. Valid per-box arrays are retained by identity
and their values are read only during preflight. Masses must be a
same-device ``wp.float64`` rank-3 array. Particle and gas concentrations
must be same-device ``wp.float64`` rank-2 arrays. Particle concentration
width matches the particle dimension; gas concentration has an independent
width but must share the mass-derived box count. Coefficients and
concentrations must be finite and nonnegative.
Only the two concentration arrays are mutated; masses and all other
caller-owned fields are preserved.

After explicit population admission, validation order is coefficient form,
time step, particle-mass schema,
per-box coefficient schema and values, particle concentration schema and
values, then gas concentration schema and values. Array values are scanned
on their active device, with only a scalar validation result observed by the
host; preflight never materializes a caller-owned array on the CPU.
validation scans may allocate or launch, but rejected calls perform no
update-kernel launch or caller mutation. A zero scalar coefficient or zero
time step returns only after full preflight, without private scalar/factor
allocation or an update launch. This atomicity guarantee applies only
before an update launch; rollback after a launched kernel failure is not
promised.

**Parameters**

- `particles` (`Any`) — Particle container with fixed-shape concentration storage.
- `gas` (`Any`) — Gas container with fixed-shape concentration storage.
- `coefficient` (`float | Any`) — Dilution coefficient [s^-1], scalar or per-box Warp array.
- `time_step` (`float`) — Finite, nonnegative duration [s].
- `factors` (`Any | None`) (default: `None`) — Optional caller-owned prepared factor storage. This private argument is supplied only by a validated resident resource view; direct calls omit it and retain fallback allocation behavior.

**Returns**

- (`_PreparedDilutionCall`) — A private frozen call record for the matching enqueue helper.

**Raises**

- `TypeError` — If a scalar coefficient or ``time_step`` is not a supported
real scalar.
- `ValueError` — If a schema, shape, device, or physical domain is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L443-L629)

## particula.gpu.kernels.dilution._scan_nonnegative_finite_1d

*function*

```python
def _scan_nonnegative_finite_1d(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record whether a one-dimensional array has an invalid value.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L164-L172)

## particula.gpu.kernels.dilution._scan_nonnegative_finite_2d

*function*

```python
def _scan_nonnegative_finite_2d(values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record whether a two-dimensional array has an invalid value.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L175-L183)

## particula.gpu.kernels.dilution._validate_concentration_schema

*function*

```python
def _validate_concentration_schema(values: Any, name: str, n_boxes: int, device: Any, expected_width: int | None = None) -> None
```

Validate concentration schema and physical values read-only.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L340-L360)

## particula.gpu.kernels.dilution._validate_mass_schema

*function*

```python
def _validate_mass_schema(particles: Any) -> tuple[Any, int, int, int, Any]
```

Validate masses before deriving authoritative dimensions and device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L291-L301)

## particula.gpu.kernels.dilution._validate_no_overlap

*function*

```python
def _validate_no_overlap(read_side_arrays: tuple[Any, ...], mutable_arrays: tuple[Any, ...]) -> None
```

Reject caller storage that aliases a concentration update target.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L384-L399)

## particula.gpu.kernels.dilution._validate_nonnegative_finite_values

*function*

```python
def _validate_nonnegative_finite_values(values: Any, name: str) -> None
```

Reject non-finite or negative values with a device-resident scan.

The single integer result is the only validation state copied to the host;
caller-owned arrays are never materialized on the host during preflight.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L304-L323)

## particula.gpu.kernels.dilution._validate_per_box_coefficient

*function*

```python
def _validate_per_box_coefficient(coefficient: Any, n_boxes: int, device: Any) -> Any
```

Validate exact same-device coefficient schema and physical values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L326-L337)

## particula.gpu.kernels.dilution._validate_warp_coefficient_form

*function*

```python
def _validate_warp_coefficient_form(coefficient: Any) -> Any
```

Validate Warp coefficient metadata independent of particle metadata.

Shape and device compatibility require particle metadata and are checked by
``_normalize_coefficient`` after time-step validation. Dtype and rank are
checked first so malformed coefficient inputs fail before time or container
access.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L217-L230)

## particula.gpu.kernels.dilution._warp_array_memory_range

*function*

```python
def _warp_array_memory_range(array: Any) -> tuple[int, int]
```

Return a contiguous Warp array's byte range for ownership checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L363-L381)

## particula.gpu.kernels.dilution.dilution_step_gpu

*function*

```python
def dilution_step_gpu(particles: Any, gas: Any, coefficient: float | Any, time_step: float) -> tuple[Any, Any]
```

Prepare and enqueue GPU dilution after complete entry-point preflight.

The wrapper preserves the direct API while the private prepared record
isolates validation and allocation from device enqueue.

**Parameters**

- `particles` (`Any`) — Caller-owned particle container with Warp concentration storage.
- `gas` (`Any`) — Caller-owned gas container with Warp concentration storage.
- `coefficient` (`float | Any`) — Nonnegative dilution coefficient in ``s^-1``, either a scalar or a per-box ``wp.float64`` array.
- `time_step` (`float`) — Nonnegative dilution duration in seconds.

**Returns**

- (`tuple[Any, Any]`) — The original ``particles`` and ``gas`` containers, in that order.

**Raises**

- `TypeError` — If a scalar input has an unsupported type.
- `ValueError` — If schemas, devices, values, or storage overlap are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/dilution.py#L692-L720)
