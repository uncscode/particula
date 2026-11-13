# particula.particles.exhaustion

*module*

Particle-capacity exhaustion planning and concrete apply helpers.

Policy resolution consumes fixed-shape capacity sidecars and creates an
immutable, later
Commit-boundary plan.  It validates every box before resolving any box, writes
no caller state, and can therefore be retried safely with corrected sidecars.
The exact requested/admitted and required-release counts, policy code, and
activation-prefix identity are diagnostics.  Release tuples are empty and
``float("nan")`` is the no-scale sentinel in policy resolution; ``-1`` is the
unused-index
sentinel.

Weighted inventory is calculated in float64 as ``sum(weights)``,
``sum(weights[..., None] * masses, axis=particle)``, and
``sum(weights * charge)`` (without materializing the weighted-mass product).
Intensive values divide these quantities by volume.  A later unscaled commit
must equal ``pre_state + source`` and a scaling commit must equal
``scale * pre_state + source``.  No radius-cubed or other moment is exact in
Policy resolution; any later preservation claim must specify its moment, domain,
and
tolerance before a commit can rely on it.

resampling adds a CPU-only deterministic equal-weight resampling reference. It
plans
detached fixed-capacity remaps without mutation, then commits one validated
all-box plan. Resampling neither provides GPU parity nor implements activation,
resizing, or compaction. Volume scaling is an opt-in, concrete-module-only
direct commit:
after all-box preflight, it scales selected rows' box volume, particle
concentration, and provisional demand by a resolved factor. It consumes
caller-owned sidecars, reports the resolved factor, and neither consumes a
policy resolution
plan nor resolves capacity policy, transfers data, resizes capacity, or adds a
runnable.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L1-L1)

## particula.particles.exhaustion.POLICY_ACTIVATE

*attribute* · *module attribute*

```python
POLICY_ACTIVATE = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L45-L45)

## particula.particles.exhaustion.POLICY_RESAMPLE_DEFERRED

*attribute* · *module attribute*

```python
POLICY_RESAMPLE_DEFERRED = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L46-L46)

## particula.particles.exhaustion.POLICY_SCALE_DEFERRED

*attribute* · *module attribute*

```python
POLICY_SCALE_DEFERRED = 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L47-L47)

## particula.particles.exhaustion.ExhaustionBoxPlan

*class* · *dataclass*

```python
class ExhaustionBoxPlan
```

Describe one immutable capacity decision at the later commit boundary.

**Attributes**

- `requested_count` (`int`) — Requested activation count for the box.
- `admitted_count` (`int`) — Count admitted by this plan; equals ``requested_count`` in policy resolution.
- `required_release_count` (`int`) — Additional slots needed after free capacity is exhausted.
- `releasable_count` (`int`) — Capacity reported as releasable by resampling.
- `policy_code` (`int`) — ``POLICY_ACTIVATE``, ``POLICY_RESAMPLE_DEFERRED``, or ``POLICY_SCALE_DEFERRED``.
- `activation_indices` (`tuple[int, ...]`) — Exact free-index prefix for activation plans; empty for deferred plans.
- `release_indices` (`tuple[int, ...]`) — Empty in policy resolution because release selection is deferred.
- `scale_factor` (`float`) — ``float("nan")`` in policy resolution because scaling is deferred.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L104-L132)

### particula.particles.exhaustion.ExhaustionBoxPlan.activation_indices

*attribute* · *instance attribute*

```python
activation_indices: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L130-L130)

### particula.particles.exhaustion.ExhaustionBoxPlan.admitted_count

*attribute* · *instance attribute*

```python
admitted_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L126-L126)

### particula.particles.exhaustion.ExhaustionBoxPlan.policy_code

*attribute* · *instance attribute*

```python
policy_code: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L129-L129)

### particula.particles.exhaustion.ExhaustionBoxPlan.releasable_count

*attribute* · *instance attribute*

```python
releasable_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L128-L128)

### particula.particles.exhaustion.ExhaustionBoxPlan.release_indices

*attribute* · *instance attribute*

```python
release_indices: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L131-L131)

### particula.particles.exhaustion.ExhaustionBoxPlan.requested_count

*attribute* · *instance attribute*

```python
requested_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L125-L125)

### particula.particles.exhaustion.ExhaustionBoxPlan.required_release_count

*attribute* · *instance attribute*

```python
required_release_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L127-L127)

### particula.particles.exhaustion.ExhaustionBoxPlan.scale_factor

*attribute* · *instance attribute*

```python
scale_factor: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L132-L132)

### particula.particles.exhaustion.ExhaustionBoxPlan.__init__

*method*

```python
def __init__(requested_count: int, admitted_count: int, required_release_count: int, releasable_count: int, policy_code: int, activation_indices: tuple[int, ...], release_indices: tuple[int, ...], scale_factor: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion.ExhaustionControls

*class* · *dataclass*

```python
class ExhaustionControls
```

Control which deferred policies may represent exhausted capacity.

**Attributes**

- `resampling` (`bool`) — Whether a sufficiently releasable box may use the deferred resampling policy.
- `representative_volume_scaling` (`bool`) — Whether a box not represented by resampling may use the deferred scaling policy.

**Raises**

- `TypeError` — If either control is not an exact Python ``bool``.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L50-L77)

### particula.particles.exhaustion.ExhaustionControls.representative_volume_scaling

*attribute* · *class attribute* · *instance attribute*

```python
representative_volume_scaling: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L65-L65)

### particula.particles.exhaustion.ExhaustionControls.resampling

*attribute* · *class attribute* · *instance attribute*

```python
resampling: bool = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L64-L64)

### particula.particles.exhaustion.ExhaustionControls.__init__

*method*

```python
def __init__(resampling: bool = True, representative_volume_scaling: bool = False) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

### particula.particles.exhaustion.ExhaustionControls.__post_init__

*method*

```python
def __post_init__() -> None
```

Reject bool-like values so policy controls remain unambiguous.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L67-L77)

## particula.particles.exhaustion.ExhaustionInputs

*class* · *dataclass*

```python
class ExhaustionInputs
```

Hold fixed-shape capacity sidecars for exhaustion planning.

**Attributes**

- `requested_count` (`NDArray[np.int32]`) — Requested activation count per box, as int32 values with shape ``(B,)``.
- `free_count` (`NDArray[np.int32]`) — Available free-slot count per box, as int32 values with shape ``(B,)``.
- `resampling_releasable_count` (`NDArray[np.int32]`) — Count per box that deferred resampling could release, as int32 values with shape ``(B,)``.
- `free_indices` (`NDArray[np.int32]`) — Ascending free-slot prefixes followed by ``-1`` unused sentinels, as int32 values with shape ``(B, N)``.

Validation occurs when :func:`resolve_exhaustion` consumes this record;
record construction intentionally performs no coercion or mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L80-L101)

### particula.particles.exhaustion.ExhaustionInputs.free_count

*attribute* · *instance attribute*

```python
free_count: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L99-L99)

### particula.particles.exhaustion.ExhaustionInputs.free_indices

*attribute* · *instance attribute*

```python
free_indices: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L101-L101)

### particula.particles.exhaustion.ExhaustionInputs.requested_count

*attribute* · *instance attribute*

```python
requested_count: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L98-L98)

### particula.particles.exhaustion.ExhaustionInputs.resampling_releasable_count

*attribute* · *instance attribute*

```python
resampling_releasable_count: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L100-L100)

### particula.particles.exhaustion.ExhaustionInputs.__init__

*method*

```python
def __init__(requested_count: NDArray[np.int32], free_count: NDArray[np.int32], resampling_releasable_count: NDArray[np.int32], free_indices: NDArray[np.int32]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion.ExhaustionPlan

*class* · *dataclass*

```python
class ExhaustionPlan
```

Hold immutable per-box decisions for one later commit boundary.

**Attributes**

- `box_plans` (`tuple[ExhaustionBoxPlan, ...]`) — One plan per validated input box, in input-box order.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L135-L143)

### particula.particles.exhaustion.ExhaustionPlan.box_plans

*attribute* · *instance attribute*

```python
box_plans: tuple[ExhaustionBoxPlan, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L143-L143)

### particula.particles.exhaustion.ExhaustionPlan.__init__

*method*

```python
def __init__(box_plans: tuple[ExhaustionBoxPlan, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion.WeightedInventory

*class* · *dataclass*

```python
class WeightedInventory
```

Hold tuple-backed float64 extensive and intensive inventories.

**Attributes**

- `number` (`tuple[np.float64, ...]`) — Weighted particle number, ``sum(weights)``, per box.
- `mass` (`tuple[tuple[np.float64, ...], ...]`) — Weighted mass, ``sum(weights * masses)``, per box and species.
- `charge` (`tuple[np.float64, ...]`) — Weighted charge, ``sum(weights * charge)``, per box.
- `number_per_volume` (`tuple[np.float64, ...]`) — Weighted number divided by box volume.
- `mass_per_volume` (`tuple[tuple[np.float64, ...], ...]`) — Weighted species mass divided by box volume.
- `charge_per_volume` (`tuple[np.float64, ...]`) — Weighted charge divided by box volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L146-L164)

### particula.particles.exhaustion.WeightedInventory.charge

*attribute* · *instance attribute*

```python
charge: tuple[np.float64, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L161-L161)

### particula.particles.exhaustion.WeightedInventory.charge_per_volume

*attribute* · *instance attribute*

```python
charge_per_volume: tuple[np.float64, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L164-L164)

### particula.particles.exhaustion.WeightedInventory.mass

*attribute* · *instance attribute*

```python
mass: tuple[tuple[np.float64, ...], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L160-L160)

### particula.particles.exhaustion.WeightedInventory.mass_per_volume

*attribute* · *instance attribute*

```python
mass_per_volume: tuple[tuple[np.float64, ...], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L163-L163)

### particula.particles.exhaustion.WeightedInventory.number

*attribute* · *instance attribute*

```python
number: tuple[np.float64, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L159-L159)

### particula.particles.exhaustion.WeightedInventory.number_per_volume

*attribute* · *instance attribute*

```python
number_per_volume: tuple[np.float64, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L162-L162)

### particula.particles.exhaustion.WeightedInventory.__init__

*method*

```python
def __init__(number: tuple[np.float64, ...], mass: tuple[tuple[np.float64, ...], ...], charge: tuple[np.float64, ...], number_per_volume: tuple[np.float64, ...], mass_per_volume: tuple[tuple[np.float64, ...], ...], charge_per_volume: tuple[np.float64, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion._CachedBoxState

*class* · *dataclass*

```python
class _CachedBoxState
```

Hold one preflighted box's active values for planning only.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L209-L217)

### particula.particles.exhaustion._CachedBoxState.active_indices

*attribute* · *instance attribute*

```python
active_indices: NDArray[np.intp]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L213-L213)

### particula.particles.exhaustion._CachedBoxState.charges

*attribute* · *instance attribute*

```python
charges: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L216-L216)

### particula.particles.exhaustion._CachedBoxState.concentrations

*attribute* · *instance attribute*

```python
concentrations: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L214-L214)

### particula.particles.exhaustion._CachedBoxState.masses

*attribute* · *instance attribute*

```python
masses: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L215-L215)

### particula.particles.exhaustion._CachedBoxState.radii

*attribute* · *instance attribute*

```python
radii: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L217-L217)

### particula.particles.exhaustion._CachedBoxState.__init__

*method*

```python
def __init__(active_indices: NDArray[np.intp], concentrations: NDArray[np.float64], masses: NDArray[np.float64], charges: NDArray[np.float64], radii: NDArray[np.float64]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion._ResamplingBounds

*class* · *dataclass*

```python
class _ResamplingBounds
```

Hold finite nonnegative diagnostic tolerances for resampling.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L199-L206)

### particula.particles.exhaustion._ResamplingBounds.diversity_absolute_error

*attribute* · *instance attribute*

```python
diversity_absolute_error: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L206-L206)

### particula.particles.exhaustion._ResamplingBounds.mean_radius_relative_error

*attribute* · *instance attribute*

```python
mean_radius_relative_error: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L204-L204)

### particula.particles.exhaustion._ResamplingBounds.radius_cubed_relative_error

*attribute* · *instance attribute*

```python
radius_cubed_relative_error: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L203-L203)

### particula.particles.exhaustion._ResamplingBounds.surface_relative_error

*attribute* · *instance attribute*

```python
surface_relative_error: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L205-L205)

### particula.particles.exhaustion._ResamplingBounds.__init__

*method*

```python
def __init__(radius_cubed_relative_error: float, mean_radius_relative_error: float, surface_relative_error: float, diversity_absolute_error: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion._ResamplingBoxPlan

*class* · *dataclass*

```python
class _ResamplingBoxPlan
```

Hold one detached fixed-capacity equal-weight remap.

resampling is CPU-only and deliberately fixed-capacity: retained slots
receive
replacement particles and released slots are cleared during the later,
all-box commit. It does not provide GPU parity, activation, or scaling.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L167-L184)

### particula.particles.exhaustion._ResamplingBoxPlan.released_indices

*attribute* · *instance attribute*

```python
released_indices: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L178-L178)

### particula.particles.exhaustion._ResamplingBoxPlan.replacement_charges

*attribute* · *instance attribute*

```python
replacement_charges: tuple[np.float64, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L181-L181)

### particula.particles.exhaustion._ResamplingBoxPlan.replacement_concentrations

*attribute* · *instance attribute*

```python
replacement_concentrations: tuple[np.float64, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L180-L180)

### particula.particles.exhaustion._ResamplingBoxPlan.replacement_masses

*attribute* · *instance attribute*

```python
replacement_masses: tuple[tuple[np.float64, ...], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L179-L179)

### particula.particles.exhaustion._ResamplingBoxPlan.retained_indices

*attribute* · *instance attribute*

```python
retained_indices: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L177-L177)

### particula.particles.exhaustion._ResamplingBoxPlan.source_charges

*attribute* · *class attribute* · *instance attribute*

```python
source_charges: tuple[np.float64, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L184-L184)

### particula.particles.exhaustion._ResamplingBoxPlan.source_concentrations

*attribute* · *class attribute* · *instance attribute*

```python
source_concentrations: tuple[np.float64, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L182-L182)

### particula.particles.exhaustion._ResamplingBoxPlan.source_masses

*attribute* · *class attribute* · *instance attribute*

```python
source_masses: tuple[tuple[np.float64, ...], ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L183-L183)

### particula.particles.exhaustion._ResamplingBoxPlan.__init__

*method*

```python
def __init__(retained_indices: tuple[int, ...], released_indices: tuple[int, ...], replacement_masses: tuple[tuple[np.float64, ...], ...], replacement_concentrations: tuple[np.float64, ...], replacement_charges: tuple[np.float64, ...], source_concentrations: tuple[np.float64, ...] = (), source_masses: tuple[tuple[np.float64, ...], ...] = (), source_charges: tuple[np.float64, ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion._ResamplingPlan

*class* · *dataclass*

```python
class _ResamplingPlan
```

Hold detached resampling remaps in particle-box order for one commit.

Empty per-box remaps preserve boxes that do not resample, including
zero-release resampling boxes. The tuple-backed records retain no views
into the policy resolution plan or particle storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L187-L196)

### particula.particles.exhaustion._ResamplingPlan.box_plans

*attribute* · *instance attribute*

```python
box_plans: tuple[_ResamplingBoxPlan, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L196-L196)

### particula.particles.exhaustion._ResamplingPlan.__init__

*method*

```python
def __init__(box_plans: tuple[_ResamplingBoxPlan, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L0-L0)

## particula.particles.exhaustion._build_box_remap

*function*

```python
def _build_box_remap(source: _CachedBoxState, required_release_count: int, density: NDArray[np.float64], bounds: _ResamplingBounds) -> _ResamplingBoxPlan
```

Build a stable, linear-sweep equal-weight remap from cached state.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L821-L964)

## particula.particles.exhaustion._cache_particle_state

*function*

```python
def _cache_particle_state(particles: ParticleData) -> tuple[_CachedBoxState, ...]
```

Fully validate particle state once and cache active values by box.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L584-L636)

## particula.particles.exhaustion._relative_error

*function*

```python
def _relative_error(before: np.float64, after: np.float64) -> float
```

Calculate a finite relative diagnostic error, including zero values.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L733-L737)

## particula.particles.exhaustion._riemer_diversity

*function*

```python
def _riemer_diversity(masses: NDArray[np.float64], concentrations: NDArray[np.float64]) -> np.float64
```

Calculate particle mixing-state Riemer diversity using natural logs.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L740-L772)

## particula.particles.exhaustion._validate_capacity_counts

*function*

```python
def _validate_capacity_counts(inputs: ExhaustionInputs, particle_count: int, *, allow_oversized_requests: bool = False) -> None
```

Validate count ranges against fixed particle capacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L255-L278)

## particula.particles.exhaustion._validate_exact_float

*function*

```python
def _validate_exact_float(name: str, value: object) -> float
```

Return an exact finite nonnegative Python float diagnostic bound.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L500-L506)

## particula.particles.exhaustion._validate_free_indices

*function*

```python
def _validate_free_indices(inputs: ExhaustionInputs, particle_count: int) -> None
```

Validate free-index prefixes and unused suffix sentinels per box.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L281-L297)

## particula.particles.exhaustion._validate_p1_plan

*function*

```python
def _validate_p1_plan(exhaustion_plan: object, box_count: int, particle_count: int) -> ExhaustionPlan
```

Validate policy resolution records without resolving policy or retaining
their data.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L639-L730)

## particula.particles.exhaustion._validate_p4_non_aliasing

*function*

```python
def _validate_p4_non_aliasing(particles: ParticleData, sidecars: tuple[NDArray[Any], ...]) -> None
```

Reject volume scaling sidecar and writable particle-storage overlap
before writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L1237-L1254)

## particula.particles.exhaustion._validate_p4_sidecar

*function*

```python
def _validate_p4_sidecar(name: str, values: object, dtype: np.dtype[np.generic], box_count: int) -> NDArray[Any]
```

Validate one writable contiguous volume scaling per-box sidecar.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L1215-L1234)

## particula.particles.exhaustion._validate_particle_schema

*function*

```python
def _validate_particle_schema(particles: object) -> ParticleData
```

Validate resampling's exact writable CPU particle storage schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L534-L581)

## particula.particles.exhaustion._validate_remap_diagnostics

*function*

```python
def _validate_remap_diagnostics(source: _CachedBoxState, replacement_masses: NDArray[np.float64], replacement_concentrations: NDArray[np.float64], replacement_charges: NDArray[np.float64], bounds: _ResamplingBounds | None) -> None
```

Check exact inventories and bounded scalar resampling diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L775-L818)

## particula.particles.exhaustion._validate_resampling_bounds

*function*

```python
def _validate_resampling_bounds(radius_cubed_relative_error: object, mean_radius_relative_error: object, surface_relative_error: object, diversity_absolute_error: object) -> _ResamplingBounds
```

Validate and detach resampling diagnostic bounds before particle
inspection.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L509-L531)

## particula.particles.exhaustion._validate_resampling_plan

*function*

```python
def _validate_resampling_plan(plan: object, particles: ParticleData) -> _ResamplingPlan
```

Validate detached remaps before the apply boundary writes anything.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L1038-L1132)

## particula.particles.exhaustion._validate_sidecar_schema

*function*

```python
def _validate_sidecar_schema(inputs: ExhaustionInputs) -> None
```

Validate sidecar array types, dtypes, and ranks without coercion.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L220-L238)

## particula.particles.exhaustion._validate_sidecar_shapes

*function*

```python
def _validate_sidecar_shapes(inputs: ExhaustionInputs) -> tuple[int, int]
```

Validate cross-box sidecar shapes and return box and particle counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L241-L252)

## particula.particles.exhaustion._validate_sidecars

*function*

```python
def _validate_sidecars(inputs: ExhaustionInputs, *, allow_oversized_requests: bool = False) -> tuple[int, int]
```

Validate all sidecars without coercion, sorting, copying, or mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L300-L314)

## particula.particles.exhaustion.apply_representative_volume_scaling

*function*

```python
def apply_representative_volume_scaling(particles: ParticleData, provisional_source_demand: NDArray[np.float64], scaling_required: NDArray[np.bool_], requested_scale: NDArray[np.float64], minimum_scale: NDArray[np.float64], minimum_volume: NDArray[np.float64], resolved_scale: NDArray[np.float64]) -> tuple[ParticleData, NDArray[np.float64], NDArray[np.float64]]
```

Apply volume scaling to selected CPU particle-data rows in place.

A row is selected exactly when ``scaling_required`` is true and its
provisional demand is positive. Each selected row scales its box volume,
every particle concentration, and provisional demand by its requested
factor. Masses, charge, density, all configuration sidecars, and
unselected rows remain unchanged. Complete all-box preflight precedes the
diagnostic write and commit, so validation rejection leaves every
caller-owned array unchanged. ``resolved_scale`` receives the requested
factor for selected rows and ``1.0`` otherwise.

**Parameters**

- `particles` (`ParticleData`) — Writable fixed-shape CPU particle data to update.
- `provisional_source_demand` (`NDArray[np.float64]`) — Writable contiguous ``float64`` demand per box, shaped ``(B,)``.
- `scaling_required` (`NDArray[np.bool_]`) — Writable contiguous ``bool`` selection flags per box, shaped ``(B,)``.
- `requested_scale` (`NDArray[np.float64]`) — Writable contiguous ``float64`` requested factors per box, shaped ``(B,)``.
- `minimum_scale` (`NDArray[np.float64]`) — Writable contiguous ``float64`` lower factor bounds per box, shaped ``(B,)``.
- `minimum_volume` (`NDArray[np.float64]`) — Writable contiguous ``float64`` post-scaling box-volume bounds in m³, shaped ``(B,)``.
- `resolved_scale` (`NDArray[np.float64]`) — Writable contiguous ``float64`` diagnostic output per box, shaped ``(B,)``.

**Returns**

- (`ParticleData`) — The identical particle container, demand sidecar, and resolved-scale
- (`NDArray[np.float64]`) — sidecar after a successful in-place diagnostic write and selected-row
- (`NDArray[np.float64]`) — commit.

**Raises**

- `TypeError` — If particle storage or a sidecar has an unsupported type or
dtype.
- `ValueError` — If schemas, physical values, factor bounds, or selected
scaled-volume bounds are invalid. Rejection is atomic.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L1257-L1390)

## particula.particles.exhaustion.apply_resampling

*function*

```python
def apply_resampling(particles: ParticleData, plan: _ResamplingPlan) -> ParticleData
```

Atomically apply a previously planned CPU fixed-capacity remap.

Every plan and target-state validation runs before the first assignment, so
ordinary validation errors cannot partially mutate an earlier box. This
commit performs no policy resolution, selection, activation, scaling,
resizing, or compaction, and has no GPU-parity path.

**Parameters**

- `particles` (`ParticleData`) — Writable fixed-shape CPU particle storage to update in place.
- `plan` (`_ResamplingPlan`) — Detached plan returned by :func:`plan_resampling`.

**Returns**

- (`ParticleData`) — The identical ``particles`` container after the commit.

**Raises**

- `TypeError` — If the target storage or detached plan has an unsupported
type or dtype.
- `ValueError` — If the target state or plan is invalid, or the plan no
longer exactly covers the active slots in a resampling box.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L1135-L1212)

## particula.particles.exhaustion.get_weighted_inventory

*function*

```python
def get_weighted_inventory(masses: object, weights: object, charge: object, volume: object) -> WeightedInventory
```

Calculate float64 weighted inventories without retaining caller arrays.

Array-like inputs are converted to float64. The mass reduction avoids a
full weighted-mass broadcast product. Returned tuples do not retain caller
array storage.

**Parameters**

- `masses` (`object`) — Array-like particle masses with shape ``(B, N, S)``.
- `weights` (`object`) — Array-like particle weights with shape ``(B, N)``.
- `charge` (`object`) — Array-like particle charge with shape ``(B, N)``.
- `volume` (`object`) — Array-like box volumes with shape ``(B,)``.

**Returns**

- (`WeightedInventory`) — Tuple-backed extensive and intensive float64 inventories.

**Raises**

- `ValueError` — If converted inputs have invalid shape or physical values.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L402-L497)

## particula.particles.exhaustion.plan_resampling

*function*

```python
def plan_resampling(particles: ParticleData, exhaustion_plan: ExhaustionPlan, *, radius_cubed_relative_error: float = 1.0, mean_radius_relative_error: float = 1.0, surface_relative_error: float = 1.0, diversity_absolute_error: float = 1.0) -> _ResamplingPlan
```

Create an immutable, deterministic CPU fixed-capacity resampling plan.

The function validates all boxes and only plans; it does not mutate policy
resolution
records or particle state. The detached plan retains existing active
capacity only, so it neither activates, scales, resizes, nor compacts
particles, and it provides no GPU-parity path. Pass it to
:func:`apply_resampling` for the all-box commit boundary.

**Parameters**

- `particles` (`ParticleData`) — Writable, fixed-shape CPU particle data.
- `exhaustion_plan` (`ExhaustionPlan`) — Immutable policy resolution capacity decisions to consume unchanged.
- `radius_cubed_relative_error` (`float`) (default: `1.0`) — Exact Python float, finite nonnegative bound for the weighted radius-cubed diagnostic.
- `mean_radius_relative_error` (`float`) (default: `1.0`) — Exact Python float, finite nonnegative bound for the number-weighted-radius diagnostic.
- `surface_relative_error` (`float`) (default: `1.0`) — Exact Python float, finite nonnegative bound for the weighted-surface-area diagnostic.
- `diversity_absolute_error` (`float`) (default: `1.0`) — Exact Python float, finite nonnegative bound for the Riemer bulk-diversity diagnostic.

**Returns**

- (`_ResamplingPlan`) — A detached immutable plan, one remap record per particle box.

**Raises**

- `TypeError` — If particle storage, policy resolution records, or diagnostic
bounds have
an unsupported type or dtype.
- `ValueError` — If the fixed-capacity state or policy resolution decisions
are invalid,
resampling is infeasible, or a diagnostic exceeds its bound.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L967-L1035)

## particula.particles.exhaustion.resolve_exhaustion

*function*

```python
def resolve_exhaustion(inputs: ExhaustionInputs, controls: ExhaustionControls, *, allow_oversized_requests: bool = False) -> ExhaustionPlan
```

Resolve immutable capacity decisions after complete sidecar validation.

The resolver validates every input box before creating a returned plan.
Activation uses the exact requested free-index prefix. Exhausted capacity
uses sufficiently releasable resampling before enabled scaling; both remain
deferred and therefore have empty release indices and a ``nan`` scale.

**Parameters**

- `inputs` (`ExhaustionInputs`) — Fixed-shape, int32 capacity sidecars.
- `controls` (`ExhaustionControls`) — Strict policy enablement controls.
- `allow_oversized_requests` (`bool`) (default: `False`) — Permit provisional requests above physical capacity so a later scaling policy can reduce them. The default preserves strict fixed-capacity validation.

**Returns**

- (`ExhaustionPlan`) — A tuple-backed plan with activation-prefix identity or deferred policy.

**Raises**

- `TypeError` — If either public record has the wrong type.
- `ValueError` — If sidecars are invalid or no enabled policy can represent
exhausted capacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/exhaustion.py#L317-L399)
