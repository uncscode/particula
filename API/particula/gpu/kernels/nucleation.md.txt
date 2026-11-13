# particula.gpu.kernels.nucleation

*module*

Define direct-Warp GPU nucleation planning seams and one public step.

This unexported module validates fixed-capacity particle, gas, environment,
and caller-owned sidecar schemas for a direct-Warp nucleation path. Input
validation is
read-only. Private demand planning calculates source demand [#/m³], admits one
shared
inventory-safe demand per box, and commits only its designated sidecars. Demand
planning
does not select or activate slots, resolve exhaustion, resize storage, mutate
particle fields or ``gas.concentration``, transfer data to the host, or use a
CPU physics fallback. Private capacity staging converts admitted demand times
particle volume
to representable ``int32`` provisional event counts, reuses fixed-slot
diagnostics, and records only the deterministic free-slot prefix that is
currently selectable. It retains counts beyond free capacity and performs no
activation or particle/gas transaction. Private exhaustion resolution consumes
immutable demand planning/capacity staging
handoffs, applies resampling before optional representative-volume scaling, and
writes only its caller-owned finalized diagnostics. The public
``nucleation_step_gpu`` then validates exhaustion resolution's handoff and
performs one fused
particle/gas commit.

The preflight accepts only same-device, contiguous Warp arrays with fixed
shapes. Frozen records prevent rebinding their fields but do not copy, freeze,
or otherwise transfer their caller-owned arrays. Only ``nucleation_step_gpu``
is lazily exported through ``particula.gpu.kernels``; all records and helpers
remain concrete-module-only.

Prepared private execution retains phase validation ordering, normalized
controls, and established workspace allocation. Retained-reference enqueue
performs no host metadata validation, readback, allocation, transfer,
synchronization,
lookup, rebinding, or policy resolution; recovery is not promised after a
writer-capable launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L1-L1)

## particula.gpu.kernels.nucleation._AVOGADRO_NUMBER

*attribute* · *module attribute*

```python
_AVOGADRO_NUMBER = wp.constant(wp.float64(AVOGADRO_NUMBER))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L81-L81)

## particula.gpu.kernels.nucleation._MIN_NORMAL_FLOAT64

*attribute* · *module attribute*

```python
_MIN_NORMAL_FLOAT64 = wp.constant(wp.float64(2.2250738585072014e-308))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L82-L82)

## particula.gpu.kernels.nucleation._MIN_SUBNORMAL_FLOAT64

*attribute* · *module attribute*

```python
_MIN_SUBNORMAL_FLOAT64 = wp.constant(wp.float64(5e-324))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L83-L83)

## particula.gpu.kernels.nucleation._P2_GATE_ELIGIBLE

*attribute* · *module attribute*

```python
_P2_GATE_ELIGIBLE = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L85-L85)

## particula.gpu.kernels.nucleation._P2_GATE_GAS_LIMITED_OFFSET

*attribute* · *module attribute*

```python
_P2_GATE_GAS_LIMITED_OFFSET = 7
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L92-L92)

## particula.gpu.kernels.nucleation._P2_GATE_LOW_SATURATION

*attribute* · *module attribute*

```python
_P2_GATE_LOW_SATURATION = 5
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L90-L90)

## particula.gpu.kernels.nucleation._P2_GATE_ZERO_COEFFICIENT

*attribute* · *module attribute*

```python
_P2_GATE_ZERO_COEFFICIENT = 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L87-L87)

## particula.gpu.kernels.nucleation._P2_GATE_ZERO_INVENTORY

*attribute* · *module attribute*

```python
_P2_GATE_ZERO_INVENTORY = 6
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L91-L91)

## particula.gpu.kernels.nucleation._P2_GATE_ZERO_PRECURSOR

*attribute* · *module attribute*

```python
_P2_GATE_ZERO_PRECURSOR = 4
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L89-L89)

## particula.gpu.kernels.nucleation._P2_GATE_ZERO_SURVIVAL

*attribute* · *module attribute*

```python
_P2_GATE_ZERO_SURVIVAL = 3
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L88-L88)

## particula.gpu.kernels.nucleation._P2_GATE_ZERO_TIME

*attribute* · *module attribute*

```python
_P2_GATE_ZERO_TIME = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L86-L86)

## particula.gpu.kernels.nucleation.NucleationConfig

*class* · *dataclass*

```python
class NucleationConfig
```

Store immutable scalar controls for a potential nucleation calculation.

The activation law is ``J = survival_factor * coefficient * C`` and the
kinetic law is ``J = survival_factor * coefficient * C²``, where ``J`` is
formation rate [#/m³/s] and ``C`` is precursor number concentration
[#/m³]. Therefore, activation ``coefficient`` has units [1/s] and kinetic
``coefficient`` has units [m³/s]. This record performs scalar consistency
checks only; array-dependent species and physical-state validation belongs
to :func:`_preflight_nucleation`.

**Attributes**

- `rate_law` (`str`) — Either ``"activation"`` or ``"kinetic"``.
- `coefficient` (`float`) — Nonnegative rate-law coefficient, in [1/s] or [m³/s].
- `survival_factor` (`float`) — Nonnegative, dimensionless formation survival factor.
- `precursor_index` (`int`) — Zero-based precursor species index.
- `molecule_counts` (`tuple[int, ...]`) — Nonnegative, dimensionless molecule counts per species, with at least one positive count.
- `formation_diameter` (`float`) — Positive formation diameter [m].
- `precursor_number_concentration_lower` (`float`) — Inclusive lower bound for ``C`` [#/m³].
- `precursor_number_concentration_upper` (`float`) — Inclusive upper bound for ``C`` [#/m³].
- `temperature_lower` (`float`) — Inclusive temperature lower bound [K].
- `temperature_upper` (`float`) — Inclusive temperature upper bound [K].
- `saturation_lower` (`float | None`) — Optional inclusive, dimensionless saturation lower bound. Must be paired with ``saturation_upper``.
- `saturation_upper` (`float | None`) — Optional inclusive, dimensionless saturation upper bound. Must be paired with ``saturation_lower``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L150-L271)

### particula.gpu.kernels.nucleation.NucleationConfig.coefficient

*attribute* · *instance attribute*

```python
coefficient: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L183-L183)

### particula.gpu.kernels.nucleation.NucleationConfig.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L187-L187)

### particula.gpu.kernels.nucleation.NucleationConfig.molecule_counts

*attribute* · *instance attribute*

```python
molecule_counts: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L186-L186)

### particula.gpu.kernels.nucleation.NucleationConfig.precursor_index

*attribute* · *instance attribute*

```python
precursor_index: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L185-L185)

### particula.gpu.kernels.nucleation.NucleationConfig.precursor_number_concentration_lower

*attribute* · *instance attribute*

```python
precursor_number_concentration_lower: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L188-L188)

### particula.gpu.kernels.nucleation.NucleationConfig.precursor_number_concentration_upper

*attribute* · *instance attribute*

```python
precursor_number_concentration_upper: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L189-L189)

### particula.gpu.kernels.nucleation.NucleationConfig.rate_law

*attribute* · *instance attribute*

```python
rate_law: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L182-L182)

### particula.gpu.kernels.nucleation.NucleationConfig.saturation_lower

*attribute* · *class attribute* · *instance attribute*

```python
saturation_lower: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L192-L192)

### particula.gpu.kernels.nucleation.NucleationConfig.saturation_upper

*attribute* · *class attribute* · *instance attribute*

```python
saturation_upper: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L193-L193)

### particula.gpu.kernels.nucleation.NucleationConfig.survival_factor

*attribute* · *instance attribute*

```python
survival_factor: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L184-L184)

### particula.gpu.kernels.nucleation.NucleationConfig.temperature_lower

*attribute* · *instance attribute*

```python
temperature_lower: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L190-L190)

### particula.gpu.kernels.nucleation.NucleationConfig.temperature_upper

*attribute* · *instance attribute*

```python
temperature_upper: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L191-L191)

### particula.gpu.kernels.nucleation.NucleationConfig.__init__

*method*

```python
def __init__(rate_law: str, coefficient: float, survival_factor: float, precursor_index: int, molecule_counts: tuple[int, ...], formation_diameter: float, precursor_number_concentration_lower: float, precursor_number_concentration_upper: float, temperature_lower: float, temperature_upper: float, saturation_lower: float | None = None, saturation_upper: float | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

### particula.gpu.kernels.nucleation.NucleationConfig.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate scalar-only configuration consistency.

**Raises**

- `TypeError` — If a scalar or molecule count has an unsupported type.
- `ValueError` — If a value is nonfinite, out of range, or inconsistent.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L195-L250)

### particula.gpu.kernels.nucleation.NucleationConfig._interval

*method* · *staticmethod*

```python
def _interval(lower: object, upper: object, name: str, *, positive: bool) -> None
```

Validate an ordered scalar interval.

**Parameters**

- `lower` (`object`) — Proposed inclusive lower endpoint.
- `upper` (`object`) — Proposed inclusive upper endpoint.
- `name` (`str`) — Name used in validation errors.
- `positive` (`bool`) — Whether both endpoints must be strictly positive.

**Raises**

- `TypeError` — If either endpoint is not a real scalar.
- `ValueError` — If an endpoint is invalid or the interval is reversed.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L252-L271)

## particula.gpu.kernels.nucleation.NucleationDiagnosticBuffers

*class* · *dataclass*

```python
class NucleationDiagnosticBuffers
```

Reference caller-owned demand planning/capacity staging diagnostic
    sidecars.

**Attributes**

- `gate_codes` (`Any`) — demand planning-owned, same-device contiguous ``wp.int32`` array shaped ``(B,)`` for gate diagnostics. Zero denotes eligible; codes 1--6 denote the documented zero/low gates; and ``7 + s`` records an inventory-limited box whose limiting species index is ``s``.
- `selected_slot_indices` (`Any`) — Same-device contiguous ``wp.int32`` array shaped ``(B, N)`` for capacity staging's deterministic selectable free-slot prefix, limited by current free capacity rather than provisional demand. Demand planning leaves it byte-for-byte unchanged. Capacity staging reserves ``-1`` for unused tails; input validation does not inspect, initialize, or otherwise alter stale output values.
- `free_slot_indices` (`Any`) — capacity staging/fixed-slot management-owned, same-device contiguous ``wp.int32`` array shaped ``(B, N)`` for ascending free slot indices with ``-1`` tails.
- `active_slot_counts` (`Any`) — capacity staging/fixed-slot management-owned, same-device contiguous ``wp.int32`` array shaped ``(B,)`` for active slot counts.
- `free_slot_counts` (`Any`) — capacity staging/fixed-slot management-owned, same-device contiguous ``wp.int32`` array shaped ``(B,)`` for free slot counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L321-L357)

### particula.gpu.kernels.nucleation.NucleationDiagnosticBuffers.active_slot_counts

*attribute* · *instance attribute*

```python
active_slot_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L356-L356)

### particula.gpu.kernels.nucleation.NucleationDiagnosticBuffers.free_slot_counts

*attribute* · *instance attribute*

```python
free_slot_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L357-L357)

### particula.gpu.kernels.nucleation.NucleationDiagnosticBuffers.free_slot_indices

*attribute* · *instance attribute*

```python
free_slot_indices: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L355-L355)

### particula.gpu.kernels.nucleation.NucleationDiagnosticBuffers.gate_codes

*attribute* · *instance attribute*

```python
gate_codes: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L353-L353)

### particula.gpu.kernels.nucleation.NucleationDiagnosticBuffers.selected_slot_indices

*attribute* · *instance attribute*

```python
selected_slot_indices: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L354-L354)

### particula.gpu.kernels.nucleation.NucleationDiagnosticBuffers.__init__

*method*

```python
def __init__(gate_codes: Any, selected_slot_indices: Any, free_slot_indices: Any, active_slot_counts: Any, free_slot_counts: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

## particula.gpu.kernels.nucleation.NucleationExhaustionBuffers

*class* · *dataclass*

```python
class NucleationExhaustionBuffers
```

Reference caller-owned exhaustion resolution policy workspace and
    finalized diagnostics.

Demand planning admitted demand and capacity staging provisional counts are
historical immutable
handoffs. Exhaustion resolution copies admitted event counts into
``demand_workspace``.
Post-policy counts become ``final_counts``; dividing them by current box
volume produces ``final_demand``.
Expected exhaustion resolution all-box preflight rejections occur before
workspace writes or
either primitive and preserve every supplied sidecar. Once a selected
primitive begins, its own planning and commit failure contract applies;
Exhaustion resolution does not provide cross-primitive rollback. This
concrete-only record is
not exported through ``particula.gpu.kernels`` or ``particula.gpu``.

**Attributes**

- `resampling_buffers` (`ResamplingBuffers`) — Caller-owned slot exhaustion resampling scratch and diagnostic storage.
- `demand_workspace` (`Any`) — Mutable ``wp.float64`` ``(B,)`` admitted event counts, scaled with representative volume before finalization.
- `final_demand` (`Any`) — ``wp.float64`` ``(B,)`` exhaustion resolution-finalized demand [#/m³].
- `requested_scale` (`Any`) — ``wp.float64`` ``(B,)`` requested volume scales.
- `minimum_scale` (`Any`) — ``wp.float64`` ``(B,)`` lower bounds for volume scales.
- `minimum_volume` (`Any`) — ``wp.float64`` ``(B,)`` minimum allowed box volumes [m³].
- `resolved_scale` (`Any`) — ``wp.float64`` ``(B,)`` scale selected by exhaustion resolution; unscaled rows receive ``1.0``.
- `resampling_releasable_counts` (`Any`) — ``wp.int32`` ``(B,)`` maximum counts resampling may release.
- `required_release_counts` (`Any`) — ``wp.int32`` ``(B,)`` exhaustion resolution output deficits before policy application.
- `scaling_required` (`Any`) — ``wp.int32`` ``(B,)`` exhaustion resolution output flags for rows selected for scaling.
- `final_counts` (`Any`) — ``wp.int32`` ``(B,)`` exhaustion resolution-finalized event counts.
- `final_selected_slot_indices` (`Any`) — ``wp.int32`` ``(B, N)`` exhaustion resolution-finalized ascending free-slot prefixes with ``-1`` tails.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L390-L449)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.demand_workspace

*attribute* · *instance attribute*

```python
demand_workspace: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L439-L439)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.final_counts

*attribute* · *instance attribute*

```python
final_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L448-L448)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.final_demand

*attribute* · *instance attribute*

```python
final_demand: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L440-L440)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.final_selected_slot_indices

*attribute* · *instance attribute*

```python
final_selected_slot_indices: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L449-L449)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.minimum_scale

*attribute* · *instance attribute*

```python
minimum_scale: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L442-L442)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.minimum_volume

*attribute* · *instance attribute*

```python
minimum_volume: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L443-L443)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.requested_scale

*attribute* · *instance attribute*

```python
requested_scale: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L441-L441)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.required_release_counts

*attribute* · *instance attribute*

```python
required_release_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L446-L446)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.resampling_buffers

*attribute* · *instance attribute*

```python
resampling_buffers: ResamplingBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L438-L438)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.resampling_releasable_counts

*attribute* · *instance attribute*

```python
resampling_releasable_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L445-L445)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.resolved_scale

*attribute* · *instance attribute*

```python
resolved_scale: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L444-L444)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.scaling_required

*attribute* · *instance attribute*

```python
scaling_required: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L447-L447)

### particula.gpu.kernels.nucleation.NucleationExhaustionBuffers.__init__

*method*

```python
def __init__(resampling_buffers: ResamplingBuffers, demand_workspace: Any, final_demand: Any, requested_scale: Any, minimum_scale: Any, minimum_volume: Any, resolved_scale: Any, resampling_releasable_counts: Any, required_release_counts: Any, scaling_required: Any, final_counts: Any, final_selected_slot_indices: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

## particula.gpu.kernels.nucleation.NucleationExhaustionControls

*class* · *dataclass*

```python
class NucleationExhaustionControls
```

Store exact-boolean private exhaustion resolution capacity-policy
    controls.

``resampling`` is considered first for a row only when it can release its
entire capacity staging deficit. ``representative_volume_scaling`` is then
considered for
any remaining exhausted row. Exact Python booleans deliberately prevent
integer or NumPy scalar substitutes at this concrete-only seam.

**Attributes**

- `resampling` (`bool`) — Enable fully viable resampling before scaling fallback.
- `representative_volume_scaling` (`bool`) — Enable representative-volume scaling for exhausted rows not selected for resampling.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L360-L387)

### particula.gpu.kernels.nucleation.NucleationExhaustionControls.representative_volume_scaling

*attribute* · *instance attribute*

```python
representative_volume_scaling: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L378-L378)

### particula.gpu.kernels.nucleation.NucleationExhaustionControls.resampling

*attribute* · *instance attribute*

```python
resampling: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L377-L377)

### particula.gpu.kernels.nucleation.NucleationExhaustionControls.__init__

*method*

```python
def __init__(resampling: bool, representative_volume_scaling: bool) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

### particula.gpu.kernels.nucleation.NucleationExhaustionControls.__post_init__

*method*

```python
def __post_init__() -> None
```

Reject non-exact Boolean policy controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L380-L387)

## particula.gpu.kernels.nucleation.NucleationFinalizedDemandBuffers

*class* · *dataclass*

```python
class NucleationFinalizedDemandBuffers
```

Reference caller-owned demand planning/capacity staging finalized demand
    and transfer sidecars.

**Attributes**

- `accepted_counts` (`Any`) — capacity staging-owned, same-device contiguous ``wp.int32`` array shaped ``(B,)`` for full provisional event counts. Counts retain demand beyond current free capacity; demand planning leaves this field byte-for-byte unchanged.
- `accepted_demand` (`Any`) — Same-device contiguous ``wp.float64`` array shaped ``(B,)`` for demand planning inventory-admitted source demand [#/m³].
- `precursor_mass_change` (`Any`) — Same-device contiguous ``wp.float64`` array shaped ``(B, S)`` for demand planning planned positive species removal [kg/m³].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L296-L318)

### particula.gpu.kernels.nucleation.NucleationFinalizedDemandBuffers.accepted_counts

*attribute* · *instance attribute*

```python
accepted_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L316-L316)

### particula.gpu.kernels.nucleation.NucleationFinalizedDemandBuffers.accepted_demand

*attribute* · *instance attribute*

```python
accepted_demand: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L317-L317)

### particula.gpu.kernels.nucleation.NucleationFinalizedDemandBuffers.precursor_mass_change

*attribute* · *instance attribute*

```python
precursor_mass_change: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L318-L318)

### particula.gpu.kernels.nucleation.NucleationFinalizedDemandBuffers.__init__

*method*

```python
def __init__(accepted_counts: Any, accepted_demand: Any, precursor_mass_change: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

## particula.gpu.kernels.nucleation.NucleationScratchBuffers

*class* · *dataclass*

```python
class NucleationScratchBuffers
```

Reference caller-owned demand planning sidecars without taking ownership.

**Attributes**

- `precursor_number_concentration` (`Any`) — demand planning-owned, same-device contiguous ``wp.float64`` array shaped ``(B,)`` for selected precursor number concentration [#/m³].
- `potential_rate` (`Any`) — Same-device contiguous ``wp.float64`` array shaped ``(B,)`` for demand planning potential formation rate [#/m³/s].
- `potential_demand` (`Any`) — Same-device contiguous ``wp.float64`` array shaped ``(B,)`` for demand planning potential source demand [#/m³]. Capacity staging does not write this field.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L274-L293)

### particula.gpu.kernels.nucleation.NucleationScratchBuffers.potential_demand

*attribute* · *instance attribute*

```python
potential_demand: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L293-L293)

### particula.gpu.kernels.nucleation.NucleationScratchBuffers.potential_rate

*attribute* · *instance attribute*

```python
potential_rate: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L292-L292)

### particula.gpu.kernels.nucleation.NucleationScratchBuffers.precursor_number_concentration

*attribute* · *instance attribute*

```python
precursor_number_concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L291-L291)

### particula.gpu.kernels.nucleation.NucleationScratchBuffers.__init__

*method*

```python
def __init__(precursor_number_concentration: Any, potential_rate: Any, potential_demand: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

## particula.gpu.kernels.nucleation._NucleationPreflight

*class* · *dataclass*

```python
class _NucleationPreflight
```

Retain normalized read-only input validation metadata for private phase
    handoff.

This private test seam is not a supported API. Its references preserve
caller identity and never authorize mutation. Zero-valued diagnostic
fields describe all-box gates without clearing supplied stale sidecars.

**Attributes**

- `particles` (`Any`) — Validated caller-owned Warp particle container.
- `gas` (`Any`) — Validated caller-owned Warp gas container.
- `config` (`NucleationConfig`) — Validated nucleation configuration.
- `n_boxes` (`int`) — Fixed number of boxes ``B``.
- `n_particles` (`int`) — Fixed particle capacity ``N`` per box.
- `n_species` (`int`) — Fixed particle-material species count ``Sp``.
- `species_binding` (`NucleationSpeciesBinding`) — Explicit native map retaining independent gas width.
- `device` (`Any`) — Shared Warp device for all retained arrays.
- `temperature` (`Any`) — Validated scalar or same-device ``(B,)`` temperature [K].
- `saturation` (`Any | None`) — Validated same-device ``(B, S)`` saturation, if configured.
- `has_eligible_boxes` (`bool`) — Whether at least one box can proceed past input validation.
- `has_gated_boxes` (`bool`) — Whether at least one valid box is currently gated.
- `gate_reason` (`str | None`) — All-box gate reason, or ``None`` for mixed/no gates.
- `potential_rate` (`float`) — All-box gate diagnostic potential rate [#/m³/s].
- `potential_demand` (`float`) — All-box gate diagnostic potential demand.
- `accepted_count` (`int`) — All-box gate diagnostic accepted count.
- `accepted_demand` (`float`) — All-box gate diagnostic accepted demand.
- `precursor_mass_change` (`float`) — All-box gate diagnostic precursor change [kg/m³].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L452-L560)

### particula.gpu.kernels.nucleation._NucleationPreflight.accepted_count

*attribute* · *class attribute* · *instance attribute*

```python
accepted_count: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L499-L499)

### particula.gpu.kernels.nucleation._NucleationPreflight.accepted_demand

*attribute* · *class attribute* · *instance attribute*

```python
accepted_demand: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L500-L500)

### particula.gpu.kernels.nucleation._NucleationPreflight.config

*attribute* · *instance attribute*

```python
config: NucleationConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L485-L485)

### particula.gpu.kernels.nucleation._NucleationPreflight.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L489-L489)

### particula.gpu.kernels.nucleation._NucleationPreflight.diagnostics

*attribute* · *class attribute* · *instance attribute*

```python
diagnostics: NucleationDiagnosticBuffers | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L503-L503)

### particula.gpu.kernels.nucleation._NucleationPreflight.finalized_demand

*attribute* · *class attribute* · *instance attribute*

```python
finalized_demand: NucleationFinalizedDemandBuffers | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L502-L502)

### particula.gpu.kernels.nucleation._NucleationPreflight.gas

*attribute* · *instance attribute*

```python
gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L484-L484)

### particula.gpu.kernels.nucleation._NucleationPreflight.gas_arrays

*attribute* · *class attribute* · *instance attribute*

```python
gas_arrays: tuple[Any, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L507-L507)

### particula.gpu.kernels.nucleation._NucleationPreflight.gate_reason

*attribute* · *instance attribute*

```python
gate_reason: str | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L494-L494)

### particula.gpu.kernels.nucleation._NucleationPreflight.has_eligible_boxes

*attribute* · *instance attribute*

```python
has_eligible_boxes: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L492-L492)

### particula.gpu.kernels.nucleation._NucleationPreflight.has_gated_boxes

*attribute* · *instance attribute*

```python
has_gated_boxes: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L493-L493)

### particula.gpu.kernels.nucleation._NucleationPreflight.n_boxes

*attribute* · *instance attribute*

```python
n_boxes: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L486-L486)

### particula.gpu.kernels.nucleation._NucleationPreflight.n_particles

*attribute* · *instance attribute*

```python
n_particles: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L487-L487)

### particula.gpu.kernels.nucleation._NucleationPreflight.n_species

*attribute* · *instance attribute*

```python
n_species: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L488-L488)

### particula.gpu.kernels.nucleation._NucleationPreflight.normalized_time_step

*attribute* · *instance attribute*

```python
normalized_time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L495-L495)

### particula.gpu.kernels.nucleation._NucleationPreflight.p3_sidecars

*attribute* · *class attribute* · *instance attribute*

```python
p3_sidecars: tuple[Any, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L504-L504)

### particula.gpu.kernels.nucleation._NucleationPreflight.particle_arrays

*attribute* · *class attribute* · *instance attribute*

```python
particle_arrays: tuple[Any, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L506-L506)

### particula.gpu.kernels.nucleation._NucleationPreflight.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L483-L483)

### particula.gpu.kernels.nucleation._NucleationPreflight.potential_demand

*attribute* · *class attribute* · *instance attribute*

```python
potential_demand: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L498-L498)

### particula.gpu.kernels.nucleation._NucleationPreflight.potential_rate

*attribute* · *class attribute* · *instance attribute*

```python
potential_rate: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L497-L497)

### particula.gpu.kernels.nucleation._NucleationPreflight.precursor_mass_change

*attribute* · *class attribute* · *instance attribute*

```python
precursor_mass_change: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L501-L501)

### particula.gpu.kernels.nucleation._NucleationPreflight.protected_inputs

*attribute* · *class attribute* · *instance attribute*

```python
protected_inputs: tuple[Any, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L508-L508)

### particula.gpu.kernels.nucleation._NucleationPreflight.protected_sidecars

*attribute* · *class attribute* · *instance attribute*

```python
protected_sidecars: tuple[Any, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L505-L505)

### particula.gpu.kernels.nucleation._NucleationPreflight.saturation

*attribute* · *instance attribute*

```python
saturation: Any | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L491-L491)

### particula.gpu.kernels.nucleation._NucleationPreflight.species_binding

*attribute* · *instance attribute*

```python
species_binding: NucleationSpeciesBinding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L496-L496)

### particula.gpu.kernels.nucleation._NucleationPreflight.temperature

*attribute* · *instance attribute*

```python
temperature: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L490-L490)

### particula.gpu.kernels.nucleation._NucleationPreflight.__init__

*method*

```python
def __init__(particles: Any, gas: Any, config: NucleationConfig, n_boxes: int, n_particles: int, n_species: int, device: Any, temperature: Any, saturation: Any | None, has_eligible_boxes: bool, has_gated_boxes: bool, gate_reason: str | None, normalized_time_step: float, species_binding: NucleationSpeciesBinding, potential_rate: float = 0.0, potential_demand: float = 0.0, accepted_count: int = 0, accepted_demand: float = 0.0, precursor_mass_change: float = 0.0, finalized_demand: NucleationFinalizedDemandBuffers | None = None, diagnostics: NucleationDiagnosticBuffers | None = None, p3_sidecars: tuple[Any, ...] = (), protected_sidecars: tuple[Any, ...] = (), particle_arrays: tuple[Any, ...] = (), gas_arrays: tuple[Any, ...] = (), protected_inputs: tuple[Any, ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

### particula.gpu.kernels.nucleation._NucleationPreflight.__post_init__

*method*

```python
def __post_init__() -> None
```

Capture input validation storage identities for later mutable
exhaustion resolution validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L510-L560)

## particula.gpu.kernels.nucleation._PreparedNucleationCall

*class* · *dataclass*

```python
class _PreparedNucleationCall
```

Pin one allocation-complete preflight through commit nucleation launch
    sequence.

Preparation fixes container-array identities, dimensions, device, scalar
controls, immutable policy controls, and private workspace allocations.
Pinned arrays retain live device payload authority. Enqueue runs the
established device-side dynamic validation and writers using those retained
references; observation performs the sole bounded status readback. This is
concrete-module-only implementation state, not a public enqueue protocol.

**Attributes**

- `result` (`tuple[Any, Any]`) — Original particle and gas containers returned by the public boundary.
- `particle_arrays` (`tuple[Any, ...]`) — Pinned particle fields in input validation order.
- `gas_arrays` (`tuple[Any, ...]`) — Pinned gas fields in input validation order.
- `temperature` (`Any`) — Pinned or preparation-owned temperature array [K].
- `saturation` (`Any`) — Pinned or preparation-owned saturation array.
- `scalar_controls` (`tuple[Any, ...]`) — Validated scalar input validation--demand planning controls.
- `policy_controls` (`tuple[int, int]`) — Frozen integer exhaustion resolution resampling and scaling controls.
- `molecule_counts` (`Any`) — Preparation-owned molecule counts per event.
- `scratch_arrays` (`tuple[Any, ...]`) — Pinned caller-owned demand planning sidecars.
- `finalized_arrays` (`tuple[Any, ...]`) — Pinned caller-owned demand planning/capacity staging finalized sidecars.
- `diagnostic_arrays` (`tuple[Any, ...]`) — Pinned caller-owned demand planning/capacity staging diagnostic sidecars.
- `p4_arrays` (`tuple[Any, ...]`) — Pinned caller-owned exhaustion resolution workspace and diagnostic sidecars.
- `p2_work` (`tuple[Any, ...]`) — Preparation-owned demand planning work arrays.
- `p3_work` (`tuple[Any, ...]`) — Preparation-owned capacity staging work arrays.
- `p4_work` (`tuple[Any, ...]`) — Preparation-owned exhaustion resolution work arrays.
- `p1_status` (`Any`) — Preparation-owned ordered input validation status counters.
- `pipeline_invalid` (`Any`) — Preparation-owned aggregate device failure gate.
- `p2_status` (`Any`) — Preparation-owned demand planning status counters.
- `p3_status` (`Any`) — Preparation-owned capacity staging status counters.
- `molecule_status` (`Any`) — Preparation-owned commit molecule-eligibility status.
- `p4_status` (`Any`) — Preparation-owned exhaustion resolution handoff and policy status.
- `p4_final_status` (`Any`) — Preparation-owned exhaustion resolution finalized-capacity status.
- `p5_status` (`Any`) — Preparation-owned commit handoff status.
- `resampling` (`_PreparedResamplingCall | None`) — Prepared nested resampling call, when dimensions permit it.
- `scaling` (`_PreparedRepresentativeVolumeScalingCall | None`) — Prepared nested scaling call, when dimensions permit it.
- `dimensions` (`tuple[int, int, int]`) — Fixed ``(B, N, S)`` dimensions.
- `device` (`Any`) — Shared active Warp device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3356-L3434)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3432-L3432)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.diagnostic_arrays

*attribute* · *instance attribute*

```python
diagnostic_arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3416-L3416)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.dimensions

*attribute* · *instance attribute*

```python
dimensions: tuple[int, int, int]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3431-L3431)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.finalized_arrays

*attribute* · *instance attribute*

```python
finalized_arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3415-L3415)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.gas_arrays

*attribute* · *instance attribute*

```python
gas_arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3408-L3408)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.gas_species

*attribute* · *class attribute* · *instance attribute*

```python
gas_species: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3434-L3434)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.molecule_counts

*attribute* · *instance attribute*

```python
molecule_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3413-L3413)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.molecule_status

*attribute* · *instance attribute*

```python
molecule_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3425-L3425)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p1_status

*attribute* · *instance attribute*

```python
p1_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3421-L3421)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p2_status

*attribute* · *instance attribute*

```python
p2_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3423-L3423)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p2_work

*attribute* · *instance attribute*

```python
p2_work: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3418-L3418)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p3_status

*attribute* · *instance attribute*

```python
p3_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3424-L3424)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p3_work

*attribute* · *instance attribute*

```python
p3_work: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3419-L3419)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p4_arrays

*attribute* · *instance attribute*

```python
p4_arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3417-L3417)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p4_final_status

*attribute* · *instance attribute*

```python
p4_final_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3427-L3427)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p4_status

*attribute* · *instance attribute*

```python
p4_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3426-L3426)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p4_work

*attribute* · *instance attribute*

```python
p4_work: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3420-L3420)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.p5_status

*attribute* · *instance attribute*

```python
p5_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3428-L3428)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.particle_arrays

*attribute* · *instance attribute*

```python
particle_arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3407-L3407)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.particle_to_gas

*attribute* · *class attribute* · *instance attribute*

```python
particle_to_gas: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3433-L3433)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.pipeline_invalid

*attribute* · *instance attribute*

```python
pipeline_invalid: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3422-L3422)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.policy_controls

*attribute* · *instance attribute*

```python
policy_controls: tuple[int, int]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3412-L3412)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.resampling

*attribute* · *instance attribute*

```python
resampling: _PreparedResamplingCall | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3429-L3429)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.result

*attribute* · *instance attribute*

```python
result: tuple[Any, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3406-L3406)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.saturation

*attribute* · *instance attribute*

```python
saturation: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3410-L3410)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.scalar_controls

*attribute* · *instance attribute*

```python
scalar_controls: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3411-L3411)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.scaling

*attribute* · *instance attribute*

```python
scaling: _PreparedRepresentativeVolumeScalingCall | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3430-L3430)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.scratch_arrays

*attribute* · *instance attribute*

```python
scratch_arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3414-L3414)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.temperature

*attribute* · *instance attribute*

```python
temperature: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3409-L3409)

### particula.gpu.kernels.nucleation._PreparedNucleationCall.__init__

*method*

```python
def __init__(result: tuple[Any, Any], particle_arrays: tuple[Any, ...], gas_arrays: tuple[Any, ...], temperature: Any, saturation: Any, scalar_controls: tuple[Any, ...], policy_controls: tuple[int, int], molecule_counts: Any, scratch_arrays: tuple[Any, ...], finalized_arrays: tuple[Any, ...], diagnostic_arrays: tuple[Any, ...], p4_arrays: tuple[Any, ...], p2_work: tuple[Any, ...], p3_work: tuple[Any, ...], p4_work: tuple[Any, ...], p1_status: Any, pipeline_invalid: Any, p2_status: Any, p3_status: Any, molecule_status: Any, p4_status: Any, p4_final_status: Any, p5_status: Any, resampling: _PreparedResamplingCall | None, scaling: _PreparedRepresentativeVolumeScalingCall | None, dimensions: tuple[int, int, int], device: Any, particle_to_gas: Any = None, gas_species: int = 0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L0-L0)

## particula.gpu.kernels.nucleation._aggregate_p4_preflight_status

*function*

```python
def _aggregate_p4_preflight_status(invalid: wp.array(dtype=wp.int32), release: wp.array(dtype=wp.int32), scaling: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32)) -> None
```

Aggregate all exhaustion resolution preflight results into one scalar
status word.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2526-L2542)

## particula.gpu.kernels.nucleation._aggregate_prepared_status

*function*

```python
def _aggregate_prepared_status(status: wp.array(dtype=wp.int32), pipeline_invalid: wp.array(dtype=wp.int32)) -> None
```

Promote any phase-local status lane into the pipeline gate.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3305-L3313)

## particula.gpu.kernels.nucleation._array

*function*

```python
def _array(container: Any, name: str, dtype: Any, shape: tuple[int, ...], device: Any) -> Any
```

Validate one same-device contiguous Warp-array schema.

**Parameters**

- `container` (`Any`) — Object that owns the required field.
- `name` (`str`) — Required field name.
- `dtype` (`Any`) — Required Warp dtype.
- `shape` (`tuple[int, ...]`) — Required fixed array shape.
- `device` (`Any`) — Required Warp device.

**Returns**

- (`Any`) — The validated array, retaining caller identity.

**Raises**

- `ValueError` — If the field is missing or has an invalid schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L582-L610)

## particula.gpu.kernels.nucleation._commit_demand_plan

*function*

```python
def _commit_demand_plan(number_concentration: wp.array(dtype=wp.float64), rate: wp.array(dtype=wp.float64), potential: wp.array(dtype=wp.float64), admitted: wp.array(dtype=wp.float64), removal: wp.array2d(dtype=wp.float64), gate_codes: wp.array(dtype=wp.int32), species_count: wp.int32, scratch_number_concentration: wp.array(dtype=wp.float64), scratch_rate: wp.array(dtype=wp.float64), scratch_demand: wp.array(dtype=wp.float64), finalized_demand: wp.array(dtype=wp.float64), finalized_removal: wp.array2d(dtype=wp.float64), diagnostics: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Commit exactly the demand planning-owned sidecar fields for one box.

**Parameters**

- `number_concentration` (`wp.array(dtype=wp.float64)`) — Validated private precursor concentration [#/m³].
- `rate` (`wp.array(dtype=wp.float64)`) — Validated private formation rate [#/m³/s].
- `potential` (`wp.array(dtype=wp.float64)`) — Validated private potential demand [#/m³].
- `admitted` (`wp.array(dtype=wp.float64)`) — Validated private inventory-admitted demand [#/m³].
- `removal` (`wp.array2d(dtype=wp.float64)`) — Validated private planned species removal [kg/m³].
- `gate_codes` (`wp.array(dtype=wp.int32)`) — Validated private demand planning gate and limiter codes.
- `species_count` (`wp.int32`) — Fixed number of species ``S``.
- `scratch_number_concentration` (`wp.array(dtype=wp.float64)`) — demand planning-owned precursor-concentration sidecar.
- `scratch_rate` (`wp.array(dtype=wp.float64)`) — demand planning-owned potential-rate sidecar.
- `scratch_demand` (`wp.array(dtype=wp.float64)`) — demand planning-owned potential-demand sidecar.
- `finalized_demand` (`wp.array(dtype=wp.float64)`) — demand planning-owned admitted-demand sidecar.
- `finalized_removal` (`wp.array2d(dtype=wp.float64)`) — demand planning-owned planned-removal sidecar.
- `diagnostics` (`wp.array(dtype=wp.int32)`) — demand planning-owned gate-code sidecar.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L1787-L1832)

## particula.gpu.kernels.nucleation._commit_native_nucleation

*function*

```python
def _commit_native_nucleation(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), gas: wp.array2d(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), molecules: wp.array(dtype=wp.int32), particle_to_gas: wp.array(dtype=wp.int32), final_demand: wp.array(dtype=wp.float64), final_counts: wp.array(dtype=wp.int32), selected: wp.array2d(dtype=wp.int32), status: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Activate unit-count resolved slots and consume full-order gas density.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3670-L3709)

## particula.gpu.kernels.nucleation._commit_nucleation_p5

*function*

```python
def _commit_nucleation_p5(preflight: _NucleationPreflight, buffers: NucleationExhaustionBuffers, p4_storage: tuple[Any, ...]) -> bool
```

Validate the final exhaustion resolution record and launch one fused
commit if needed.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3083-L3154)

## particula.gpu.kernels.nucleation._commit_prepared_p3_diagnostics

*function*

```python
def _commit_prepared_p3_diagnostics(categories: wp.array2d(dtype=wp.int32), free_indices: wp.array2d(dtype=wp.int32), active_counts: wp.array(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Commit fixed-slot management diagnostics only after complete capacity
staging validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3327-L3353)

## particula.gpu.kernels.nucleation._commit_staged_nucleation_slots

*function*

```python
def _commit_staged_nucleation_slots(counts: wp.array(dtype=wp.int32), free_slot_indices: wp.array2d(dtype=wp.int32), free_slot_counts: wp.array(dtype=wp.int32), accepted_counts: wp.array(dtype=wp.int32), selected_slot_indices: wp.array2d(dtype=wp.int32), n_particles: int, invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Commit full capacity staging counts and the selectable fixed-slot
    management free-slot prefix.

Full provisional counts are retained even when they exceed free capacity.
Every selected-index lane outside the deterministic selectable prefix is
overwritten with ``-1``.

**Parameters**

- `counts` (`wp.array(dtype=wp.int32)`) — Validated private provisional event counts, shaped ``(B,)``.
- `free_slot_indices` (`wp.array2d(dtype=wp.int32)`) — fixed-slot management ascending free-slot indices, shaped ``(B, N)``.
- `free_slot_counts` (`wp.array(dtype=wp.int32)`) — fixed-slot management free-slot counts, shaped ``(B,)``.
- `accepted_counts` (`wp.array(dtype=wp.int32)`) — capacity staging output for full provisional counts, shaped ``(B,)``.
- `selected_slot_indices` (`wp.array2d(dtype=wp.int32)`) — capacity staging output for selectable free-slot indices, shaped ``(B, N)``.
- `n_particles` (`int`) — Fixed particle capacity ``N`` per box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2073-L2114)

## particula.gpu.kernels.nucleation._convert_admitted_demand_to_counts

*function*

```python
def _convert_admitted_demand_to_counts(accepted_demand: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), maximum_count: wp.float64, counts: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), invalid_index: wp.int32, upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Convert per-volume admitted demand to integral provisional counts.

This private capacity staging kernel writes only private workspace. It
rejects derived
counts that are nonfinite, negative, nonintegral, or outside the inclusive
``int32`` range before any caller-owned capacity staging or fixed-slot
management sidecar is written.

**Parameters**

- `accepted_demand` (`wp.array(dtype=wp.float64)`) — demand planning-admitted demand [#/m³], shaped ``(B,)``.
- `volume` (`wp.array(dtype=wp.float64)`) — Validated particle-box volume [m³], shaped ``(B,)``.
- `maximum_count` (`wp.float64`) — Largest representable provisional ``int32`` count.
- `counts` (`wp.array(dtype=wp.int32)`) — Private output for provisional counts, shaped ``(B,)``.
- `invalid` (`wp.array(dtype=wp.int32)`) — Private status counters shared with slot classification.
- `invalid_index` (`wp.int32`) — Status lane reserved for invalid conversions.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2027-L2070)

## particula.gpu.kernels.nucleation._convert_staged_nucleation_counts

*function*

```python
def _convert_staged_nucleation_counts(accepted_demand: Any, volume: Any, device: Any) -> tuple[Any, Any, Any]
```

Convert admitted demand to private counts and a shared invalid flag.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2148-L2171)

## particula.gpu.kernels.nucleation._disabled_precursor

*function*

```python
def _disabled_precursor(values: wp.array2d(dtype=wp.int32), index: wp.int32, invalid: wp.array(dtype=wp.int32)) -> None
```

Count boxes whose selected precursor partitioning is disabled.

**Parameters**

- `values` (`wp.array2d(dtype=wp.int32)`) — Device-resident binary partitioning flags.
- `index` (`wp.int32`) — Selected precursor species index.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L793-L808)

## particula.gpu.kernels.nucleation._enqueue_prepared_nucleation_call

*function*

```python
def _enqueue_prepared_nucleation_call(prepared: _PreparedNucleationCall) -> tuple[Any, Any]
```

Enqueue one fixed preflight through commit sequence using only pinned
    arrays.

The device-only phase performs no allocation, readback, synchronization,
Policy resolution, or container lookup. There is no rollback after a writer
launch.

**Parameters**

- `prepared` (`_PreparedNucleationCall`) — Allocation-complete call with pinned arrays and controls.

**Returns**

- (`Any`) — The original ``(particles, gas)`` containers. Device errors remain in
- (`Any`) — preparation-owned status arrays until observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3712-L4123)

## particula.gpu.kernels.nucleation._field

*function*

```python
def _field(container: Any, name: str) -> Any
```

Get a required container field without accepting host substitutes.

**Parameters**

- `container` (`Any`) — Object expected to carry a Warp-backed field.
- `name` (`str`) — Required field name.

**Returns**

- (`Any`) — Field value, before its Warp-array schema is validated.

**Raises**

- `ValueError` — If ``container`` does not provide ``name``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L563-L579)

## particula.gpu.kernels.nucleation._float64_predecessor

*function*

```python
def _float64_predecessor(value: wp.float64) -> wp.float64
```

Return the exact finite float64 predecessor of a positive value.

Warp 1.11 exposes neither ``bitcast`` nor ``nextafter``/``frexp`` to
device code. Powers of two and multiplication/division by two are exact in
binary64, so determine the binade with supported scalar arithmetic and
subtract its exact ULP. Subnormals retain the minimum binary64 spacing.

**Parameters**

- `value` (`wp.float64`) — Finite positive binary64 value.

**Returns**

- (`wp.float64`) — Largest representable binary64 value strictly less than ``value``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L1589-L1617)

## particula.gpu.kernels.nucleation._floating_scalar

*function*

```python
def _floating_scalar(value: object, name: str) -> float
```

Normalize a finite positive Python or NumPy floating scalar.

**Parameters**

- `value` (`object`) — Scalar to validate.
- `name` (`str`) — Value name used in validation errors.

**Returns**

- (`float`) — Validated positive scalar.

**Raises**

- `TypeError` — If ``value`` is not a Python or NumPy floating scalar.
- `ValueError` — If ``value`` is nonfinite or not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L129-L147)

## particula.gpu.kernels.nucleation._gate_union_count

*function*

```python
def _gate_union_count(first: wp.array(dtype=wp.int32), second: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)) -> None
```

Count boxes gated by either of two per-box read-only gate masks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L871-L880)

## particula.gpu.kernels.nucleation._invalid_binary

*function*

```python
def _invalid_binary(values: wp.array2d(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Count partitioning values other than zero or one.

**Parameters**

- `values` (`wp.array2d(dtype=wp.int32)`) — Device-resident partitioning flags to scan.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L776-L790)

## particula.gpu.kernels.nucleation._invalid_float

*function*

```python
def _invalid_float(values: wp.array(dtype=wp.float64), positive: wp.int32, invalid: wp.array(dtype=wp.int32)) -> None
```

Count invalid one-dimensional floats without changing input values.

**Parameters**

- `values` (`wp.array(dtype=wp.float64)`) — Device-resident values to scan.
- `positive` (`wp.int32`) — Nonzero to require positive rather than nonnegative values.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L668-L687)

## particula.gpu.kernels.nucleation._invalid_float_2d

*function*

```python
def _invalid_float_2d(values: wp.array2d(dtype=wp.float64), positive: wp.int32, invalid: wp.array(dtype=wp.int32)) -> None
```

Count invalid two-dimensional floats without changing input values.

**Parameters**

- `values` (`wp.array2d(dtype=wp.float64)`) — Device-resident values to scan.
- `positive` (`wp.int32`) — Nonzero to require positive rather than nonnegative values.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L690-L710)

## particula.gpu.kernels.nucleation._invalid_float_3d

*function*

```python
def _invalid_float_3d(values: wp.array3d(dtype=wp.float64), positive: wp.int32, invalid: wp.array(dtype=wp.int32)) -> None
```

Count invalid three-dimensional floats without changing input values.

**Parameters**

- `values` (`wp.array3d(dtype=wp.float64)`) — Device-resident values to scan.
- `positive` (`wp.int32`) — Nonzero to require positive rather than nonnegative values.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L713-L733)

## particula.gpu.kernels.nucleation._invalid_interval_1d

*function*

```python
def _invalid_interval_1d(values: wp.array(dtype=wp.float64), lower: wp.float64, upper: wp.float64, invalid: wp.array(dtype=wp.int32)) -> None
```

Count one-dimensional values outside an inclusive interval.

**Parameters**

- `values` (`wp.array(dtype=wp.float64)`) — Device-resident values to scan.
- `lower` (`wp.float64`) — Inclusive lower endpoint.
- `upper` (`wp.float64`) — Inclusive upper endpoint.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L736-L753)

## particula.gpu.kernels.nucleation._invalid_interval_2d

*function*

```python
def _invalid_interval_2d(values: wp.array2d(dtype=wp.float64), lower: wp.float64, upper: wp.float64, invalid: wp.array(dtype=wp.int32)) -> None
```

Count two-dimensional values outside an inclusive interval.

**Parameters**

- `values` (`wp.array2d(dtype=wp.float64)`) — Device-resident values to scan.
- `lower` (`wp.float64`) — Inclusive lower endpoint.
- `upper` (`wp.float64`) — Inclusive upper endpoint.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L756-L773)

## particula.gpu.kernels.nucleation._invalid_nonfinite_2d

*function*

```python
def _invalid_nonfinite_2d(values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Count nonfinite two-dimensional values without changing input values.

**Parameters**

- `values` (`wp.array2d(dtype=wp.float64)`) — Device-resident values to scan.
- `invalid` (`wp.array(dtype=wp.int32)`) — Device-resident one-element invalid-value counter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L907-L920)

## particula.gpu.kernels.nucleation._memory_range

*function*

```python
def _memory_range(array: Any) -> tuple[int, int] | None
```

Return byte bounds after rejecting unsafe noncontiguous views.

**Parameters**

- `array` (`Any`) — Warp array with a supported overlap-check dtype.

**Returns**

- (`tuple[int, int] | None`) — Half-open byte range, or ``None`` for an empty array.

**Raises**

- `ValueError` — If the dtype is unsupported or the array is noncontiguous.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L613-L641)

## particula.gpu.kernels.nucleation._native_source_counts

*function*

```python
def _native_source_counts(accepted_counts: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Stage admitted integer counts before native exhaustion scaling.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3658-L3667)

## particula.gpu.kernels.nucleation._no_overlap

*function*

```python
def _no_overlap(arrays: tuple[Any, ...]) -> None
```

Reject identity and byte-range aliases among future mutable state.

**Parameters**

- `arrays` (`tuple[Any, ...]`) — Validated arrays that later phases could mutate.

**Raises**

- `ValueError` — If two arrays are identical or overlap in byte storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L644-L665)

## particula.gpu.kernels.nucleation._observe_prepared_nucleation_call

*function*

```python
def _observe_prepared_nucleation_call(prepared: _PreparedNucleationCall) -> tuple[Any, Any]
```

Interpret bounded device status in the established phase-error order.

This is the sole host readback boundary for a prepared call. It translates
device status counters without inspecting or replacing caller-owned payload
storage. Writer failures preserve the established no-rollback contract.

**Parameters**

- `prepared` (`_PreparedNucleationCall`) — Enqueued call whose status arrays are ready for observation.

**Returns**

- (`tuple[Any, Any]`) — The original ``(particles, gas)`` containers when every phase succeeds.

**Raises**

- `ValueError` — If stage validation or a nested
exhaustion operation
recorded an error.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L4126-L4202)

## particula.gpu.kernels.nucleation._orchestrate_nucleation_exhaustion

*function*

```python
def _orchestrate_nucleation_exhaustion(preflight: _NucleationPreflight, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers, controls: NucleationExhaustionControls, buffers: NucleationExhaustionBuffers) -> tuple[Any, ...]
```

Privately apply exhaustion resolution capacity policy and write finalized
    diagnostics.

Exhaustion resolution validates immutable demand planning/capacity staging
handoffs, then selects resampling only when it
can release a row's complete deficit. It selects representative-volume
scaling only for remaining exhausted rows, invokes each selected slot
exhaustion
primitive once, and derives final demand/count/free-slot diagnostics from
the resulting device state. Demand planning/capacity staging handoffs remain
unchanged.

All expected exhaustion resolution rejections are detected before exhaustion
resolution workspace writes or an
slot exhaustion primitive call, preserving particle, gas, demand
planning/capacity staging/exhaustion resolution, and nested
sidecar state. Once a selected primitive begins, its independent
planning/commit failure contract applies; exhaustion resolution
intentionally provides no
cross-primitive rollback. This concrete-only, unexported seam does not
activate slots, mutate gas or source mass, resize storage, transfer data to
the host, use a CPU fallback, or provide direct-process integration.

**Parameters**

- `preflight` (`_NucleationPreflight`) — Exact private input validation metadata with validated particle and gas containers.
- `finalized_demand` (`NucleationFinalizedDemandBuffers`) — Exact immutable demand planning/capacity staging record containing admitted demand and provisional counts.
- `diagnostics` (`NucleationDiagnosticBuffers`) — Exact immutable demand planning/capacity staging record containing slot diagnostics and the provisional selected prefix.
- `controls` (`NucleationExhaustionControls`) — Exact-Boolean resampling-first/scaling-fallback controls.
- `buffers` (`NucleationExhaustionBuffers`) — Exact caller-owned exhaustion resolution and nested slot exhaustion workspace/diagnostic record.

**Raises**

- `ValueError` — If handoffs, policy inputs, sidecars, slot state, or final
demand capacity are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2655-L2904)

## particula.gpu.kernels.nucleation._plan_demand_work

*function*

```python
def _plan_demand_work(concentration: wp.array2d(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), saturation: wp.array2d(dtype=wp.float64), molecule_counts: wp.array(dtype=wp.int32), species_count: wp.int32, precursor_index: wp.int32, has_saturation: wp.int32, saturation_lower: wp.float64, coefficient: wp.float64, survival_factor: wp.float64, time_step: wp.float64, kinetic: wp.int32, number_concentration: wp.array(dtype=wp.float64), rate: wp.array(dtype=wp.float64), potential: wp.array(dtype=wp.float64), admitted: wp.array(dtype=wp.float64), removal: wp.array2d(dtype=wp.float64), gate_codes: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Plan one box of demand planning demand in private device-resident work
    storage.

Calculates ``C`` [#/m³], ``J`` [#/m³/s], and ``E_pot = J × dt`` [#/m³],
then limits one common admitted demand by participating precursor
inventories. The result preserves per-event precursor composition; planned
removal is ``E_admit × m_event`` [kg/m³]. This kernel never writes
caller-owned sidecars, particles, or gas inventory.

**Parameters**

- `concentration` (`wp.array2d(dtype=wp.float64)`) — Gas concentration [kg/m³], shaped ``(B, S)``.
- `molar_mass` (`wp.array(dtype=wp.float64)`) — Species molar masses [kg/mol], shaped ``(S,)``.
- `saturation` (`wp.array2d(dtype=wp.float64)`) — Dimensionless saturation ratios shaped ``(B, S)``.
- `molecule_counts` (`wp.array(dtype=wp.int32)`) — Dimensionless molecules per formed event, shaped ``(S,)``.
- `species_count` (`wp.int32`) — Fixed number of species ``S``.
- `precursor_index` (`wp.int32`) — Selected precursor species index.
- `has_saturation` (`wp.int32`) — Nonzero when saturation gating is configured.
- `saturation_lower` (`wp.float64`) — Inclusive saturation gate threshold.
- `coefficient` (`wp.float64`) — Nonnegative activation [1/s] or kinetic [m³/s] coefficient.
- `survival_factor` (`wp.float64`) — Nonnegative dimensionless survival factor.
- `time_step` (`wp.float64`) — Validated nonnegative step duration [s].
- `kinetic` (`wp.int32`) — Nonzero when the kinetic rate law is selected.
- `number_concentration` (`wp.array(dtype=wp.float64)`) — Private output for selected precursor ``C`` [#/m³], shaped ``(B,)``.
- `rate` (`wp.array(dtype=wp.float64)`) — Private output for gated formation rate ``J`` [#/m³/s], shaped ``(B,)``.
- `potential` (`wp.array(dtype=wp.float64)`) — Private output for potential demand ``E_pot`` [#/m³], shaped ``(B,)``.
- `admitted` (`wp.array(dtype=wp.float64)`) — Private output for inventory-admitted demand [#/m³], shaped ``(B,)``.
- `removal` (`wp.array2d(dtype=wp.float64)`) — Private output for planned species removal [kg/m³], shaped ``(B, S)``.
- `gate_codes` (`wp.array(dtype=wp.int32)`) — Private output for demand planning gate and limiter codes, shaped ``(B,)``.
- `invalid` (`wp.array(dtype=wp.int32)`) — Private two-lane counters for derived-domain and inventory-safety failures.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L1620-L1784)

## particula.gpu.kernels.nucleation._plan_nucleation_demand

*function*

```python
def _plan_nucleation_demand(particles: Any, gas: Any, config: NucleationConfig, time_step: Any, *, scratch: NucleationScratchBuffers, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers, temperature: Any | None = None, saturation: Any | None = None, environment: Any | None = None, species_binding: NucleationSpeciesBinding | None = None) -> None
```

Privately plan inventory-safe nucleation demand without state mutation.

This concrete-only direct-Warp demand planning seam calculates selected
precursor number
concentration ``C`` [#/m³], formation rate ``J`` [#/m³/s], and potential
demand ``E_pot = J × dt`` [#/m³]. It admits one common demand per box from
participating gas inventories, so every planned removal is
``E_admit × m_event`` [kg/m³]. Survival is already included in ``J`` and is
not applied again. It is intentionally not exported through
``particula.gpu.kernels`` or ``particula.gpu``.

Private scalar status readback is permitted for validation; no caller-state
transfer or CPU physics fallback occurs. After read-only input validation
and
derived-state validation, demand planning commits only
``scratch.precursor_number_concentration``, ``scratch.potential_rate``,
``scratch.potential_demand``, ``finalized_demand.accepted_demand``,
``finalized_demand.precursor_mass_change``, and ``diagnostics.gate_codes``.
It reads gas state for planning but never mutates particle state,
``gas.concentration``, capacity staging ``accepted_counts``, or capacity
staging selected-slot indices.
It neither selects/activates slots nor resolves exhaustion; capacity staging
only stages
metadata for a later capacity-policy and activation phase.

**Parameters**

- `particles` (`Any`) — Caller-owned Warp particle container used only for input validation schema and physical-state validation.
- `gas` (`Any`) — Caller-owned Warp gas container; its concentration [kg/m³] and molar mass [kg/mol] determine demand planning rates and inventory admission.
- `config` (`NucleationConfig`) — Immutable rate-law controls, event composition, and bounds.
- `time_step` (`Any`) — Finite nonnegative source-planning interval [s].
- `scratch` (`NucleationScratchBuffers`) — Exact caller-owned demand planning-sidecar record.
- `finalized_demand` (`NucleationFinalizedDemandBuffers`) — Exact caller-owned demand planning/capacity staging demand-sidecar record; demand planning writes only admitted demand and planned precursor removal.
- `diagnostics` (`NucleationDiagnosticBuffers`) — Exact caller-owned demand planning/capacity staging diagnostic-sidecar record; demand planning writes only gate codes.
- `temperature` (`Any | None`) (default: `None`) — Positive scalar [K] or same-device ``wp.float64`` array shaped ``(B,)`` for input validation.
- `saturation` (`Any | None`) (default: `None`) — Configured same-device dimensionless ``wp.float64`` array shaped ``(B, S)`` for input validation and demand planning gating.
- `environment` (`Any | None`) (default: `None`) — Optional owner of temperature and configured saturation.
- `species_binding` (`NucleationSpeciesBinding | None`) (default: `None`) — Required explicit native species binding.

**Raises**

- `TypeError` — If a direct scalar has an unsupported type.
- `ValueError` — If input validation schemas, ownership, physical inputs,
sidecars, or
derived demand planning demand are invalid, or inventory cannot be
made safe.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L1835-L1927)

## particula.gpu.kernels.nucleation._plan_nucleation_demand_from_preflight

*function*

```python
def _plan_nucleation_demand_from_preflight(preflight: _NucleationPreflight, scratch: NucleationScratchBuffers, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers) -> None
```

Run demand planning using an already validated input validation handoff.

This internal helper keeps the standalone demand planning seam intact while
allowing the
public commit boundary to perform input validation exactly once.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L1930-L2024)

## particula.gpu.kernels.nucleation._precursor_status

*function*

```python
def _precursor_status(concentration: wp.array2d(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), precursor_index: wp.int32, lower: wp.float64, upper: wp.float64, status: wp.array(dtype=wp.int32), gated: wp.array(dtype=wp.int32)) -> None
```

Count zero and out-of-interval precursor concentrations by box.

**Parameters**

- `concentration` (`wp.array2d(dtype=wp.float64)`) — Gas concentration [kg/m³], shaped ``(B, S)``.
- `molar_mass` (`wp.array(dtype=wp.float64)`) — Gas molar mass [kg/mol], shaped ``(S,)``.
- `precursor_index` (`wp.int32`) — Selected precursor species index.
- `lower` (`wp.float64`) — Inclusive number-concentration lower endpoint [#/m³].
- `upper` (`wp.float64`) — Inclusive number-concentration upper endpoint [#/m³].
- `status` (`wp.array(dtype=wp.int32)`) — Device counters for invalid and zero concentrations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L811-L841)

## particula.gpu.kernels.nucleation._preflight_nucleation

*function*

```python
def _preflight_nucleation(particles: Any, gas: Any, config: NucleationConfig, time_step: Any, temperature: Any | None = None, saturation: Any | None = None, environment: Any | None = None, scratch: NucleationScratchBuffers | None = None, finalized_demand: NucleationFinalizedDemandBuffers | None = None, diagnostics: NucleationDiagnosticBuffers | None = None, validate_payload: bool = True, _native_binding: NucleationSpeciesBinding | None = None) -> _NucleationPreflight
```

Validate the deferred input validation boundary without writing
    caller-owned arrays.

This private, concrete-only seam performs ordered schema, ownership,
physical-domain, and species checks. It accepts caller-owned same-device
fixed-shape Warp arrays, performs no caller-state transfer or CPU physics
fallback, and leaves particles, gas, environment, and stale sidecars
unchanged. A private scalar status readback is used solely for validation.

**Parameters**

- `particles` (`Any`) — Warp particle container with fixed ``(B, N, S)`` mass data.
- `gas` (`Any`) — Warp gas container with fixed ``(B, S)`` species data.
- `config` (`NucleationConfig`) — Scalar nucleation controls and scientific bounds.
- `time_step` (`Any`) — Finite nonnegative step duration [s].
- `temperature` (`Any | None`) (default: `None`) — Positive scalar [K] or same-device ``wp.float64 (B,)``.
- `saturation` (`Any | None`) (default: `None`) — Configured same-device dimensionless ``wp.float64 (B, S)``.
- `environment` (`Any | None`) (default: `None`) — Optional owner of temperature and configured saturation.
- `_native_binding` (`NucleationSpeciesBinding | None`) (default: `None`) — Required explicit native species binding.
- `scratch` (`NucleationScratchBuffers | None`) (default: `None`) — Optional caller-owned planning sidecars.
- `finalized_demand` (`NucleationFinalizedDemandBuffers | None`) (default: `None`) — Optional caller-owned finalized-demand sidecars.
- `diagnostics` (`NucleationDiagnosticBuffers | None`) (default: `None`) — Optional caller-owned diagnostic sidecars.
- `validate_payload` (`bool`) (default: `True`) — Whether to scan mutable device payloads. Prepared calls disable this so their enqueue phase owns dynamic validation.

**Returns**

- (`_NucleationPreflight`) — Private normalized metadata, including eligibility and all-box gate
- (`_NucleationPreflight`) — diagnostics. Returned zero diagnostics never clear supplied sidecars.

**Raises**

- `TypeError` — If a direct scalar has an unsupported type.
- `ValueError` — If schemas, ownership, physical values, or species settings
violate the input validation contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L1085-L1586)

## particula.gpu.kernels.nucleation._prepare_nucleation_step_gpu

*function*

```python
def _prepare_nucleation_step_gpu(particles: Any, gas: Any, config: NucleationConfig, time_step: Any, *, scratch: NucleationScratchBuffers, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers, exhaustion_controls: NucleationExhaustionControls, exhaustion_buffers: NucleationExhaustionBuffers, temperature: Any | None = None, saturation: Any | None = None, environment: Any | None = None, species_binding: NucleationSpeciesBinding | None = None) -> _PreparedNucleationCall
```

Validate metadata, allocate, and pin one private nucleation invocation.

Pinned arrays retain live payload authority: setup validates fixed schemas,
identities, and scalar controls without scanning mutable device values.
Enqueue owns the established device-side dynamic validation and writers,
while observation is the sole bounded status readback. Setup rejection is
pre-writer; rollback is not promised after a writer-capable launch.

**Parameters**

- `particles` (`Any`) — Caller-owned fixed-capacity Warp particle container.
- `gas` (`Any`) — Caller-owned same-device Warp gas container.
- `config` (`NucleationConfig`) — Immutable scalar nucleation controls.
- `time_step` (`Any`) — Finite nonnegative step duration [s].
- `scratch` (`NucleationScratchBuffers`) — Caller-owned demand planning sidecars.
- `finalized_demand` (`NucleationFinalizedDemandBuffers`) — Caller-owned demand planning/capacity staging finalized-demand sidecars.
- `diagnostics` (`NucleationDiagnosticBuffers`) — Caller-owned demand planning/capacity staging diagnostic sidecars.
- `exhaustion_controls` (`NucleationExhaustionControls`) — Exact-Boolean exhaustion resolution resampling and scaling controls.
- `exhaustion_buffers` (`NucleationExhaustionBuffers`) — Caller-owned exhaustion resolution workspace and diagnostics.
- `temperature` (`Any | None`) (default: `None`) — Positive scalar [K] or same-device ``wp.float64 (B,)``.
- `saturation` (`Any | None`) (default: `None`) — Configured same-device ``wp.float64 (B, S)`` ratios.
- `environment` (`Any | None`) (default: `None`) — Optional same-device owner of temperature and saturation.

**Returns**

- (`_PreparedNucleationCall`) — Private prepared call with pinned storage and allocated work arrays.

**Raises**

- `TypeError` — If a required scalar has an unsupported type.
- `ValueError` — If fixed schemas, identities, controls, or sidecars are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3437-L3655)

## particula.gpu.kernels.nucleation._promote_prepared_invalid_lane

*function*

```python
def _promote_prepared_invalid_lane(status: wp.array(dtype=wp.int32), lane: wp.int32, pipeline_invalid: wp.array(dtype=wp.int32)) -> None
```

Promote one invalid-only status lane into the pipeline gate.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3316-L3324)

## particula.gpu.kernels.nucleation._real

*function*

```python
def _real(value: object, name: str, *, positive: bool = False) -> float
```

Normalize a finite, non-Boolean real scalar.

**Parameters**

- `value` (`object`) — Scalar to validate.
- `name` (`str`) — Value name used in validation errors.
- `positive` (`bool`) (default: `False`) — Require a strictly positive rather than nonnegative value.

**Returns**

- (`float`) — Validated Python floating-point scalar.

**Raises**

- `TypeError` — If ``value`` is not a supported real scalar.
- `ValueError` — If ``value`` is nonfinite or outside the required domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L95-L126)

## particula.gpu.kernels.nucleation._reset_prepared_nucleation_status

*function*

```python
def _reset_prepared_nucleation_status(p1_status: wp.array(dtype=wp.int32), pipeline_invalid: wp.array(dtype=wp.int32), p2_status: wp.array(dtype=wp.int32), p3_status: wp.array(dtype=wp.int32), molecule_status: wp.array(dtype=wp.int32), p4_status: wp.array(dtype=wp.int32), p4_final_status: wp.array(dtype=wp.int32), p5_status: wp.array(dtype=wp.int32), release: wp.array(dtype=wp.int32), scaling: wp.array(dtype=wp.int32)) -> None
```

Reset preparation-owned status and policy work on device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3157-L3186)

## particula.gpu.kernels.nucleation._revalidate_p4_particle_state

*function*

```python
def _revalidate_p4_particle_state(preflight: _NucleationPreflight) -> None
```

Revalidate mutable input validation particle storage before exhaustion
resolution writes or primitives.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2618-L2652)

## particula.gpu.kernels.nucleation._revalidate_p5_storage

*function*

```python
def _revalidate_p5_storage(preflight: _NucleationPreflight) -> None
```

Reject rebound input validation storage before commit's device handoff
validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3034-L3080)

## particula.gpu.kernels.nucleation._saturation_status

*function*

```python
def _saturation_status(saturation: wp.array2d(dtype=wp.float64), precursor_index: wp.int32, lower: wp.float64, upper: wp.float64, status: wp.array(dtype=wp.int32), gated: wp.array(dtype=wp.int32)) -> None
```

Count saturation lower gates and upper-domain failures by box.

**Parameters**

- `saturation` (`wp.array2d(dtype=wp.float64)`) — Dimensionless saturation ratios shaped ``(B, S)``.
- `precursor_index` (`wp.int32`) — Selected precursor species index.
- `lower` (`wp.float64`) — Inclusive saturation lower endpoint.
- `upper` (`wp.float64`) — Inclusive saturation upper endpoint.
- `status` (`wp.array(dtype=wp.int32)`) — Device counters for invalid and lower-gated values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L844-L868)

## particula.gpu.kernels.nucleation._scan

*function*

```python
def _scan(values: Any, name: str, *, positive: bool = False) -> None
```

Perform a read-only finite-domain scan with scalar status readback.

**Parameters**

- `values` (`Any`) — One-, two-, or three-dimensional ``wp.float64`` array.
- `name` (`str`) — Input name used in validation errors.
- `positive` (`bool`) (default: `False`) — Require positive rather than nonnegative values.

**Raises**

- `ValueError` — If any value is nonfinite or outside the requested domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L883-L904)

## particula.gpu.kernels.nucleation._scan_binary

*function*

```python
def _scan_binary(values: Any) -> None
```

Validate binary partitioning with a device-resident read-only scan.

**Parameters**

- `values` (`Any`) — ``wp.int32`` partitioning flags shaped ``(B, S)``.

**Raises**

- `ValueError` — If any partitioning value is not zero or one.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L968-L985)

## particula.gpu.kernels.nucleation._scan_finite_2d

*function*

```python
def _scan_finite_2d(values: Any, name: str) -> None
```

Perform a read-only scan that permits finite signed values.

**Parameters**

- `values` (`Any`) — Two-dimensional ``wp.float64`` array to scan.
- `name` (`str`) — Input name used in validation errors.

**Raises**

- `ValueError` — If any value is nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L923-L941)

## particula.gpu.kernels.nucleation._scan_interval

*function*

```python
def _scan_interval(values: Any, name: str, lower: float, upper: float) -> None
```

Perform a read-only inclusive-interval scan with status readback.

**Parameters**

- `values` (`Any`) — One- or two-dimensional ``wp.float64`` array to scan.
- `name` (`str`) — Input name used in validation errors.
- `lower` (`float`) — Inclusive lower endpoint.
- `upper` (`float`) — Inclusive upper endpoint.

**Raises**

- `ValueError` — If any value falls outside the configured interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L944-L965)

## particula.gpu.kernels.nucleation._select_p4_policy

*function*

```python
def _select_p4_policy(demand: wp.array(dtype=wp.float64), counts: wp.array(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32), releasable: wp.array(dtype=wp.int32), requested: wp.array(dtype=wp.float64), minimum_volume: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), resampling_enabled: wp.int32, scaling_enabled: wp.int32, release: wp.array(dtype=wp.int32), scaling: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Select all-or-nothing resampling before scaling for one box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2375-L2416)

## particula.gpu.kernels.nucleation._stage_nucleation_slots

*function*

```python
def _stage_nucleation_slots(preflight: _NucleationPreflight, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers) -> None
```

Privately stage capacity staging event counts and selectable free-slot
    metadata.

Capacity staging consumes demand planning's inventory-admitted demand [#/m³]
and validated particle
volume [m³] to calculate full ``int32`` provisional event counts. It
delegates active/free classification and ascending free-slot ordering to
Fixed-slot management, retains counts beyond free capacity, and writes only
the supplied
count and slot-diagnostic sidecars. It does not activate slots, resolve
exhaustion, resize storage, or mutate particles or gas.

Capacity staging shares one private scalar status readback between its
conversion and the
fixed-slot-state classification before either output writer launches.
After a successful fixed-slot management or capacity staging writer launch,
callers must synchronize
before consuming outputs; asynchronous writer failures do not promise
rollback.

**Parameters**

- `preflight` (`_NucleationPreflight`) — Exact private input validation result reused without repeated validation.
- `finalized_demand` (`NucleationFinalizedDemandBuffers`) — Exact demand planning/capacity staging sidecar record containing admitted demand and receiving full provisional counts.
- `diagnostics` (`NucleationDiagnosticBuffers`) — Exact demand planning/capacity staging diagnostic record receiving fixed-slot management layouts and Capacity staging's capacity-limited selectable prefix.

**Raises**

- `ValueError` — If private handoff records are invalid, slots are invalid,
or event conversion does not produce finite, nonnegative, integral
``int32`` counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2206-L2306)

## particula.gpu.kernels.nucleation._validate_p4_buffers

*function*

```python
def _validate_p4_buffers(preflight: _NucleationPreflight, buffers: NucleationExhaustionBuffers) -> tuple[Any, ...]
```

Validate exhaustion resolution and nested slot exhaustion storage before
any exhaustion resolution writer runs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2545-L2601)

## particula.gpu.kernels.nucleation._validate_p4_final

*function*

```python
def _validate_p4_final(demand: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), free_counts: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Validate final exhaustion resolution counts before finalized diagnostic
writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2419-L2443)

## particula.gpu.kernels.nucleation._validate_p4_handoff

*function*

```python
def _validate_p4_handoff(accepted_demand: wp.array(dtype=wp.float64), accepted_counts: wp.array(dtype=wp.int32), volume: wp.array(dtype=wp.float64), categories: wp.array2d(dtype=wp.int32), free_indices: wp.array2d(dtype=wp.int32), active_counts: wp.array(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32), selected_indices: wp.array2d(dtype=wp.int32), releasable: wp.array(dtype=wp.int32), requested: wp.array(dtype=wp.float64), minimum: wp.array(dtype=wp.float64), minimum_volume: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Validate immutable demand planning/capacity staging handoffs and
exhaustion resolution policy inputs per box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2309-L2372)

## particula.gpu.kernels.nucleation._validate_p5_handoff

*function*

```python
def _validate_p5_handoff(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), gas_concentration: wp.array2d(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), partitioning: wp.array2d(dtype=wp.int32), molecule_counts: wp.array(dtype=wp.int32), final_demand: wp.array(dtype=wp.float64), final_counts: wp.array(dtype=wp.int32), selected_indices: wp.array2d(dtype=wp.int32), status: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Validate exhaustion resolution's final handoff before the sole commit
writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2955-L3031)

## particula.gpu.kernels.nucleation._validate_participating_molecule_counts

*function*

```python
def _validate_participating_molecule_counts(partitioning: wp.array2d(dtype=wp.int32), molecule_counts: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Reject positive molecule counts for non-participating gas lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2907-L2920)

## particula.gpu.kernels.nucleation._validate_prepared_nucleation_p1

*function*

```python
def _validate_prepared_nucleation_p1(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), gas_concentration: wp.array2d(dtype=wp.float64), partitioning: wp.array2d(dtype=wp.int32), temperature: wp.array(dtype=wp.float64), saturation: wp.array2d(dtype=wp.float64), n_boxes: wp.int32, precursor_index: wp.int32, temperature_lower: wp.float64, temperature_upper: wp.float64, coefficient: wp.float64, survival_factor: wp.float64, has_saturation: wp.int32, saturation_upper: wp.float64, precursor_lower: wp.float64, precursor_upper: wp.float64, status: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Validate all live input validation payloads and retain ordered error
lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L3189-L3302)

## particula.gpu.kernels.nucleation._validate_public_molecule_eligibility

*function*

```python
def _validate_public_molecule_eligibility(preflight: _NucleationPreflight) -> None
```

Validate commit gas-removal eligibility before exhaustion resolution can
mutate particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2923-L2952)

## particula.gpu.kernels.nucleation._validate_public_p4_inputs

*function*

```python
def _validate_public_p4_inputs(preflight: _NucleationPreflight, controls: NucleationExhaustionControls, buffers: NucleationExhaustionBuffers) -> None
```

Validate public exhaustion resolution controls and storage before any
no-work return.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2604-L2615)

## particula.gpu.kernels.nucleation._validate_sidecars

*function*

```python
def _validate_sidecars(scratch: NucleationScratchBuffers | None, finalized: NucleationFinalizedDemandBuffers | None, diagnostics: NucleationDiagnosticBuffers | None, n_boxes: int, n_particles: int, n_species: int, device: Any) -> tuple[Any, ...]
```

Validate supplied sidecar schemas without reading stale contents.

**Parameters**

- `scratch` (`NucleationScratchBuffers | None`) — Optional caller-owned planning sidecars.
- `finalized` (`NucleationFinalizedDemandBuffers | None`) — Optional caller-owned finalized-demand sidecars.
- `diagnostics` (`NucleationDiagnosticBuffers | None`) — Optional caller-owned diagnostic sidecars.
- `n_boxes` (`int`) — Fixed box count ``B``.
- `n_particles` (`int`) — Fixed particle capacity ``N``.
- `n_species` (`int`) — Fixed species count ``S``.
- `device` (`Any`) — Required shared Warp device.

**Returns**

- (`tuple[Any, ...]`) — Validated sidecar arrays retaining caller identity.

**Raises**

- `ValueError` — If a supplied record or sidecar schema is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L988-L1082)

## particula.gpu.kernels.nucleation._validate_staged_nucleation_sidecars

*function*

```python
def _validate_staged_nucleation_sidecars(preflight: _NucleationPreflight, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers) -> None
```

Validate exact capacity staging handoff records before staging writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2117-L2145)

## particula.gpu.kernels.nucleation._write_current_p4_diagnostics

*function*

```python
def _write_current_p4_diagnostics(categories: wp.array2d(dtype=wp.int32), free_indices: wp.array2d(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32)) -> None
```

Write private ascending current free-slot diagnostics for exhaustion
resolution.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2473-L2490)

## particula.gpu.kernels.nucleation._write_p4_final

*function*

```python
def _write_p4_final(demand: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), free_indices: wp.array2d(dtype=wp.int32), final_demand: wp.array(dtype=wp.float64), final_counts: wp.array(dtype=wp.int32), final_indices: wp.array2d(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Commit finalized exhaustion resolution demand, counts, and ascending
free-slot prefixes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2446-L2470)

## particula.gpu.kernels.nucleation._write_p4_policy_diagnostics

*function*

```python
def _write_p4_policy_diagnostics(counts: wp.array(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32), scaling: wp.array(dtype=wp.int32), required_release: wp.array(dtype=wp.int32), scaling_required: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Commit validated private policy selection to exhaustion resolution
diagnostic sidecars.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2507-L2523)

## particula.gpu.kernels.nucleation._write_staged_nucleation_diagnostics

*function*

```python
def _write_staged_nucleation_diagnostics(preflight: _NucleationPreflight, categories: Any, diagnostics: NucleationDiagnosticBuffers) -> None
```

Write fixed-slot management classification outputs to the supplied
diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2174-L2203)

## particula.gpu.kernels.nucleation._write_unscaled_p4_resolution

*function*

```python
def _write_unscaled_p4_resolution(scaling: wp.array(dtype=wp.int32), resolved_scale: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Set the exhaustion resolution resolved scale to one for unselected policy
rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L2493-L2504)

## particula.gpu.kernels.nucleation.nucleation_step_gpu

*function*

```python
def nucleation_step_gpu(particles: Any, gas: Any, config: NucleationConfig, time_step: Any, *, scratch: NucleationScratchBuffers, finalized_demand: NucleationFinalizedDemandBuffers, diagnostics: NucleationDiagnosticBuffers, exhaustion_controls: NucleationExhaustionControls, exhaustion_buffers: NucleationExhaustionBuffers, temperature: Any | None = None, saturation: Any | None = None, environment: Any | None = None, species_binding: NucleationSpeciesBinding | None = None) -> tuple[Any, Any]
```

Execute one fixed-capacity direct-Warp nucleation step.

The supported public boundary composes private ``prepare → enqueue →
observe`` helpers.  Prepared records pin references rather than payload
snapshots, so the direct executor consumes current device data.  Private
records and helpers remain concrete-module-only.

**Parameters**

- `particles` (`Any`) — Caller-owned fixed-capacity Warp particle container.
- `gas` (`Any`) — Caller-owned same-device Warp gas container.
- `config` (`NucleationConfig`) — Immutable scalar nucleation controls.
- `time_step` (`Any`) — Finite nonnegative step duration [s].
- `scratch` (`NucleationScratchBuffers`) — Caller-owned demand planning sidecars.
- `finalized_demand` (`NucleationFinalizedDemandBuffers`) — Caller-owned demand planning/capacity staging finalized-demand sidecars.
- `diagnostics` (`NucleationDiagnosticBuffers`) — Caller-owned demand planning/capacity staging diagnostic sidecars.
- `exhaustion_controls` (`NucleationExhaustionControls`) — Concrete exhaustion resolution resampling and scaling controls.
- `exhaustion_buffers` (`NucleationExhaustionBuffers`) — Caller-owned exhaustion resolution workspace and diagnostics.
- `temperature` (`Any | None`) (default: `None`) — Positive scalar [K] or same-device ``wp.float64 (B,)``.
- `saturation` (`Any | None`) (default: `None`) — Configured same-device ``wp.float64 (B, S)`` ratios.
- `environment` (`Any | None`) (default: `None`) — Optional same-device owner of temperature and saturation.

**Returns**

- (`tuple[Any, Any]`) — The identical caller-owned ``(particles, gas)`` containers.

**Raises**

- `TypeError` — If a required scalar or configuration has an unsupported
type.
- `ValueError` — If stage validation, ownership, physical
state, or a
finalized handoff is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/nucleation.py#L4205-L4272)
