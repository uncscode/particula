# particula.execution.diagnostics

*module*

Write closed resident diagnostics into caller-owned Warp arrays.

This concrete direct-import-only module has no callback registration or package
export. Its shared validator performs read-only metadata checks without
constructing an executor or mutating resident state. Registrations execute in
this fixed order: gas-concentration snapshot,
saturation-ratio snapshot, total species mass, particle-number concentration,
latent heat energy, and conservation residual. Matrix operations use ``(B, S)``
``wp.float64`` arrays; particle number uses a ``(B,)`` ``wp.float64`` array.

For native resolved counts, total mass adds ``Σp(m * N)`` to ``V * gas``
using an explicit complete material-to-gas map. PMF particle density
use ``V * Σp(m * c)``. Native mass diagnostics reject unmapped particle-only
lanes instead of dropping their inventory. Matrix lanes follow full gas order.
Resolved number concentration is ``Σp(N) / V``; density carriers use ``Σp(c)``.
Latent energy copies signed
whole-call inventory-limited energy in J. The residual is
``total_mass - baseline_total_mass - source_ledger + sink_ledger`` in kg;
source and sink ledgers are nonnegative accumulated extensive-mass inputs.
Execution validates caller-owned same-device bindings without host readback,
synchronization, transfer, allocation, or physics mutation. Empty matrix
operations are write-free for ``B == 0`` or ``S == 0``; particle number is
write-free only for ``B == 0``. Prepared setup binds the closed plan; its
observation-free enqueue repeats no host/setup metadata validation and performs
no allocation, host readback, transfer, synchronization, lookup, or rebinding.
Retained diagnostic writers still perform device-side status and physical-state
Validation. Setup rejection is pre-writer; rollback is not promised after a
diagnostic writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L1-L1)

## particula.execution.diagnostics.PreparedResidentDiagnostics

*class* · *dataclass*

```python
class PreparedResidentDiagnostics
```

Bind validated diagnostic sources and registrations for enqueue only.

This concrete-only record retains every source, output, and operation by
identity. Its observation-free enqueue dispatches those retained writers
only; it does not revalidate or rediscover the plan.

**Attributes**

- `plan` (`ResidentDiagnosticsPlan`) — Exact diagnostics plan validated during setup.
- `particle_masses` (`object`) — Bound resident particle mass array.
- `particle_concentration` (`object`) — Bound resident particle concentration array.
- `particle_volume` (`object`) — Bound resident particle volume array.
- `gas_concentration` (`object`) — Bound resident gas concentration array.
- `saturation_ratio` (`object`) — Bound resident saturation-ratio array.
- `device` (`object`) — Device shared by bound arrays and outputs.
- `dimensions` (`object`) — Exact resident dimensions used for empty-schema checks.
- `registrations` (`tuple[ResidentDiagnosticRegistration, ...]`) — Canonically ordered diagnostic registrations.
- `outputs` (`tuple[object, ...]`) — Registration outputs in canonical launch order.
- `energy_transfers` (`tuple[object | None, ...]`) — Latent-energy inputs aligned with registrations.
- `baseline_total_masses` (`tuple[object | None, ...]`) — Residual baselines aligned with registrations.
- `source_ledgers` (`tuple[object | None, ...]`) — Residual source ledgers aligned with registrations.
- `sink_ledgers` (`tuple[object | None, ...]`) — Residual sink ledgers aligned with registrations.
- `total_mass_output` (`object | None`) — Bound total-mass output used by the residual writer.
- `resolved_counts` (`bool`) — Whether particle population stores raw counts.
- `particle_mass_pairs` (`tuple[tuple[int, int], ...]`) — Validated gas-to-particle inventory lane pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L184-L228)

### particula.execution.diagnostics.PreparedResidentDiagnostics.baseline_total_masses

*attribute* · *instance attribute*

```python
baseline_total_masses: tuple[object | None, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L223-L223)

### particula.execution.diagnostics.PreparedResidentDiagnostics.device

*attribute* · *instance attribute*

```python
device: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L218-L218)

### particula.execution.diagnostics.PreparedResidentDiagnostics.dimensions

*attribute* · *instance attribute*

```python
dimensions: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L219-L219)

### particula.execution.diagnostics.PreparedResidentDiagnostics.energy_transfers

*attribute* · *instance attribute*

```python
energy_transfers: tuple[object | None, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L222-L222)

### particula.execution.diagnostics.PreparedResidentDiagnostics.gas_concentration

*attribute* · *instance attribute*

```python
gas_concentration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L216-L216)

### particula.execution.diagnostics.PreparedResidentDiagnostics.outputs

*attribute* · *instance attribute*

```python
outputs: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L221-L221)

### particula.execution.diagnostics.PreparedResidentDiagnostics.particle_concentration

*attribute* · *instance attribute*

```python
particle_concentration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L214-L214)

### particula.execution.diagnostics.PreparedResidentDiagnostics.particle_mass_pairs

*attribute* · *class attribute* · *instance attribute*

```python
particle_mass_pairs: tuple[tuple[int, int], ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L228-L228)

### particula.execution.diagnostics.PreparedResidentDiagnostics.particle_masses

*attribute* · *instance attribute*

```python
particle_masses: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L213-L213)

### particula.execution.diagnostics.PreparedResidentDiagnostics.particle_volume

*attribute* · *instance attribute*

```python
particle_volume: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L215-L215)

### particula.execution.diagnostics.PreparedResidentDiagnostics.plan

*attribute* · *instance attribute*

```python
plan: ResidentDiagnosticsPlan
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L212-L212)

### particula.execution.diagnostics.PreparedResidentDiagnostics.registrations

*attribute* · *instance attribute*

```python
registrations: tuple[ResidentDiagnosticRegistration, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L220-L220)

### particula.execution.diagnostics.PreparedResidentDiagnostics.resolved_counts

*attribute* · *class attribute* · *instance attribute*

```python
resolved_counts: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L227-L227)

### particula.execution.diagnostics.PreparedResidentDiagnostics.saturation_ratio

*attribute* · *instance attribute*

```python
saturation_ratio: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L217-L217)

### particula.execution.diagnostics.PreparedResidentDiagnostics.sink_ledgers

*attribute* · *instance attribute*

```python
sink_ledgers: tuple[object | None, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L225-L225)

### particula.execution.diagnostics.PreparedResidentDiagnostics.source_ledgers

*attribute* · *instance attribute*

```python
source_ledgers: tuple[object | None, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L224-L224)

### particula.execution.diagnostics.PreparedResidentDiagnostics.total_mass_output

*attribute* · *instance attribute*

```python
total_mass_output: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L226-L226)

### particula.execution.diagnostics.PreparedResidentDiagnostics.__init__

*method*

```python
def __init__(plan: ResidentDiagnosticsPlan, particle_masses: object, particle_concentration: object, particle_volume: object, gas_concentration: object, saturation_ratio: object, device: object, dimensions: object, registrations: tuple[ResidentDiagnosticRegistration, ...], outputs: tuple[object, ...], energy_transfers: tuple[object | None, ...], baseline_total_masses: tuple[object | None, ...], source_ledgers: tuple[object | None, ...], sink_ledgers: tuple[object | None, ...], total_mass_output: object | None, resolved_counts: bool = False, particle_mass_pairs: tuple[tuple[int, int], ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L0-L0)

## particula.execution.diagnostics.ResidentDiagnosticOperation

*class*

```python
class ResidentDiagnosticOperation(str, Enum)
```

Bases: `str`, `Enum`

Enumerate the closed resident diagnostic operations in launch order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L55-L63)

### particula.execution.diagnostics.ResidentDiagnosticOperation.CONSERVATION_RESIDUAL

*attribute* · *class attribute* · *instance attribute*

```python
CONSERVATION_RESIDUAL = 'conservation_residual'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L63-L63)

### particula.execution.diagnostics.ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT

*attribute* · *class attribute* · *instance attribute*

```python
GAS_CONCENTRATION_SNAPSHOT = 'gas_concentration_snapshot'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L58-L58)

### particula.execution.diagnostics.ResidentDiagnosticOperation.LATENT_HEAT_ENERGY

*attribute* · *class attribute* · *instance attribute*

```python
LATENT_HEAT_ENERGY = 'latent_heat_energy'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L62-L62)

### particula.execution.diagnostics.ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION

*attribute* · *class attribute* · *instance attribute*

```python
PARTICLE_NUMBER_CONCENTRATION = 'particle_number_concentration'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L61-L61)

### particula.execution.diagnostics.ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT

*attribute* · *class attribute* · *instance attribute*

```python
SATURATION_RATIO_SNAPSHOT = 'saturation_ratio_snapshot'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L59-L59)

### particula.execution.diagnostics.ResidentDiagnosticOperation.TOTAL_SPECIES_MASS

*attribute* · *class attribute* · *instance attribute*

```python
TOTAL_SPECIES_MASS = 'total_species_mass'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L60-L60)

## particula.execution.diagnostics.ResidentDiagnosticRegistration

*class* · *dataclass*

```python
class ResidentDiagnosticRegistration
```

Bind one closed diagnostic operation to caller-owned Warp arrays.

**Attributes**

- `operation` (`ResidentDiagnosticOperation`) — Exact closed operation that selects the diagnostic reduction.
- `output` (`object`) — Caller-owned Warp ``float64`` output validated by the executor.
- `energy_transfer` (`object | None`) — Required ``(B, S)`` signed whole-call energy input in J for latent-energy output; forbidden otherwise.
- `baseline_total_mass` (`object | None`) — Required ``(B, S)`` extensive mass baseline in kg for residual output; forbidden otherwise.
- `source_ledger` (`object | None`) — Required nonnegative extensive source ledger for the residual in kg; forbidden otherwise.
- `sink_ledger` (`object | None`) — Required nonnegative extensive sink ledger for residual in kg; forbidden otherwise.
- `species_map` (`object | None`) — Native total-mass gas-to-particle map covering all material lanes; forbidden for other operations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L66-L127)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.baseline_total_mass

*attribute* · *class attribute* · *instance attribute*

```python
baseline_total_mass: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L88-L88)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.energy_transfer

*attribute* · *class attribute* · *instance attribute*

```python
energy_transfer: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L87-L87)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.operation

*attribute* · *instance attribute*

```python
operation: ResidentDiagnosticOperation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L85-L85)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.output

*attribute* · *instance attribute*

```python
output: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L86-L86)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.sink_ledger

*attribute* · *class attribute* · *instance attribute*

```python
sink_ledger: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L90-L90)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.source_ledger

*attribute* · *class attribute* · *instance attribute*

```python
source_ledger: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L89-L89)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.species_map

*attribute* · *class attribute* · *instance attribute*

```python
species_map: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L91-L91)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.__init__

*method*

```python
def __init__(operation: ResidentDiagnosticOperation, output: object, energy_transfer: object | None = None, baseline_total_mass: object | None = None, source_ledger: object | None = None, sink_ledger: object | None = None, species_map: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L0-L0)

### particula.execution.diagnostics.ResidentDiagnosticRegistration.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the exact closed diagnostic operation.

**Raises**

- `TypeError` — If ``operation`` is not an exact supported operation.
- `ValueError` — If required accounting inputs are missing or forbidden
accounting inputs are supplied for ``operation``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L93-L127)

## particula.execution.diagnostics.ResidentDiagnosticsExecutor

*class*

```python
class ResidentDiagnosticsExecutor
```

Execute an already-bound closed diagnostics plan without transfers.

Validation preserves caller ownership and rejects output or accounting-input
aliases with resident primaries, published sidecars, or diagnostic outputs.
Execution dispatches the six canonical registrations without host readback,
synchronization, transfer, allocation, or physics mutation. Matrix
registrations are write-free for empty ``(B, S)`` schemas; particle number
remains writable for ``(B, 0)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L346-L609)

### particula.execution.diagnostics.ResidentDiagnosticsExecutor._execute_validated

*method*

```python
def _execute_validated(plan: ResidentDiagnosticsPlan) -> None
```

Dispatch a plan already validated by its owning scheduler step.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L471-L609)

### particula.execution.diagnostics.ResidentDiagnosticsExecutor._validate

*method*

```python
def _validate(plan: ResidentDiagnosticsPlan) -> None
```

Validate plan provenance, closed operation order, and outputs.

**Parameters**

- `plan` (`ResidentDiagnosticsPlan`) — Exact diagnostics plan whose retained bindings are checked.

**Raises**

- `ValueError` — If lifecycle, graph, schedule, protocol, or output
metadata validation fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L422-L435)

### particula.execution.diagnostics.ResidentDiagnosticsExecutor._validate_graph_and_schedule

*method*

```python
def _validate_graph_and_schedule(plan: ResidentDiagnosticsPlan) -> None
```

Validate graph provenance, membership, and canonical order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L357-L400)

### particula.execution.diagnostics.ResidentDiagnosticsExecutor._validate_outputs

*method*

```python
def _validate_outputs(plan: ResidentDiagnosticsPlan) -> None
```

Validate caller-owned diagnostic outputs against resident state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L415-L420)

### particula.execution.diagnostics.ResidentDiagnosticsExecutor._validate_registrations

*method*

```python
def _validate_registrations(plan: ResidentDiagnosticsPlan) -> None
```

Validate diagnostic registration ordering and uniqueness.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L402-L413)

### particula.execution.diagnostics.ResidentDiagnosticsExecutor.execute

*method*

```python
def execute(plan: object) -> None
```

Validate and dispatch each registration in declared order.

Matrix schemas complete without their writer launch when ``B == 0`` or
``S == 0``. Particle number still launches for ``(B, 0)`` because its
``(B,)`` output exists. Successful launches are asynchronous; callers
synchronize before inspecting outputs on the host.

**Parameters**

- `plan` (`object`) — Exact plan selecting the sources and caller-owned outputs.

**Raises**

- `TypeError` — If ``plan`` is not an exact diagnostics plan.
- `ValueError` — If its bindings or output metadata are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L453-L469)

### particula.execution.diagnostics.ResidentDiagnosticsExecutor.validate

*method*

```python
def validate(plan: object) -> ResidentDiagnosticsPlan
```

Validate one exact diagnostics plan without dispatching a kernel.

**Parameters**

- `plan` (`object`) — Candidate concrete diagnostics plan.

**Returns**

- (`ResidentDiagnosticsPlan`) — The unchanged, exact validated plan.

**Raises**

- `TypeError` — If ``plan`` is not an exact diagnostics plan.
- `ValueError` — If the plan's graph, bindings, or registration protocol
is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L437-L451)

## particula.execution.diagnostics.ResidentDiagnosticsPlan

*class* · *dataclass*

```python
class ResidentDiagnosticsPlan
```

Bind ordered closed diagnostics to one resident graph and schedule.

**Attributes**

- `session` (`ResidentSession`) — Exact active resident session that owns diagnostic sources.
- `registry` (`object`) — Exact registry pinned to ``session``.
- `graph` (`ResolvedProcessGraph`) — Resolver-produced graph containing ``node`` by identity.
- `schedule` (`ResolvedTimestepSchedule`) — Matching resolved schedule that ends with ``node``.
- `node` (`ProcessNode`) — Canonical ``diagnostics`` process node.
- `registrations` (`tuple[ResidentDiagnosticRegistration, ...]`) — Exact canonical tuple of the six ordered closed operation and output bindings, validated by the executor.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L130-L181)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.graph

*attribute* · *instance attribute*

```python
graph: ResolvedProcessGraph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L146-L146)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.node

*attribute* · *instance attribute*

```python
node: ProcessNode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L148-L148)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.registrations

*attribute* · *instance attribute*

```python
registrations: tuple[ResidentDiagnosticRegistration, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L149-L149)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.registry

*attribute* · *instance attribute*

```python
registry: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L145-L145)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.schedule

*attribute* · *instance attribute*

```python
schedule: ResolvedTimestepSchedule
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L147-L147)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L144-L144)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.__init__

*method*

```python
def __init__(session: ResidentSession, registry: object, graph: ResolvedProcessGraph, schedule: ResolvedTimestepSchedule, node: ProcessNode, registrations: tuple[ResidentDiagnosticRegistration, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L0-L0)

### particula.execution.diagnostics.ResidentDiagnosticsPlan.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact types for the resident diagnostics binding.

Structural graph, lifecycle, and output validation is deferred to the
executor so plan construction does not inspect Warp-array metadata.

**Raises**

- `TypeError` — If a carrier or registration has an inexact type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L151-L181)

## particula.execution.diagnostics._accumulate_mapped_particle_mass

*function*

```python
def _accumulate_mapped_particle_mass(masses: Any, concentration: Any, volume: Any, output: Any, particle_species: int, gas_species: int, resolved: int) -> None
```

Accumulate one explicit material-to-gas extensive inventory lane.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L288-L308)

## particula.execution.diagnostics._accumulate_particle_number

*function*

```python
def _accumulate_particle_number(concentration: Any, output: Any) -> None
```

Accumulate one particle concentration lane in parallel.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L254-L258)

## particula.execution.diagnostics._accumulate_resolved_number

*function*

```python
def _accumulate_resolved_number(concentration: Any, volume: Any, output: Any) -> None
```

Reduce represented counts to per-volume number concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L279-L285)

## particula.execution.diagnostics._clear_particle_number

*function*

```python
def _clear_particle_number(output: Any) -> None
```

Clear one particle-number output lane before staged accumulation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L247-L251)

## particula.execution.diagnostics._conservation_residual

*function*

```python
def _conservation_residual(total_mass: Any, baseline: Any, source: Any, sink: Any, output: Any) -> None
```

Write the ledger-aware residual from the already-reduced total mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L261-L276)

## particula.execution.diagnostics._copy_snapshot

*function*

```python
def _copy_snapshot(source: Any, output: Any) -> None
```

Copy one resident diagnostic matrix element to its output.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L231-L235)

## particula.execution.diagnostics._enqueue_prepared_resident_diagnostics

*function*

```python
def _enqueue_prepared_resident_diagnostics(prepared: PreparedResidentDiagnostics) -> None
```

Dispatch only registrations retained by prepared diagnostics setup.

This private retained-reference seam performs no validation, allocation,
readback, transfer, synchronization, lookup, or rebinding. Empty matrix
schemas remain write-free; rollback is not promised after a writer launches.

**Parameters**

- `prepared` (`PreparedResidentDiagnostics`) — Previously validated immutable diagnostics binding.

**Raises**

- `ValueError` — If residual dispatch lacks its bound total-mass output.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L706-L827)

## particula.execution.diagnostics._initialize_total_species_mass

*function*

```python
def _initialize_total_species_mass(gas_concentration: Any, volume: Any, output: Any) -> None
```

Initialize each extensive-mass lane from resident gas.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L238-L244)

## particula.execution.diagnostics._mass_pairs

*function*

```python
def _mass_pairs(plan: ResidentDiagnosticsPlan) -> tuple[tuple[int, int], ...]
```

Validate total-mass provenance without device work or inferred lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L311-L343)

## particula.execution.diagnostics.setup_prepared_resident_diagnostics

*function*

```python
def setup_prepared_resident_diagnostics(prepared_timestep: object, plan: object) -> PreparedResidentDiagnostics
```

Validate a declaration diagnostics attachment and bind enqueue
    identities.

**Parameters**

- `prepared_timestep` (`object`) — Exact declaration timestep retaining ``plan`` by identity.
- `plan` (`object`) — Exact resident diagnostics plan to validate and bind.

**Returns**

- (`PreparedResidentDiagnostics`) — Immutable binding that dispatches the canonical registrations without
- (`PreparedResidentDiagnostics`) — repeating setup validation.

**Raises**

- `TypeError` — If either carrier has an unsupported exact type.
- `ValueError` — If declaration identities, plan metadata, registrations, or
primary
arrays are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L612-L703)

## particula.execution.diagnostics.validate_resident_diagnostics_plan

*function*

```python
def validate_resident_diagnostics_plan(plan: object) -> ResidentDiagnosticsPlan
```

Validate an exact diagnostics plan without constructing an executor.

This read-only shared validation seam performs no diagnostic dispatch,
resource acquisition, payload inspection, transfer, synchronization, or
resident-state mutation.

**Parameters**

- `plan` (`object`) — Candidate concrete diagnostics plan.

**Returns**

- (`ResidentDiagnosticsPlan`) — The unchanged, exact validated plan.

**Raises**

- `TypeError` — If ``plan`` is not an exact diagnostics plan.
- `ValueError` — If its binding, graph, or registration metadata is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/diagnostics.py#L830-L899)
