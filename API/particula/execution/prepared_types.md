# particula.execution.prepared_types

*module*

CPU-only result types for prepared simulation observations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L1-L1)

## particula.execution.prepared_types.SimulationCheckpoint

*class* · *dataclass*

```python
class SimulationCheckpoint
```

In-memory, exact-device continuation of a prepared simulation.

Canonical resident bytes preserve primary state, acquired sidecars and
random words. The frozen process plans and timestep identify the compatible
runnable sequence. Schema v2 adds absolute initial time, field descriptors
and bytes, declared adapter-state bytes, and the device clock. Python
implementations are supplied again through the registry at restart.
The explicit v1 reader accepts built-in-only state with a zero time origin.
Auxiliary bytes preserve last-step latent energy.
Resident CPU inspection arrays are detached but are not restart authority.
This is not a portable disk format or a device-migration interface.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L20-L46)

### particula.execution.prepared_types.SimulationCheckpoint.auxiliary

*attribute* · *class attribute* · *instance attribute*

```python
auxiliary: tuple[CheckpointPayload, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L39-L39)

### particula.execution.prepared_types.SimulationCheckpoint.clock

*attribute* · *class attribute* · *instance attribute*

```python
clock: CheckpointPayload | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L44-L44)

### particula.execution.prepared_types.SimulationCheckpoint.execution_mode

*attribute* · *class attribute* · *instance attribute*

```python
execution_mode: str = 'resident'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L45-L45)

### particula.execution.prepared_types.SimulationCheckpoint.extension_state

*attribute* · *class attribute* · *instance attribute*

```python
extension_state: tuple[CheckpointPayload, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L43-L43)

### particula.execution.prepared_types.SimulationCheckpoint.field_payloads

*attribute* · *class attribute* · *instance attribute*

```python
field_payloads: tuple[CheckpointPayload, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L42-L42)

### particula.execution.prepared_types.SimulationCheckpoint.field_specs

*attribute* · *class attribute* · *instance attribute*

```python
field_specs: tuple[FieldSpec, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L41-L41)

### particula.execution.prepared_types.SimulationCheckpoint.host_rng

*attribute* · *class attribute* · *instance attribute*

```python
host_rng: bytes | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L46-L46)

### particula.execution.prepared_types.SimulationCheckpoint.initial_time

*attribute* · *class attribute* · *instance attribute*

```python
initial_time: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L40-L40)

### particula.execution.prepared_types.SimulationCheckpoint.processes

*attribute* · *instance attribute*

```python
processes: tuple[_ProcessPlan | GPUProcessPlan, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L37-L37)

### particula.execution.prepared_types.SimulationCheckpoint.resident

*attribute* · *instance attribute*

```python
resident: ResidentCheckpoint
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L36-L36)

### particula.execution.prepared_types.SimulationCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L35-L35)

### particula.execution.prepared_types.SimulationCheckpoint.time_step

*attribute* · *instance attribute*

```python
time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L38-L38)

### particula.execution.prepared_types.SimulationCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, resident: ResidentCheckpoint, processes: tuple[_ProcessPlan | GPUProcessPlan, ...], time_step: float, auxiliary: tuple[CheckpointPayload, ...] = (), initial_time: float = 0.0, field_specs: tuple[FieldSpec, ...] = (), field_payloads: tuple[CheckpointPayload, ...] = (), extension_state: tuple[CheckpointPayload, ...] = (), clock: CheckpointPayload | None = None, execution_mode: str = 'resident', host_rng: bytes | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L0-L0)

## particula.execution.prepared_types.SimulationDiagnostics

*class* · *dataclass*

```python
class SimulationDiagnostics
```

Detached per-box diagnostics; no particle-slot arrays are downloaded.

Number and charge concentrations are in m^-3 and elementary charges/m³.
Particle/gas mass concentrations are kg/m³ in their independent species
orders. Active slots count representatives, while number concentration
includes multiplicity. Latent energy, when configured, is the last step's
signed per-box/full-gas energy [J]. Temperature is K, pressure Pa,
and volume m³.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L49-L73)

### particula.execution.prepared_types.SimulationDiagnostics.active_slots

*attribute* · *instance attribute*

```python
active_slots: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L64-L64)

### particula.execution.prepared_types.SimulationDiagnostics.charge_concentration

*attribute* · *instance attribute*

```python
charge_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L66-L66)

### particula.execution.prepared_types.SimulationDiagnostics.completed_steps

*attribute* · *instance attribute*

```python
completed_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L61-L61)

### particula.execution.prepared_types.SimulationDiagnostics.gas_concentration

*attribute* · *instance attribute*

```python
gas_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L68-L68)

### particula.execution.prepared_types.SimulationDiagnostics.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L63-L63)

### particula.execution.prepared_types.SimulationDiagnostics.latent_heat_energy

*attribute* · *instance attribute*

```python
latent_heat_energy: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L73-L73)

### particula.execution.prepared_types.SimulationDiagnostics.number_concentration

*attribute* · *instance attribute*

```python
number_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L65-L65)

### particula.execution.prepared_types.SimulationDiagnostics.particle_mass_concentration

*attribute* · *instance attribute*

```python
particle_mass_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L67-L67)

### particula.execution.prepared_types.SimulationDiagnostics.pressure

*attribute* · *instance attribute*

```python
pressure: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L70-L70)

### particula.execution.prepared_types.SimulationDiagnostics.saturation_ratio

*attribute* · *instance attribute*

```python
saturation_ratio: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L71-L71)

### particula.execution.prepared_types.SimulationDiagnostics.simulated_time

*attribute* · *instance attribute*

```python
simulated_time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L62-L62)

### particula.execution.prepared_types.SimulationDiagnostics.temperature

*attribute* · *instance attribute*

```python
temperature: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L69-L69)

### particula.execution.prepared_types.SimulationDiagnostics.volume

*attribute* · *instance attribute*

```python
volume: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L72-L72)

### particula.execution.prepared_types.SimulationDiagnostics.__init__

*method*

```python
def __init__(completed_steps: int, simulated_time: float, gas_names: tuple[str, ...], active_slots: np.ndarray, number_concentration: np.ndarray, charge_concentration: np.ndarray, particle_mass_concentration: np.ndarray, gas_concentration: np.ndarray, temperature: np.ndarray, pressure: np.ndarray, saturation_ratio: np.ndarray, volume: np.ndarray, latent_heat_energy: np.ndarray | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L0-L0)
