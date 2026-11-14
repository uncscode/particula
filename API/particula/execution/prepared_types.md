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
runnable sequence. Auxiliary bytes preserve last-step latent energy.
Resident CPU inspection arrays are detached but are not restart authority.
This is not a portable disk format or a device-migration interface.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L18-L33)

### particula.execution.prepared_types.SimulationCheckpoint.auxiliary

*attribute* · *class attribute* · *instance attribute*

```python
auxiliary: tuple[CheckpointPayload, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L33-L33)

### particula.execution.prepared_types.SimulationCheckpoint.processes

*attribute* · *instance attribute*

```python
processes: tuple[_ProcessPlan, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L31-L31)

### particula.execution.prepared_types.SimulationCheckpoint.resident

*attribute* · *instance attribute*

```python
resident: ResidentCheckpoint
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L30-L30)

### particula.execution.prepared_types.SimulationCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L29-L29)

### particula.execution.prepared_types.SimulationCheckpoint.time_step

*attribute* · *instance attribute*

```python
time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L32-L32)

### particula.execution.prepared_types.SimulationCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, resident: ResidentCheckpoint, processes: tuple[_ProcessPlan, ...], time_step: float, auxiliary: tuple[CheckpointPayload, ...] = ()) -> None
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

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L36-L60)

### particula.execution.prepared_types.SimulationDiagnostics.active_slots

*attribute* · *instance attribute*

```python
active_slots: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L51-L51)

### particula.execution.prepared_types.SimulationDiagnostics.charge_concentration

*attribute* · *instance attribute*

```python
charge_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L53-L53)

### particula.execution.prepared_types.SimulationDiagnostics.completed_steps

*attribute* · *instance attribute*

```python
completed_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L48-L48)

### particula.execution.prepared_types.SimulationDiagnostics.gas_concentration

*attribute* · *instance attribute*

```python
gas_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L55-L55)

### particula.execution.prepared_types.SimulationDiagnostics.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L50-L50)

### particula.execution.prepared_types.SimulationDiagnostics.latent_heat_energy

*attribute* · *instance attribute*

```python
latent_heat_energy: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L60-L60)

### particula.execution.prepared_types.SimulationDiagnostics.number_concentration

*attribute* · *instance attribute*

```python
number_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L52-L52)

### particula.execution.prepared_types.SimulationDiagnostics.particle_mass_concentration

*attribute* · *instance attribute*

```python
particle_mass_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L54-L54)

### particula.execution.prepared_types.SimulationDiagnostics.pressure

*attribute* · *instance attribute*

```python
pressure: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L57-L57)

### particula.execution.prepared_types.SimulationDiagnostics.saturation_ratio

*attribute* · *instance attribute*

```python
saturation_ratio: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L58-L58)

### particula.execution.prepared_types.SimulationDiagnostics.simulated_time

*attribute* · *instance attribute*

```python
simulated_time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L49-L49)

### particula.execution.prepared_types.SimulationDiagnostics.temperature

*attribute* · *instance attribute*

```python
temperature: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L56-L56)

### particula.execution.prepared_types.SimulationDiagnostics.volume

*attribute* · *instance attribute*

```python
volume: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L59-L59)

### particula.execution.prepared_types.SimulationDiagnostics.__init__

*method*

```python
def __init__(completed_steps: int, simulated_time: float, gas_names: tuple[str, ...], active_slots: np.ndarray, number_concentration: np.ndarray, charge_concentration: np.ndarray, particle_mass_concentration: np.ndarray, gas_concentration: np.ndarray, temperature: np.ndarray, pressure: np.ndarray, saturation_ratio: np.ndarray, volume: np.ndarray, latent_heat_energy: np.ndarray | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_types.py#L0-L0)
