# particula.dynamics.chemistry.adapter

*module*

Prepare and budget the resident gas mechanism on an explicit device.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/adapter.py#L1-L1)

## particula.dynamics.chemistry.adapter.GasChemistryAdapter

*class* · *dataclass*

```python
class GasChemistryAdapter
```

Resident implementation for an explicitly registered GasChemistry.

workspace_bytes bounds chemistry scratch in bytes across all boxes.
RK23 uses linear species workspace. ROS3 also owns two dense matrices
and pivot indices per box. Mechanism inputs and counters are separate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/adapter.py#L17-L142)

### particula.dynamics.chemistry.adapter.GasChemistryAdapter.workspace_bytes

*attribute* · *class attribute* · *instance attribute*

```python
workspace_bytes: int = 256 * 1024 ** 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/adapter.py#L26-L26)

### particula.dynamics.chemistry.adapter.GasChemistryAdapter.__init__

*method*

```python
def __init__(workspace_bytes: int = 256 * 1024 ** 2) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/adapter.py#L0-L0)

### particula.dynamics.chemistry.adapter.GasChemistryAdapter.plan

*method*

```python
def plan(process, aerosol)
```

Check the species layout and all scratch sizes before allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/adapter.py#L28-L70)

### particula.dynamics.chemistry.adapter.GasChemistryAdapter.prepare

*method*

```python
def prepare(plan, context)
```

Upload fixed chemistry data once and retain all work on device.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/adapter.py#L72-L142)
