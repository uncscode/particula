# particula.dynamics.chemistry.resolved_adapter

*module*

Resident adapter for resolved bulk chemistry and reactive uptake.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_adapter.py#L1-L1)

## particula.dynamics.chemistry.resolved_adapter.ResolvedChemistryAdapter

*class* · *dataclass*

```python
class ResolvedChemistryAdapter
```

Preallocate all trial vectors and share a failure latch across boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_adapter.py#L17-L121)

### particula.dynamics.chemistry.resolved_adapter.ResolvedChemistryAdapter.workspace_bytes

*attribute* · *class attribute* · *instance attribute*

```python
workspace_bytes: int = 256 * 1024 ** 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_adapter.py#L21-L21)

### particula.dynamics.chemistry.resolved_adapter.ResolvedChemistryAdapter.__init__

*method*

```python
def __init__(workspace_bytes: int = 256 * 1024 ** 2) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_adapter.py#L0-L0)

### particula.dynamics.chemistry.resolved_adapter.ResolvedChemistryAdapter.plan

*method*

```python
def plan(process, aerosol)
```

Validate metadata, index limits and scratch demand before upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_adapter.py#L23-L67)

### particula.dynamics.chemistry.resolved_adapter.ResolvedChemistryAdapter.prepare

*method*

```python
def prepare(plan, context)
```

Bind native storage and allocate per-box trial workspace.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_adapter.py#L69-L121)
