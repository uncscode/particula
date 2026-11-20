# particula.dynamics.chemistry

*module*

Native, phase-qualified gas reaction mechanisms.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/__init__.py#L1-L1)

## particula.dynamics.chemistry.ReactionMechanism

*class* · *dataclass*

```python
class ReactionMechanism
```

Re-exported from: `particula.dynamics.chemistry.mechanism`

Closed reaction mechanism, independent of storage order and solver.

Construct with a version-1 Python mapping or load its JSON/YAML equivalent.
Version 2 admits explicit particle materials and reactive gas uptake.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L268-L376)

### particula.dynamics.chemistry.ReactionMechanism.declared_masses

*attribute* · *instance attribute*

```python
declared_masses: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L278-L278)

### particula.dynamics.chemistry.ReactionMechanism.reactions

*attribute* · *instance attribute*

```python
reactions: tuple[Reaction, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L277-L277)

### particula.dynamics.chemistry.ReactionMechanism.species

*attribute* · *instance attribute*

```python
species: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L276-L276)

### particula.dynamics.chemistry.ReactionMechanism.version

*attribute* · *instance attribute*

```python
version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L279-L279)

### particula.dynamics.chemistry.ReactionMechanism.__init__

*method*

```python
def __init__(record: dict)
```

Validate the versioned fields and copy the reaction definitions.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L281-L320)

### particula.dynamics.chemistry.ReactionMechanism.bind

*method*

```python
def bind(gas, overrides=None)
```

Resolve gas identities using exact names and explicit overrides.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L358-L364)

### particula.dynamics.chemistry.ReactionMechanism.bind_resolved

*method*

```python
def bind_resolved(aerosol, particle_columns, gas_overrides=None)
```

Bind explicit material columns and gas names in resolved state.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L366-L376)

### particula.dynamics.chemistry.ReactionMechanism.from_dict

*method* · *classmethod*

```python
def from_dict(record)
```

Validate and detach native Python mechanism data.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L322-L325)

### particula.dynamics.chemistry.ReactionMechanism.load

*method* · *classmethod*

```python
def load(path)
```

Read a native .json, .yaml or .yml file with unique mapping keys.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L342-L345)

### particula.dynamics.chemistry.ReactionMechanism.save

*method*

```python
def save(path)
```

Write a normalized native mechanism file.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L347-L356)

### particula.dynamics.chemistry.ReactionMechanism.to_dict

*method*

```python
def to_dict()
```

Return a detached, normalized representation for interchange.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L327-L340)

**Modules**

- `particula.dynamics.chemistry.adapter`
- `particula.dynamics.chemistry.binding`
- `particula.dynamics.chemistry.device`
- `particula.dynamics.chemistry.mechanism`
- `particula.dynamics.chemistry.ncar`
- `particula.dynamics.chemistry.process`
- `particula.dynamics.chemistry.resolved`
- `particula.dynamics.chemistry.resolved_adapter`
- `particula.dynamics.chemistry.resolved_binding`
- `particula.dynamics.chemistry.resolved_device`
