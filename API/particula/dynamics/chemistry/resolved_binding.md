# particula.dynamics.chemistry.resolved_binding

*module*

Immutable physical layout for bulk material reactions and gas uptake.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L1-L1)

## particula.dynamics.chemistry.resolved_binding.ResolvedBinding

*class* · *dataclass*

```python
class ResolvedBinding
```

Reaction columns follow all gas lanes, then all particle material lanes.

Densities are kg/m³ of material and molar masses are kg/mol. Unmapped
material contributes to geometry without participating in any reaction.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L71-L181)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.density

*attribute* · *instance attribute*

```python
density: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L82-L82)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.gas_masses

*attribute* · *instance attribute*

```python
gas_masses: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L81-L81)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L80-L80)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.indices

*attribute* · *instance attribute*

```python
indices: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L84-L84)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.kinds

*attribute* · *instance attribute*

```python
kinds: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L88-L88)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.mechanism

*attribute* · *instance attribute*

```python
mechanism: ReactionMechanism
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L79-L79)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.net

*attribute* · *instance attribute*

```python
net: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L85-L85)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.orders

*attribute* · *instance attribute*

```python
orders: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L86-L86)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.parameters

*attribute* · *instance attribute*

```python
parameters: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L87-L87)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.particle_masses

*attribute* · *instance attribute*

```python
particle_masses: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L83-L83)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.sources

*attribute* · *instance attribute*

```python
sources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L89-L89)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.__init__

*method*

```python
def __init__(mechanism: ReactionMechanism, gas_names: tuple, gas_masses: tuple, density: tuple, particle_masses: tuple, indices: tuple, net: tuple, orders: tuple, parameters: tuple, kinds: tuple, sources: tuple) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L0-L0)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.create

*method* · *classmethod*

```python
def create(mechanism, aerosol, particle_columns, gas_overrides)
```

Resolve names once and check units, stoichiometry and collisions.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L91-L147)

### particula.dynamics.chemistry.resolved_binding.ResolvedBinding.validate

*method*

```python
def validate(aerosol)
```

Require the ordered metadata used at binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L149-L181)

## particula.dynamics.chemistry.resolved_binding._reaction_kinds

*function*

```python
def _reaction_kinds(mechanism, species, lanes, n_gas)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L52-L68)

## particula.dynamics.chemistry.resolved_binding._resolve_columns

*function*

```python
def _resolve_columns(species, materials, names, gas, particle_masses, declared, particle_columns, overrides)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_binding.py#L14-L49)
