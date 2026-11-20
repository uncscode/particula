# particula.dynamics.chemistry.mechanism

*module*

Versioned native mechanism records for irreversible gas mass action.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L1-L1)

## particula.dynamics.chemistry.mechanism.Reaction

*class* · *dataclass*

```python
class Reaction
```

Validated immutable reaction with SI molar coefficients.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L86-L197)

### particula.dynamics.chemistry.mechanism.Reaction.coefficient

*attribute* · *instance attribute*

```python
coefficient: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L95-L95)

### particula.dynamics.chemistry.mechanism.Reaction.energy

*attribute* · *instance attribute*

```python
energy: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L97-L97)

### particula.dynamics.chemistry.mechanism.Reaction.exponent

*attribute* · *instance attribute*

```python
exponent: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L96-L96)

### particula.dynamics.chemistry.mechanism.Reaction.law

*attribute* · *instance attribute*

```python
law: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L94-L94)

### particula.dynamics.chemistry.mechanism.Reaction.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L90-L90)

### particula.dynamics.chemistry.mechanism.Reaction.orders

*attribute* · *instance attribute*

```python
orders: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L93-L93)

### particula.dynamics.chemistry.mechanism.Reaction.products

*attribute* · *instance attribute*

```python
products: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L92-L92)

### particula.dynamics.chemistry.mechanism.Reaction.reactants

*attribute* · *instance attribute*

```python
reactants: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L91-L91)

### particula.dynamics.chemistry.mechanism.Reaction.reference_temperature

*attribute* · *instance attribute*

```python
reference_temperature: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L98-L98)

### particula.dynamics.chemistry.mechanism.Reaction.__init__

*method*

```python
def __init__(name: str, reactants: tuple, products: tuple, orders: tuple, law: str, coefficient: float, exponent: float, energy: float, reference_temperature: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L0-L0)

### particula.dynamics.chemistry.mechanism.Reaction.parse

*method* · *classmethod*

```python
def parse(record, names)
```

Check a reaction against the declared species and SI units.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L100-L153)

### particula.dynamics.chemistry.mechanism.Reaction.signed_coefficients

*method*

```python
def signed_coefficients()
```

Return affected species and net molecular counts per reaction.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L155-L162)

### particula.dynamics.chemistry.mechanism.Reaction.to_dict

*method*

```python
def to_dict()
```

Serialize reaction orders and rate parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L174-L197)

### particula.dynamics.chemistry.mechanism.Reaction.validate_masses

*method*

```python
def validate_masses(signed, masses)
```

Check closed mass balance when every net participant has a mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L164-L172)

## particula.dynamics.chemistry.mechanism.ReactionMechanism

*class* · *dataclass*

```python
class ReactionMechanism
```

Closed reaction mechanism, independent of storage order and solver.

Construct with a version-1 Python mapping or load its JSON/YAML equivalent.
Version 2 admits explicit particle materials and reactive gas uptake.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L268-L376)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.declared_masses

*attribute* · *instance attribute*

```python
declared_masses: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L278-L278)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.reactions

*attribute* · *instance attribute*

```python
reactions: tuple[Reaction, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L277-L277)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.species

*attribute* · *instance attribute*

```python
species: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L276-L276)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.version

*attribute* · *instance attribute*

```python
version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L279-L279)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.__init__

*method*

```python
def __init__(record: dict)
```

Validate the versioned fields and copy the reaction definitions.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L281-L320)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.bind

*method*

```python
def bind(gas, overrides=None)
```

Resolve gas identities using exact names and explicit overrides.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L358-L364)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.bind_resolved

*method*

```python
def bind_resolved(aerosol, particle_columns, gas_overrides=None)
```

Bind explicit material columns and gas names in resolved state.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L366-L376)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.from_dict

*method* · *classmethod*

```python
def from_dict(record)
```

Validate and detach native Python mechanism data.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L322-L325)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.load

*method* · *classmethod*

```python
def load(path)
```

Read a native .json, .yaml or .yml file with unique mapping keys.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L342-L345)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.save

*method*

```python
def save(path)
```

Write a normalized native mechanism file.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L347-L356)

### particula.dynamics.chemistry.mechanism.ReactionMechanism.to_dict

*method*

```python
def to_dict()
```

Return a detached, normalized representation for interchange.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L327-L340)

## particula.dynamics.chemistry.mechanism._MechanismLoader

*class*

```python
class _MechanismLoader(yaml.SafeLoader)
```

Bases: `yaml.SafeLoader`

Preserve duplicate-key diagnostics for scientific input.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L20-L21)

## particula.dynamics.chemistry.mechanism._catalog

*function*

```python
def _catalog(catalog, phases)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L230-L252)

## particula.dynamics.chemistry.mechanism._check_phases

*function*

```python
def _check_phases(reaction)
```

Restrict the kinetic collision law to one gas and particle products.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L200-L227)

## particula.dynamics.chemistry.mechanism._counts

*function*

```python
def _counts(values, names, label, *, minimum)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L63-L73)

## particula.dynamics.chemistry.mechanism._declared_masses

*function*

```python
def _declared_masses(catalog)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L255-L265)

## particula.dynamics.chemistry.mechanism._fields

*function*

```python
def _fields(value, required, optional=())
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L48-L54)

## particula.dynamics.chemistry.mechanism._number

*function*

```python
def _number(value, name)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L57-L60)

## particula.dynamics.chemistry.mechanism._read_mapping

*function*

```python
def _read_mapping(path)
```

Read JSON or YAML without silently discarding duplicate keys.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L33-L45)

## particula.dynamics.chemistry.mechanism._unique

*function*

```python
def _unique(pairs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L11-L17)

## particula.dynamics.chemistry.mechanism._units

*function*

```python
def _units(order)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L76-L83)

## particula.dynamics.chemistry.mechanism._yaml_mapping

*function*

```python
def _yaml_mapping(loader, node)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/mechanism.py#L24-L25)
