# particula.dynamics.chemistry.binding

*module*

Resolve native gas identities and compile molar kinetic coefficients.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L1-L1)

## particula.dynamics.chemistry.binding.BoundMechanism

*class* · *dataclass*

```python
class BoundMechanism
```

Immutable layout and mechanism shared by CPU and device preparation.

Matrix rows are reactions and columns follow the mechanism species order.
Concentrations supplied to rate and Jacobian evaluation are mol/m³ of air.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L80-L140)

### particula.dynamics.chemistry.binding.BoundMechanism.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L89-L89)

### particula.dynamics.chemistry.binding.BoundMechanism.indices

*attribute* · *instance attribute*

```python
indices: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L90-L90)

### particula.dynamics.chemistry.binding.BoundMechanism.mechanism

*attribute* · *instance attribute*

```python
mechanism: ReactionMechanism
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L88-L88)

### particula.dynamics.chemistry.binding.BoundMechanism.molar_masses

*attribute* · *instance attribute*

```python
molar_masses: tuple[float, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L91-L91)

### particula.dynamics.chemistry.binding.BoundMechanism.orders

*attribute* · *instance attribute*

```python
orders: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L93-L93)

### particula.dynamics.chemistry.binding.BoundMechanism.parameters

*attribute* · *instance attribute*

```python
parameters: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L94-L94)

### particula.dynamics.chemistry.binding.BoundMechanism.stoichiometry

*attribute* · *instance attribute*

```python
stoichiometry: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L92-L92)

### particula.dynamics.chemistry.binding.BoundMechanism.__init__

*method*

```python
def __init__(mechanism: ReactionMechanism, gas_names: tuple[str, ...], indices: tuple[int, ...], molar_masses: tuple[float, ...], stoichiometry: tuple, orders: tuple, parameters: tuple) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L0-L0)

### particula.dynamics.chemistry.binding.BoundMechanism.from_gas

*method* · *classmethod*

```python
def from_gas(mechanism, gas, overrides)
```

Compile layout indices, balance checks and rate parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L96-L113)

### particula.dynamics.chemistry.binding.BoundMechanism.kinetics

*method*

```python
def kinetics(temperature)
```

Build per-box molar rate and Jacobian evaluators at fixed kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L126-L140)

### particula.dynamics.chemistry.binding.BoundMechanism.validate

*method*

```python
def validate(gas)
```

Reject a stale ordered layout or mass before evaluating chemistry.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L115-L124)

## particula.dynamics.chemistry.binding.MolarKinetics

*class* · *dataclass*

```python
class MolarKinetics
```

Rate of concentration change in mol/m³/s and its state derivative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L143-L168)

### particula.dynamics.chemistry.binding.MolarKinetics.coefficients

*attribute* · *instance attribute*

```python
coefficients: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L149-L149)

### particula.dynamics.chemistry.binding.MolarKinetics.net

*attribute* · *instance attribute*

```python
net: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L147-L147)

### particula.dynamics.chemistry.binding.MolarKinetics.orders

*attribute* · *instance attribute*

```python
orders: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L148-L148)

### particula.dynamics.chemistry.binding.MolarKinetics.__init__

*method*

```python
def __init__(net: np.ndarray, orders: np.ndarray, coefficients: np.ndarray) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L0-L0)

### particula.dynamics.chemistry.binding.MolarKinetics.jacobian

*method*

```python
def jacobian(time, state)
```

Evaluate the rate derivative even at zero concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L157-L168)

### particula.dynamics.chemistry.binding.MolarKinetics.rate

*method*

```python
def rate(time, state)
```

Evaluate all reaction extents before summing species tendencies.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L151-L155)

## particula.dynamics.chemistry.binding._compile_network

*function*

```python
def _compile_network(mechanism, species, masses)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L49-L77)

## particula.dynamics.chemistry.binding._resolve_species

*function*

```python
def _resolve_species(mechanism, gas, overrides)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L11-L38)

## particula.dynamics.chemistry.binding._sparse_stoichiometry

*function*

```python
def _sparse_stoichiometry(rows)
```

Compile reaction offsets, affected columns and signed coefficients.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/binding.py#L41-L46)
