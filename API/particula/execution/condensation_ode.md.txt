# particula.execution.condensation_ode

*module*

Native CPU condensation as explicitly declared continuous mass rates.

The owner detaches initial state and process configuration. Trial coordinates
are scaled per-particle kg and gas kg/m3, with independent absolute tolerances.
Full species maps, resolved counts/volume and PDF quadrature remain native.
This path integrates isothermal rates, not inventory-limited finite-step maps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L1-L1)

## particula.execution.condensation_ode.CondensationODE

*class*

```python
class CondensationODE
```

Own detached native state and one shared deterministic CPU controller.

particle_atol is per-particle kg; gas_atol is kg/m3. Both accept scalars or
their full native array shape. rtol is dimensionless. All boxes share one
accepted clock and a maximum component error, so no small box/species can
be hidden in an average. Forcing, if supplied, is a pure function of time
returning (temperature[K], pressure[Pa]), each scalar or one value per box.
Discontinuities must be explicit run boundaries. Full evaporation of an
active slot is a domain boundary; unsupported crossings fail at min_step.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L25-L267)

### particula.execution.condensation_ode.CondensationODE._base

*attribute* · *instance attribute*

```python
_base = _cpu_batch.detached(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L60-L60)

### particula.execution.condensation_ode.CondensationODE._forcing

*attribute* · *instance attribute*

```python
_forcing = forcing
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L62-L62)

### particula.execution.condensation_ode.CondensationODE._gas_indices

*attribute* · *instance attribute*

```python
_gas_indices = tuple(gas_indices)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L102-L102)

### particula.execution.condensation_ode.CondensationODE._maps

*attribute* · *instance attribute*

```python
_maps = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L75-L75)

### particula.execution.condensation_ode.CondensationODE._mass_count

*attribute* · *instance attribute*

```python
_mass_count = len(self._mass_indices)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L103-L103)

### particula.execution.condensation_ode.CondensationODE._mass_indices

*attribute* · *instance attribute*

```python
_mass_indices = tuple(mass_indices)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L101-L101)

### particula.execution.condensation_ode.CondensationODE._scale

*attribute* · *instance attribute*

```python
_scale = np.array(mass_scale + gas_scale)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L114-L114)

### particula.execution.condensation_ode.CondensationODE._shape

*attribute* · *instance attribute*

```python
_shape = aerosol.particles.masses.shape
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L72-L72)

### particula.execution.condensation_ode.CondensationODE._strategy

*attribute* · *instance attribute*

```python
_strategy = deepcopy(strategy)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L61-L61)

### particula.execution.condensation_ode.CondensationODE._viscosity

*attribute* · *instance attribute*

```python
_viscosity = None if dynamic_viscosity is None else self._tolerances(dynamic_viscosity, (aerosol.particles.n_boxes), 'dynamic_viscosity').copy()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L63-L71)

### particula.execution.condensation_ode.CondensationODE._weights

*attribute* · *instance attribute*

```python
_weights = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L76-L76)

### particula.execution.condensation_ode.CondensationODE.problem

*attribute* · *instance attribute*

```python
problem = ODEProblem(self._rate, self._admissible, self._jacobian, self._partial_time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L123-L125)

### particula.execution.condensation_ode.CondensationODE.solver

*attribute* · *instance attribute*

```python
solver = AdaptiveSolver(self.problem, values / self._scale, time=time, method=method, atol=tolerance / self._scale, rtol=rtol, **controls)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L126-L134)

### particula.execution.condensation_ode.CondensationODE.time

*property*

```python
time
```

Return accepted absolute time [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L257-L259)

### particula.execution.condensation_ode.CondensationODE.__init__

*method*

```python
def __init__(aerosol, strategy, *, particle_atol=1e-24, gas_atol=1e-12, rtol=1e-05, time=0.0, method='rk23', forcing=None, dynamic_viscosity=None, **controls)
```

Validate physics and construct scaled continuous coordinates.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L37-L136)

### particula.execution.condensation_ode.CondensationODE._admissible

*method*

```python
def _admissible(values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L187-L198)

### particula.execution.condensation_ode.CondensationODE._jacobian

*method*

```python
def _jacobian(time, values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L231-L246)

### particula.execution.condensation_ode.CondensationODE._pack

*method*

```python
def _pack(state)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L151-L155)

### particula.execution.condensation_ode.CondensationODE._partial_time

*method*

```python
def _partial_time(time, values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L248-L254)

### particula.execution.condensation_ode.CondensationODE._rate

*method*

```python
def _rate(time, values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L200-L229)

### particula.execution.condensation_ode.CondensationODE._tolerances

*method* · *staticmethod*

```python
def _tolerances(value, shape, name)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L138-L149)

### particula.execution.condensation_ode.CondensationODE._unpack

*method*

```python
def _unpack(time, values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L157-L185)

### particula.execution.condensation_ode.CondensationODE.snapshot

*method*

```python
def snapshot()
```

Return detached accepted native state, including forcing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L265-L267)

### particula.execution.condensation_ode.CondensationODE.step

*method*

```python
def step(bound)
```

Accept one physical step no later than bound [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/condensation_ode.py#L261-L263)
