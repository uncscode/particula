# particula.dynamics.wall_loss._native

*module*

Native CPU wall-loss admission and staged updates, without facade state.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L1-L1)

## particula.dynamics.wall_loss._native._controls

*function*

```python
def _controls(strategy: Any) -> None
```

Revalidate mutable concrete geometry and field configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L23-L60)

## particula.dynamics.wall_loss._native._scalar

*function*

```python
def _scalar(value: Any, name: str, *, positive: bool = False) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L12-L20)

## particula.dynamics.wall_loss._native._storage

*function*

```python
def _storage(strategy: Any, particle: ParticleData, resolved: bool) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L159-L184)

## particula.dynamics.wall_loss._native.coefficient

*function*

```python
def coefficient(strategy: Any, particle: ParticleData, temperature: float, pressure: float, *, include_inactive: bool = False) -> NDArray[np.float64]
```

Apply existing coefficient laws to native material and charge arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L101-L144)

## particula.dynamics.wall_loss._native.rate

*function*

```python
def rate(strategy: Any, particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Return number-density (or dn/dr) loss per second.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L147-L156)

## particula.dynamics.wall_loss._native.step

*function*

```python
def step(strategy: Any, particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Stage the existing CPU deterministic or Bernoulli removal law.

Resolved removal clears selected masses/counts/charge so removed slots can
be reused. Untouched inactive lanes are not repaired. The CPU Bernoulli
removal law and its random lifecycle are unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L187-L271)

## particula.dynamics.wall_loss._native.validate

*function*

```python
def validate(strategy: Any, particle: ParticleData, temperature: float, pressure: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Check current native storage, physics, representation and controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/_native.py#L63-L98)
