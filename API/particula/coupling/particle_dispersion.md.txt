# particula.coupling.particle_dispersion

*module*

Resolved particle transport using prescribed eddy diffusivity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L1-L1)

## particula.coupling.particle_dispersion.ParticleDispersion

*class* · *dataclass*

```python
class ParticleDispersion
```

Configure seeded particle dispersion on the accepted forcing sequence.

**Parameters**

- `seed` (`int`) — Unsigned 64-bit integer selecting an independent ensemble.
- `max_step` (`float`) — Maximum particle substep in seconds. Decrease this when resolving first contact with an open or depositing boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L10-L106)

### particula.coupling.particle_dispersion.ParticleDispersion.algorithm

*attribute* · *class attribute*

```python
algorithm: str = 'prescribed-dispersion-v1'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L22-L22)

### particula.coupling.particle_dispersion.ParticleDispersion.max_step

*attribute* · *instance attribute*

```python
max_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L21-L21)

### particula.coupling.particle_dispersion.ParticleDispersion.seed

*attribute* · *instance attribute*

```python
seed: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L20-L20)

### particula.coupling.particle_dispersion.ParticleDispersion.__init__

*method*

```python
def __init__(seed: int, max_step: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L0-L0)

### particula.coupling.particle_dispersion.ParticleDispersion.__post_init__

*method*

```python
def __post_init__()
```

Validate the seed and duration before allocating particle arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L24-L31)

### particula.coupling.particle_dispersion.ParticleDispersion._uniform_cartesian

*method* · *staticmethod*

```python
def _uniform_cartesian(grid, frame, k, active)
```

Check both cell samples and face conductances against constant K.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L83-L106)

### particula.coupling.particle_dispersion.ParticleDispersion.coefficients

*method*

```python
def coefficients(grid, frame)
```

Return constant Cartesian K or full-interval jump fractions.

Each shared face moves an equal fraction of the same physical volume
in either direction. An absorbing exterior uses the distance from
the cell center to the face; reflecting faces transfer zero mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L55-L81)

### particula.coupling.particle_dispersion.ParticleDispersion.configuration

*method*

```python
def configuration()
```

Return the restart contract for counter-based particle draws.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L33-L37)

### particula.coupling.particle_dispersion.ParticleDispersion.keys

*method*

```python
def keys(particle_ids)
```

Derive one stable key per identity, independent of storage order.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/particle_dispersion.py#L39-L53)
