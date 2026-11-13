# particula.aerosol

*module*

Identity-preserving aggregate of native CPU aerosol data.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L1-L1)

## particula.aerosol.Aerosol

*class*

```python
class Aerosol
```

Hold particle, gas and environment containers without copying.

Array mutation remains caller-owned. Processes must revalidate their
physical and chemical requirements. Construction never transfers data or
rebinds prepared execution state.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L9-L120)

### particula.aerosol.Aerosol.environment

*property* · *writable*

```python
environment: EnvironmentData
```

Return the held environment container by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L64-L66)

### particula.aerosol.Aerosol.gas

*property* · *writable*

```python
gas: GasData
```

Return the held full gas container by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L52-L54)

### particula.aerosol.Aerosol.particles

*property* · *writable*

```python
particles: ParticleData
```

Return the held particle container by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L40-L42)

### particula.aerosol.Aerosol.__init__

*method*

```python
def __init__(particles: ParticleData, gas: GasData, environment: EnvironmentData) -> None
```

Validate the complete triple before retaining supplied identities.

**Parameters**

- `particles` (`ParticleData`) — Explicitly tagged CPU particle storage.
- `gas` (`GasData`) — Full ordered CPU gas storage, including inert species.
- `environment` (`EnvironmentData`) — CPU environment with one ratio lane per gas species.

**Raises**

- `TypeError` — If a container has the wrong type.
- `ValueError` — If structure or particle representation is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L17-L37)

### particula.aerosol.Aerosol.__str__

*method*

```python
def __str__() -> str
```

Summarize the current three containers.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L118-L120)

### particula.aerosol.Aerosol._replace_data

*method*

```python
def _replace_data(particles: ParticleData, gas: GasData, environment: EnvironmentData) -> None
```

Validate candidates before publishing their references.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L102-L116)

### particula.aerosol.Aerosol.replace_data

*method*

```python
def replace_data(particles: ParticleData, gas: GasData, environment: EnvironmentData) -> None
```

Publish a complete candidate triple only after read-only admission.

**Parameters**

- `particles` (`ParticleData`) — Tagged CPU particle data, retained by identity.
- `gas` (`GasData`) — Full ordered gas data, retained by identity.
- `environment` (`EnvironmentData`) — Matching per-box environment, retained by identity.

**Raises**

- `TypeError` — If a supplied container has the wrong type.
- `ValueError` — If structure or particle representation is invalid.

Rejection leaves old and candidate state unchanged. Same-object
replacements revalidate current data. Success selects supplied state;
it neither conserves/rescales inventory nor rebinds GPU resources.
This is validation atomicity, not a concurrent-access guarantee.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L75-L100)
