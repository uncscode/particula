# particula.gpu.population_metadata

*module*

Host-side interpretation of explicit Warp particle population metadata.

Codes are stable transfer metadata, not algorithm selectors. A known kind
does not establish that a particular numerical method supports it. Zero is
an invalid admission sentinel; it never implies resolved counts.
This module does not import Warp or inspect device payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L1-L1)

## particula.gpu.population_metadata.CONTINUOUS_PDF

*attribute* · *module attribute*

```python
CONTINUOUS_PDF = 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L14-L14)

## particula.gpu.population_metadata.DISCRETE

*attribute* · *module attribute*

```python
DISCRETE = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L13-L13)

## particula.gpu.population_metadata.PARTICLE_RESOLVED

*attribute* · *module attribute*

```python
PARTICLE_RESOLVED = 3
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L15-L15)

## particula.gpu.population_metadata._KINDS

*attribute* · *module attribute*

```python
_KINDS = {_UNTAGGED: None, DISCRETE: 'discrete', CONTINUOUS_PDF: 'continuous_pdf', PARTICLE_RESOLVED: 'particle_resolved'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L17-L22)

## particula.gpu.population_metadata._UNTAGGED

*attribute* · *module attribute*

```python
_UNTAGGED = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L12-L12)

## particula.gpu.population_metadata.particle_distribution_code

*function*

```python
def particle_distribution_code(particles: Any) -> int
```

Read a known integer metadata code without touching device arrays.

Missing metadata yields the invalid zero sentinel for consumer rejection.
Warp scalar wrappers expose their integer through ``value``. Unknown codes,
booleans and noninteger values reject without coercion.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L25-L39)

## particula.gpu.population_metadata.particle_distribution_type

*function*

```python
def particle_distribution_type(particles: Any) -> str
```

Decode transferable population metadata, rejecting unsupported PDF.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/population_metadata.py#L42-L49)
