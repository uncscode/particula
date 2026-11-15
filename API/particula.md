# particula

*module*

Top-level package for the particula particle simulator.

This package exposes the primary aerosol, execution, runnable, and namespace
modules used by the public API. Its frozen execution subset provides
dependency-neutral backend-selection metadata, context-local adapter
registration, the closed capability-error taxonomy, and fallback-policy enums.
Concrete execution mechanics and optional backends remain outside the import
path.

[View source](https://github.com/uncscode/particula/blob/main/particula/__init__.py#L1-L1)

## particula.__version__

*attribute* · *module attribute*

```python
__version__ = '0.3.1'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/__init__.py#L60-L60)

## particula.logger

*attribute* · *module attribute*

```python
logger = setup()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/__init__.py#L63-L63)

## particula.__getattr__

*function*

```python
def __getattr__(name: str)
```

Expose prepared simulation types without loading optional runtimes.

[View source](https://github.com/uncscode/particula/blob/main/particula/__init__.py#L68-L78)

**Modules**

- `particula._cpu_batch`
- `particula.abc_builder`
- `particula.abc_factory`
- `particula.activity`
- `particula.aerosol`
- `particula.aerosol_builder`
- `particula.aerosol_validation`
- `particula.box_process`
- `particula.builder_mixin`
- `particula.cpu_transfer`
- `particula.dynamics`
- `particula.equilibria`
- `particula.execution`
- `particula.extension_fields`
- `particula.gas`
- `particula.gpu`
- `particula.logger_setup`
- `particula.optics`
- `particula.particles`
- `particula.runnable`
- `particula.util`
