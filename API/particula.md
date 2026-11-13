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
__version__ = '0.2.15'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/__init__.py#L57-L57)

## particula.logger

*attribute* · *module attribute*

```python
logger = setup()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/__init__.py#L60-L60)

**Modules**

- `particula.abc_builder`
- `particula.abc_factory`
- `particula.activity`
- `particula.aerosol`
- `particula.aerosol_builder`
- `particula.aerosol_validation`
- `particula.builder_mixin`
- `particula.dynamics`
- `particula.equilibria`
- `particula.execution`
- `particula.gas`
- `particula.gpu`
- `particula.logger_setup`
- `particula.particles`
- `particula.runnable`
- `particula.util`
