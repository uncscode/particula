# particula.dynamics.nucleation.nucleation_factories

*module*

Create bounded public construction nucleation strategies with fresh strict
    builders.

The factory dispatches only activation and kinetic potential-rate strategies.
It does not construct source-selection metadata or expose concrete demand
planning/source transaction
``particle_source`` finalization and mutation helpers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_factories.py#L1-L1)

## particula.dynamics.nucleation.nucleation_factories.NucleationFactory

*class*

```python
class NucleationFactory(StrategyFactoryABC[ActivationNucleationBuilder | KineticNucleationBuilder, ActivationNucleationStrategy | KineticNucleationStrategy])
```

Bases: ``

Build activation or kinetic public construction potential-rate strategies
    by name.

Each request gets a new builder, so failed or successful requests cannot
retain configuration into subsequent calls. Identifiers are case-insensitive
but are limited to ``"activation"`` and ``"kinetic"``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_factories.py#L21-L76)

### particula.dynamics.nucleation.nucleation_factories.NucleationFactory.get_builders

*method*

```python
def get_builders() -> dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]
```

Return fresh builders for the two supported identifiers.

**Returns**

- (`dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]`) — New activation and kinetic builders under their canonical lowercase
- (`dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]`) — identifiers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_factories.py#L35-L47)

### particula.dynamics.nucleation.nucleation_factories.NucleationFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: dict[str, object] | None = None) -> ActivationNucleationStrategy | KineticNucleationStrategy
```

Build a validated strategy from a copied strict-schema payload.

**Parameters**

- `strategy_type` (`str`) — Case-insensitive ``"activation"`` or ``"kinetic"`` identifier.
- `parameters` (`dict[str, object] | None`) (default: `None`) — Required dictionary accepted by the selected builder.

**Returns**

- (`ActivationNucleationStrategy | KineticNucleationStrategy`) — Newly built immutable activation or kinetic potential-rate strategy.

**Raises**

- `ValueError` — If the identifier or payload is invalid, unsupported, or
fails the selected builder's schema validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_factories.py#L49-L76)
