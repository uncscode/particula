# particula.dynamics.coagulation.coagulation_factories

*module*

Coagulation Factory Module.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_factories.py#L1-L1)

## particula.dynamics.coagulation.coagulation_factories.CoagulationFactory

*class*

```python
class CoagulationFactory(StrategyFactoryABC[Union[BrownianCoagulationBuilder, ChargedCoagulationBuilder, TurbulentShearCoagulationBuilder, TurbulentDNSCoagulationBuilder, CombineCoagulationStrategyBuilder], CoagulationStrategyABC])
```

Bases: ``

Factory class for creating coagulation strategy instances
based on a given type string. Supported types include:
    - 'brownian'
    - 'charged'
    - 'turbulent_shear'
    - 'turbulent_dns'
    - 'combine'.

- get_builders() : Returns the mapping of strategy types to builder
    instances.
- get_strategy(strategy_type, parameters): Gets the strategy instance
    for the specified strategy type.
    - strategy_type: Type of coagulation strategy to use, can be
        'brownian', 'charged', 'turbulent_shear', 'turbulent_dns', or
        'combine'.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_factories.py#L27-L69)

### particula.dynamics.coagulation.coagulation_factories.CoagulationFactory.get_builders

*method*

```python
def get_builders() -> Dict[str, Any]
```

Return the mapping of strategy types to builder instances.

**Returns**

- (`Dict[str, Any]`) — Dictionary mapping strategy names to their corresponding builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_factories.py#L57-L69)
