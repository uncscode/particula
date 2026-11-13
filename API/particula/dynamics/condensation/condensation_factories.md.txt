# particula.dynamics.condensation.condensation_factories

*module*

Factory for building condensation strategies.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_factories.py#L1-L1)

## particula.dynamics.condensation.condensation_factories.CondensationFactory

*class*

```python
class CondensationFactory(StrategyFactoryABC[Union[CondensationLatentHeatBuilder, CondensationIsothermalBuilder, CondensationIsothermalStaggeredBuilder], CondensationStrategy])
```

Bases: ``

Factory class for condensation strategies.

> **Supports strategy types**
>
> - "isothermal": Standard isothermal condensation.
> - "isothermal_staggered": Staggered isothermal condensation with
>   batch stepping for stability.
> - "latent_heat": Condensation with latent-heat coupling.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_factories.py#L15-L53)

### particula.dynamics.condensation.condensation_factories.CondensationFactory.get_builders

*method*

```python
def get_builders() -> Dict[str, Union[CondensationLatentHeatBuilder, CondensationIsothermalBuilder, CondensationIsothermalStaggeredBuilder]]
```

Return the mapping of strategy types to builder instances.

**Returns**

- (`Dict[str, Union[CondensationLatentHeatBuilder, CondensationIsothermalBuilder, CondensationIsothermalStaggeredBuilder]]`) — Dictionary mapping condensation strategy names to builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_factories.py#L34-L53)
