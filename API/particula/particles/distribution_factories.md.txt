# particula.particles.distribution_factories

*module*

Factory Classes for selecting the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_factories.py#L1-L1)

## particula.particles.distribution_factories.DistributionFactory

*class*

```python
class DistributionFactory(StrategyFactoryABC[Union[MassBasedMovingBinBuilder, RadiiBasedMovingBinBuilder, SpeciatedMassMovingBinBuilder, ParticleResolvedSpeciatedMassBuilder], Union[MassBasedMovingBin, RadiiBasedMovingBin, SpeciatedMassMovingBin, ParticleResolvedSpeciatedMass]])
```

Bases: ``

Factory class to create distribution strategies from builders.

This factory is used to obtain particle distribution strategies
based on the specified representation type (mass-based, radius-based,
speciated, or particle-resolved).

**Functions**

- `- get_builders ` — Return a mapping of strategy types to builder
instances.
- `- get_strategy ` — Return a strategy instance for a given strategy type.

**Returns**

- `result` (`DistributionStrategy`) — An instance configured for the chosen distribution representation.

**Raises**

- `ValueError` — If an unknown strategy type is provided or if
required parameters are missing or invalid.

**Examples**

```py title="DistributionFactory Example"
import particula as par
factory = par.particles.DistributionFactory()
strategy = factory.get_strategy("mass_based_moving_bin")
# strategy -> MassBasedMovingBin()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_factories.py#L20-L87)

### particula.particles.distribution_factories.DistributionFactory.get_builders

*method*

```python
def get_builders()
```

Return a mapping of strategy types to builder instances.

**Returns**

- `result` (`dict`) — A dictionary where each key is a string identifying the strategy type, and each value is the corresponding builder object.

**Examples**

```py title="get_builders Example"
import particula as par
factory = par.particles.DistributionFactory()
builder_map = factory.get_builders()
# builder_map["mass_based_moving_bin"] -> MassBasedMovingBinBuilder
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_factories.py#L64-L87)
