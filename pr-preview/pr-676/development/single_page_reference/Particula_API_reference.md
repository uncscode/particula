

---
# README.md

# Particula Index

> Auto-generated documentation index.

A full list of [Particula](https://github.com/uncscode/particula) project modules.

- [Particula](particula/index.md#particula)
    - [Abc Builder](particula/abc_builder.md#abc-builder)
    - [Abc Factory](particula/abc_factory.md#abc-factory)
    - [Activity](particula/activity/index.md#activity)
        - [Activity Coefficients](particula/activity/activity_coefficients.md#activity-coefficients)
        - [Bat Blending](particula/activity/bat_blending.md#bat-blending)
        - [Bat Coefficients](particula/activity/bat_coefficients.md#bat-coefficients)
        - [Convert Functional Group](particula/activity/convert_functional_group.md#convert-functional-group)
        - [Gibbs](particula/activity/gibbs.md#gibbs)
        - [Gibbs Mixing](particula/activity/gibbs_mixing.md#gibbs-mixing)
        - [Phase Separation](particula/activity/phase_separation.md#phase-separation)
        - [Ratio](particula/activity/ratio.md#ratio)
        - [Species Density](particula/activity/species_density.md#species-density)
        - [Water Activity](particula/activity/water_activity.md#water-activity)
    - [Aerosol](particula/aerosol.md#aerosol)
    - [Builder Mixin](particula/builder_mixin.md#builder-mixin)
    - [Dynamics](particula/dynamics/index.md#dynamics)
        - [Coagulation](particula/dynamics/coagulation/index.md#coagulation)
            - [Brownian Kernel](particula/dynamics/coagulation/brownian_kernel.md#brownian-kernel)
            - [Charged Dimensional Kernel](particula/dynamics/coagulation/charged_dimensional_kernel.md#charged-dimensional-kernel)
            - [Charged Dimensionless Kernel](particula/dynamics/coagulation/charged_dimensionless_kernel.md#charged-dimensionless-kernel)
            - [Charged Kernel Strategy](particula/dynamics/coagulation/charged_kernel_strategy.md#charged-kernel-strategy)
            - [Coagulation Builder](particula/dynamics/coagulation/coagulation_builder/index.md#coagulation-builder)
                - [BrownianCoagulationBuilder](particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.md#browniancoagulationbuilder)
                - [ChargedCoagulationBuilder](particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.md#chargedcoagulationbuilder)
                - [Coagulation Builder Mixin](particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.md#coagulation-builder-mixin)
                - [CombineCoagulationStrategyBuilder](particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.md#combinecoagulationstrategybuilder)
                - [TurbulentDNSCoagulationBuilder](particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.md#turbulentdnscoagulationbuilder)
                - [TurbulentShearCoagulationBuilder](particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.md#turbulentshearcoagulationbuilder)
            - [Coagulation Factories](particula/dynamics/coagulation/coagulation_factories.md#coagulation-factories)
            - [Coagulation Rate](particula/dynamics/coagulation/coagulation_rate.md#coagulation-rate)
            - [Coagulation Strategy](particula/dynamics/coagulation/coagulation_strategy/index.md#coagulation-strategy)
                - [BrownianCoagulationStrategy](particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.md#browniancoagulationstrategy)
                - [ChargedCoagulationStrategy](particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.md#chargedcoagulationstrategy)
                - [CoagulationStrategyABC](particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.md#coagulationstrategyabc)
                - [CombineCoagulationStrategy](particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.md#combinecoagulationstrategy)
                - [SedimentationCoagulationStrategy](particula/dynamics/coagulation/coagulation_strategy/sedimentation_coagulation_strategy.md#sedimentationcoagulationstrategy)
                - [TurbulentDNSCoagulationStrategy](particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.md#turbulentdnscoagulationstrategy)
                - [TurbulentShearCoagulationStrategy](particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.md#turbulentshearcoagulationstrategy)
            - [Particle Resolved Step](particula/dynamics/coagulation/particle_resolved_step/index.md#particle-resolved-step)
                - [Particle Resolved Method](particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.md#particle-resolved-method)
                - [Super Droplet Method](particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.md#super-droplet-method)
            - [Sedimentation Kernel](particula/dynamics/coagulation/sedimentation_kernel.md#sedimentation-kernel)
            - [Turbulent Dns Kernel](particula/dynamics/coagulation/turbulent_dns_kernel/index.md#turbulent-dns-kernel)
                - [G12 Radial Distribution Ao2008](particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.md#g12-radial-distribution-ao2008)
                - [Phi Ao2008](particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.md#phi-ao2008)
                - [Psi Ao2008](particula/dynamics/coagulation/turbulent_dns_kernel/psi_ao2008.md#psi-ao2008)
                - [Radial Velocity Module](particula/dynamics/coagulation/turbulent_dns_kernel/radial_velocity_module.md#radial-velocity-module)
                - [Sigma Relative Velocity Ao2008](particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.md#sigma-relative-velocity-ao2008)
                - [Turbulent Dns Kernel Ao2008](particula/dynamics/coagulation/turbulent_dns_kernel/turbulent_dns_kernel_ao2008.md#turbulent-dns-kernel-ao2008)
                - [Velocity Correlation F2 Ao2008](particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_f2_ao2008.md#velocity-correlation-f2-ao2008)
                - [Velocity Correlation Terms Ao2008](particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.md#velocity-correlation-terms-ao2008)
            - [Turbulent Shear Kernel](particula/dynamics/coagulation/turbulent_shear_kernel.md#turbulent-shear-kernel)
        - [Condensation](particula/dynamics/condensation/index.md#condensation)
            - [Condensation Strategies](particula/dynamics/condensation/condensation_strategies.md#condensation-strategies)
            - [Mass Transfer](particula/dynamics/condensation/mass_transfer.md#mass-transfer)
        - [Dilution](particula/dynamics/dilution.md#dilution)
        - [Particle Process](particula/dynamics/particle_process.md#particle-process)
        - [Properties](particula/dynamics/properties/index.md#properties)
            - [Wall Loss Coefficient](particula/dynamics/properties/wall_loss_coefficient.md#wall-loss-coefficient)
        - [Wall Loss](particula/dynamics/wall_loss.md#wall-loss)
    - [Equilibria](particula/equilibria/index.md#equilibria)
        - [Partitioning](particula/equilibria/partitioning.md#partitioning)
    - [Gas](particula/gas/index.md#gas)
        - [Atmosphere](particula/gas/atmosphere.md#atmosphere)
        - [Atmosphere Builders](particula/gas/atmosphere_builders.md#atmosphere-builders)
        - [Properties](particula/gas/properties/index.md#properties)
            - [Concentration Function](particula/gas/properties/concentration_function.md#concentration-function)
            - [Dynamic Viscosity](particula/gas/properties/dynamic_viscosity.md#dynamic-viscosity)
            - [Fluid Rms Velocity](particula/gas/properties/fluid_rms_velocity.md#fluid-rms-velocity)
            - [Integral Scale Module](particula/gas/properties/integral_scale_module.md#integral-scale-module)
            - [Kinematic Viscosity](particula/gas/properties/kinematic_viscosity.md#kinematic-viscosity)
            - [Kolmogorov Module](particula/gas/properties/kolmogorov_module.md#kolmogorov-module)
            - [Mean Free Path](particula/gas/properties/mean_free_path.md#mean-free-path)
            - [Normalize Accel Variance](particula/gas/properties/normalize_accel_variance.md#normalize-accel-variance)
            - [Pressure Function](particula/gas/properties/pressure_function.md#pressure-function)
            - [Taylor Microscale Module](particula/gas/properties/taylor_microscale_module.md#taylor-microscale-module)
            - [Thermal Conductivity](particula/gas/properties/thermal_conductivity.md#thermal-conductivity)
            - [Vapor Pressure Module](particula/gas/properties/vapor_pressure_module.md#vapor-pressure-module)
        - [Species](particula/gas/species.md#species)
        - [Species Builders](particula/gas/species_builders.md#species-builders)
        - [Species Factories](particula/gas/species_factories.md#species-factories)
        - [Vapor Pressure Builders](particula/gas/vapor_pressure_builders.md#vapor-pressure-builders)
        - [Vapor Pressure Factories](particula/gas/vapor_pressure_factories.md#vapor-pressure-factories)
        - [Vapor Pressure Strategies](particula/gas/vapor_pressure_strategies.md#vapor-pressure-strategies)
    - [Logger Setup](particula/logger_setup.md#logger-setup)
    - [Particles](particula/particles/index.md#particles)
        - [Activity Builders](particula/particles/activity_builders.md#activity-builders)
        - [Activity Factories](particula/particles/activity_factories.md#activity-factories)
        - [Activity Strategies](particula/particles/activity_strategies.md#activity-strategies)
        - [Change Particle Representation](particula/particles/change_particle_representation.md#change-particle-representation)
        - [Distribution Builders](particula/particles/distribution_builders.md#distribution-builders)
        - [Distribution Factories](particula/particles/distribution_factories.md#distribution-factories)
        - [Distribution Strategies](particula/particles/distribution_strategies.md#distribution-strategies)
        - [Properties](particula/particles/properties/index.md#properties)
            - [Activity Module](particula/particles/properties/activity_module.md#activity-module)
            - [Aerodynamic Mobility Module](particula/particles/properties/aerodynamic_mobility_module.md#aerodynamic-mobility-module)
            - [Aerodynamic Size](particula/particles/properties/aerodynamic_size.md#aerodynamic-size)
            - [Collision Radius Module](particula/particles/properties/collision_radius_module.md#collision-radius-module)
            - [Convert Kappa Volumes](particula/particles/properties/convert_kappa_volumes.md#convert-kappa-volumes)
            - [Convert Mass Concentration](particula/particles/properties/convert_mass_concentration.md#convert-mass-concentration)
            - [Convert Mole Fraction](particula/particles/properties/convert_mole_fraction.md#convert-mole-fraction)
            - [Convert Size Distribution](particula/particles/properties/convert_size_distribution.md#convert-size-distribution)
            - [Coulomb Enhancement](particula/particles/properties/coulomb_enhancement.md#coulomb-enhancement)
            - [Diffusion Coefficient](particula/particles/properties/diffusion_coefficient.md#diffusion-coefficient)
            - [Diffusive Knudsen Module](particula/particles/properties/diffusive_knudsen_module.md#diffusive-knudsen-module)
            - [Friction Factor Module](particula/particles/properties/friction_factor_module.md#friction-factor-module)
            - [Inertia Time](particula/particles/properties/inertia_time.md#inertia-time)
            - [Kelvin Effect Module](particula/particles/properties/kelvin_effect_module.md#kelvin-effect-module)
            - [Knudsen Number Module](particula/particles/properties/knudsen_number_module.md#knudsen-number-module)
            - [Lognormal Size Distribution](particula/particles/properties/lognormal_size_distribution.md#lognormal-size-distribution)
            - [Mean Thermal Speed Module](particula/particles/properties/mean_thermal_speed_module.md#mean-thermal-speed-module)
            - [Partial Pressure Module](particula/particles/properties/partial_pressure_module.md#partial-pressure-module)
            - [Reynolds Number](particula/particles/properties/reynolds_number.md#reynolds-number)
            - [Settling Velocity](particula/particles/properties/settling_velocity.md#settling-velocity)
            - [Slip Correction Module](particula/particles/properties/slip_correction_module.md#slip-correction-module)
            - [Special Functions](particula/particles/properties/special_functions.md#special-functions)
            - [Stokes Number](particula/particles/properties/stokes_number.md#stokes-number)
            - [Vapor Correction Module](particula/particles/properties/vapor_correction_module.md#vapor-correction-module)
        - [Representation](particula/particles/representation.md#representation)
        - [Representation Builders](particula/particles/representation_builders.md#representation-builders)
        - [Representation Factories](particula/particles/representation_factories.md#representation-factories)
        - [Surface Builders](particula/particles/surface_builders.md#surface-builders)
        - [Surface Factories](particula/particles/surface_factories.md#surface-factories)
        - [Surface Strategies](particula/particles/surface_strategies.md#surface-strategies)
    - [Runnable](particula/runnable.md#runnable)
    - [Util](particula/util/index.md#util)
        - [Arbitrary Round](particula/util/arbitrary_round.md#arbitrary-round)
        - [Colors](particula/util/colors.md#colors)
        - [Constants](particula/util/constants.md#constants)
        - [Convert Dtypes](particula/util/convert_dtypes.md#convert-dtypes)
        - [Convert Units](particula/util/convert_units.md#convert-units)
        - [Lf2013 Coagulation](particula/util/lf2013_coagulation/index.md#lf2013-coagulation)
            - [Src Lf2013 Coagulation](particula/util/lf2013_coagulation/src_lf2013_coagulation.md#src-lf2013-coagulation)
        - [Machine Limit](particula/util/machine_limit.md#machine-limit)
        - [Reduced Quantity](particula/util/reduced_quantity.md#reduced-quantity)
        - [Refractive Index Mixing](particula/util/refractive_index_mixing.md#refractive-index-mixing)
        - [Validate Inputs](particula/util/validate_inputs.md#validate-inputs)


---
# abc_builder.md

# Abc Builder

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Abc Builder

> Auto-generated documentation for [particula.abc_builder](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py) module.

## BuilderABC

[Show source in abc_builder.py:15](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L15)

Abstract base class for builders with common methods to check keys and
set parameters from a dictionary.

#### Attributes

- `-` *required_parameters* - List of required parameters for the builder.

#### Raises

- `-` *ValueError* - If any required key is missing during check_keys or
  pre_build_check, or if trying to set an invalid parameter.
- `-` *Warning* - If using default units for any parameter.

#### Examples

```py
class MyBuilder(BuilderABC):
    def set_parameter1(self, value, units=None):
        ...
    def set_parameter2(self, value, units=None):
        ...
    def build(self):
        return SomeStrategy()

strategy = (
    MyBuilder()
    .set_parameters1(10, 'm')
    .set_parameters2(20, 's')
    .build()
)
```

#### References

- "Builder Pattern,"
[Refactoring Guru](https://refactoring.guru/design-patterns/builder)

#### Signature

```python
class BuilderABC(ABC):
    def __init__(self, required_parameters: Optional[list[str]] = None): ...
```

### BuilderABC().build

[Show source in abc_builder.py:160](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L160)

Build and return the strategy object with the set parameters.

#### Returns

- `Any` - The built strategy object.

#### Examples

```py
builder = Builder()
strategy = builder.build()
```

#### Signature

```python
@abstractmethod
def build(self) -> Any: ...
```

### BuilderABC().check_keys

[Show source in abc_builder.py:54](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L54)

Check if the keys are present and valid.

#### Arguments

- `-` *parameters* - The parameters dictionary to check.

#### Raises

- `-` *ValueError* - If any required key is missing or if trying to set
  an invalid parameter.

#### Examples

```py
builder = Builder()
builder.check_keys({
    "parameter1": 1,
    "parameter2": 2,
})
```

#### Signature

```python
def check_keys(self, parameters: dict[str, Any]): ...
```

### BuilderABC().pre_build_check

[Show source in abc_builder.py:138](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L138)

Check if all required attribute parameters are set before building.

#### Raises

- `-` *ValueError* - If any required parameter is missing.

#### Examples

```py
builder = Builder()
builder.pre_build_check()
```

#### Signature

```python
def pre_build_check(self): ...
```

### BuilderABC().set_parameters

[Show source in abc_builder.py:101](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L101)

Set parameters from a dictionary, handling any '_units' suffix.

#### Arguments

- `-` *parameters* - The parameters dictionary to set.

#### Returns

- [BuilderABC](#builderabc) - This builder object with the set parameters.

#### Raises

- `-` *ValueError* - If any required key is missing.
- `-` *Warning* - If using default units for any parameter.

#### Examples

```py
builder = Builder().set_parameters({
    "parameter1": 1,
    "parameter2": 2,
    "parameter2_units": "K",
})
```

#### Signature

```python
def set_parameters(self, parameters: dict[str, Any]): ...
```


---
# abc_factory.md

# Abc Factory

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Abc Factory

> Auto-generated documentation for [particula.abc_factory](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py) module.

#### Attributes

- `BuilderT` - Define a generic type variable for the strategy type, to get good type hints: TypeVar('BuilderT')


## StrategyFactoryABC

[Show source in abc_factory.py:16](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L16)

Abstract base class for strategy factories.

This class provides a generic interface for creating strategy objects
using builder objects.

#### Methods

- get_builders:
    Returns the mapping of strategy types to builder instances.
- get_strategy:
    Gets the strategy instance for the specified strategy.

#### Examples

```py title="Simple Usage"
my_factory = SomeSpecificFactory()
strategy_instance = my_factory.get_strategy(
    "example_type", {"param": 123}
)
# strategy_instance is now built using "example_type"
```

#### References

- "Factory Method Pattern,"
[Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

#### Signature

```python
class StrategyFactoryABC(ABC, Generic[BuilderT, StrategyT]): ...
```

#### See also

- [BuilderT](#buildert)
- [StrategyT](#strategyt)

### StrategyFactoryABC().get_builders

[Show source in abc_factory.py:43](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L43)

Retrieve a mapping of strategy types to builder instances.

#### Returns

- `dict` - A dictionary that maps strategy type names (str) to
builder instances.

#### Examples

```py title="Coagulation Factory Example"
from particula.coagulation_factory import CoagulationFactory

builders = CoagulationFactory().get_builders()
# Example result:
# {
#     "brownian": BrownianCoagulationBuilder(),
#     "charged": ChargedCoagulationBuilder(),
#     "turbulent_shear": TurbulentShearCoagulationBuilder(),
#     "turbulent_dns": TurbulentDNSCoagulationBuilder(),
#     "combine": CombineCoagulationStrategyBuilder(),
# }
```

#### References

- "Factory Method Pattern,"
[Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

```

#### Signature

```python
@abstractmethod
def get_builders(self) -> Dict[str, BuilderT]: ...
```

#### See also

- [BuilderT](#buildert)

### StrategyFactoryABC().get_strategy

[Show source in abc_factory.py:73](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L73)

Create a strategy instance for the specified type using its
corresponding builder.

#### Arguments

- strategy_type (str): Name of the strategy to build.
- parameters (Dict[str, Any], optional): Dictionary of parameters
  to configure the chosen builder.

#### Returns

- [StrategyT](#abc-factory) - The built strategy object corresponding to the
    specified type.

#### Raises

- `-` *ValueError* - If the `strategy_type` is unknown, or if any required
  parameter is invalid/missing for the chosen builder.

#### Examples

```py title="Strategy Creation Example"
my_factory = SomeStrategyFactory()
my_strategy = my_factory.get_strategy(
    "desired_strategy", {"param_x": 42}
)
# my_strategy is now an instance configured with param_x=42
```

#### References

- "Factory Method Pattern,"
[Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

#### Signature

```python
def get_strategy(
    self, strategy_type: str, parameters: Optional[Dict[str, Any]] = None
) -> StrategyT: ...
```

#### See also

- [StrategyT](#strategyt)


---
# activity_coefficients.md

# Activity Coefficients

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Activity Coefficients

> Auto-generated documentation for [particula.activity.activity_coefficients](https://github.com/uncscode/particula/blob/main/particula/activity/activity_coefficients.py) module.

## bat_activity_coefficients

[Show source in activity_coefficients.py:23](https://github.com/uncscode/particula/blob/main/particula/activity/activity_coefficients.py#L23)

Calculate the activity coefficients for water and organic matter in
organic-water mixtures.

#### Arguments

- molar_mass_ratio : Ratio of the molecular weight of water to the
  molecular weight of organic matter.
- organic_mole_fraction : Molar fraction of organic matter in the
  mixture.
- oxygen2carbon : Oxygen to carbon ratio in the organic compound.
- density : Density of the mixture, in kg/m^3.
- functional_group : Optional functional group(s) of the organic
  compound, if applicable.

#### Returns

- A tuple containing the activity of water, activity
  of organic matter, mass fraction of water, and mass
  fraction of organic matter, gamma_water (activity coefficient),
  and gamma_organic (activity coefficient).

#### Signature

```python
@validate_inputs(
    {
        "molar_mass_ratio": "positive",
        "organic_mole_fraction": "nonnegative",
        "density": "positive",
    }
)
def bat_activity_coefficients(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    organic_mole_fraction: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    functional_group: Optional[Union[str, List[str]]] = None,
) -> Tuple[
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
]: ...
```


---
# bat_blending.md

# Bat Blending

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Bat Blending

> Auto-generated documentation for [particula.activity.bat_blending](https://github.com/uncscode/particula/blob/main/particula/activity/bat_blending.py) module.

## _calculate_blending_weights

[Show source in bat_blending.py:62](https://github.com/uncscode/particula/blob/main/particula/activity/bat_blending.py#L62)

Helper function to calculate blending weights for a single value of
oxygen2carbon.

#### Arguments

- oxygen2carbon : The oxygen to carbon ratio.
- oxygen2carbon_ml : The single-phase oxygen to carbon ratio.

#### Returns

- blending_weights : List of blending weights for the BAT model
    in the low, mid, and high oxygen2carbon regions.

#### Signature

```python
def _calculate_blending_weights(
    oxygen2carbon: float, oxygen2carbon_ml: float
) -> NDArray[np.float64]: ...
```



## bat_blending_weights

[Show source in bat_blending.py:20](https://github.com/uncscode/particula/blob/main/particula/activity/bat_blending.py#L20)

Function to estimate the blending weights for the BAT model.

#### Arguments

- molar_mass_ratio : The molar mass ratio of water to organic
    matter.
- oxygen2carbon : The oxygen to carbon ratio.

#### Returns

- blending_weights : Array of blending weights for the BAT model
    in the low, mid, and high oxygen2carbon regions. The weights
    size is (3,) if oxygen2carbon is a single value, or (n, 3)
    if oxygen2carbon is an array of size n.

#### Signature

```python
@validate_inputs({"molar_mass_ratio": "positive", "oxygen2carbon": "nonnegative"})
def bat_blending_weights(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
) -> NDArray[np.float64]: ...
```


---
# bat_coefficients.md

# Bat Coefficients

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Bat Coefficients

> Auto-generated documentation for [particula.activity.bat_coefficients](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py) module.

## FitValues

[Show source in bat_coefficients.py:17](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L17)

Named tuple for the fit values for the activity model.

#### Signature

```python
class FitValues(NamedTuple): ...
```



## coefficients_c

[Show source in bat_coefficients.py:46](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L46)

Coefficients for activity model, see Gorkowski (2019). equation S1 S2.

#### Arguments

- molar_mass_ratio : The molar mass ratio of water to organic
  matter.
- oxygen2carbon : The oxygen to carbon ratio.
- fit_values : The fit values for the activity model.

#### Returns

- The coefficients for the activity model.

#### Signature

```python
def coefficients_c(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    fit_values: List[float],
) -> NDArray[np.float64]: ...
```


---
# convert_functional_group.md

# Convert Functional Group

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Convert Functional Group

> Auto-generated documentation for [particula.activity.convert_functional_group](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py) module.

## convert_to_oh_equivalent

[Show source in convert_functional_group.py:17](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L17)

Just a pass through now, but will
add the oh equivalent conversion.

#### Arguments

- oxygen2carbon : The oxygen to carbon ratio.
- molar_mass_ratio : The molar mass ratio of water to organic
  matter.
- functional_group : Optional functional group(s) of the organic
  compound, if applicable.

#### Returns

- A tuple containing the converted oxygen to carbon ratio and
  molar mass ratio.

#### Signature

```python
def convert_to_oh_equivalent(
    oxygen2carbon: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    functional_group: Optional[Union[list[str], str]] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...
```


---
# gibbs.md

# Gibbs

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Gibbs

> Auto-generated documentation for [particula.activity.gibbs](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py) module.

## gibbs_free_engery

[Show source in gibbs.py:13](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py#L13)

Calculate the gibbs free energy of the mixture. Ideal and non-ideal.

#### Arguments

- organic_mole_fraction : A numpy array of organic mole
    fractions.
- gibbs_mix : A numpy array of gibbs free energy of mixing.

#### Returns

- gibbs_ideal : The ideal gibbs free energy of mixing.
- gibbs_real : The real gibbs free energy of mixing.

#### Signature

```python
def gibbs_free_engery(
    organic_mole_fraction: NDArray[np.float64], gibbs_mix: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```


---
# gibbs_mixing.md

# Gibbs Mixing

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Gibbs Mixing

> Auto-generated documentation for [particula.activity.gibbs_mixing](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py) module.

## _calculate_gibbs_mix_single

[Show source in gibbs_mixing.py:160](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py#L160)

Calculate Gibbs free energy of mixing for a single set of inputs.

#### Arguments

- molar_mass_ratio : The molar mass ratio of water to organic
  matter.
- organic_mole_fraction : The fraction of organic matter.
- oxygen2carbon : The oxygen to carbon ratio.
- density : The density of the mixture, in kg/m^3.
- weights : Blending weights for the BAT model.

#### Returns

- gibbs_mix : Gibbs energy of mixing (including 1/RT)
- derivative_gibbs : derivative of Gibbs energy with respect to
  mole fraction of organics (includes 1/RT)

#### Signature

```python
def _calculate_gibbs_mix_single(
    molar_mass_ratio: float,
    organic_mole_fraction: float,
    oxygen2carbon: float,
    density: float,
    weights: NDArray[np.float64],
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## gibbs_mix_weight

[Show source in gibbs_mixing.py:99](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py#L99)

Gibbs free energy of mixing, see Gorkowski (2019), with weighted
oxygen2carbon regions. Only can run one compound at a time.

#### Arguments

- molar_mass_ratio : The molar mass ratio of water to organic
  matter.
- organic_mole_fraction : The fraction of organic matter.
- oxygen2carbon : The oxygen to carbon ratio.
- density : The density of the mixture, in kg/m^3.
- functional_group : Optional functional group(s) of the organic
  compound, if applicable.

#### Returns

- gibbs_mix : Gibbs energy of mixing (including 1/RT)
- derivative_gibbs : derivative of Gibbs energy with respect to
  mole fraction of organics (includes 1/RT)

#### Signature

```python
def gibbs_mix_weight(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    organic_mole_fraction: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    functional_group: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## gibbs_of_mixing

[Show source in gibbs_mixing.py:27](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py#L27)

Calculate the Gibbs free energy of mixing for a binary mixture.

#### Arguments

- molar_mass_ratio : The molar mass ratio of water to organic
  matter.
- organic_mole_fraction : The fraction of organic matter.
- oxygen2carbon : The oxygen to carbon ratio.
- density : The density of the mixture, in kg/m^3.
- fit_dict : A dictionary of fit values for the low oxygen2carbon
    region

#### Returns

- A tuple containing the Gibbs free energy of mixing and its
  derivative.

#### Signature

```python
@validate_inputs(
    {
        "molar_mass_ratio": "positive",
        "organic_mole_fraction": "nonnegative",
        "oxygen2carbon": "nonnegative",
        "density": "positive",
    }
)
def gibbs_of_mixing(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    organic_mole_fraction: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    fit_dict: Tuple[str, List[float]],
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]: ...
```


---
# phase_separation.md

# Phase Separation

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Phase Separation

> Auto-generated documentation for [particula.activity.phase_separation](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py) module.

## find_phase_sep_index

[Show source in phase_separation.py:53](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L53)

This function finds phase separation using activity>1 and
inflections in the activity curve data.
In physical systems activity can not be above one and
curve should be monotonic. Or else there will be phase separation.

#### Arguments

- `-` *activity_data* - A array of activity data.

#### Returns

- `dict` - A dictionary containing the following keys:
    - `-` *'phase_sep_activity'* - Phase separation via activity
        (1 if there is phase separation, 0 otherwise)
    - `-` *'phase_sep_curve'* - Phase separation via activity curvature
        (1 if there is phase separation, 0 otherwise)
    - `-` *'index_phase_sep_starts'* - Index where phase separation starts
    - `-` *'index_phase_sep_end'* - Index where phase separation ends

#### Signature

```python
def find_phase_sep_index(activity_data: NDArray[np.float64]) -> dict: ...
```



## find_phase_separation

[Show source in phase_separation.py:143](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L143)

This function checks for phase separation in each activity curve.

#### Arguments

- activity_water (np.array): A numpy array of water activity values.
- activity_org (np.array): A numpy array of organic activity values.

#### Returns

- `dict` - A dictionary containing the following keys:
    - `-` *'phase_sep_check'* - An integer indicating whether phase separation
            is present (1) or not (0).
    - `-` *'lower_seperation_index'* - The index of the lower separation point
            in the activity curve.
    - `-` *'upper_seperation_index'* - The index of the upper separation point in
            the activity curve.
    - `-` *'matching_upper_seperation_index'* - The index where the difference
            between activity_water_beta and match_a_w is greater than 0.
    - `-` *'lower_seperation'* - The value of water activity at the lower
            separation point.
    - `-` *'upper_seperation'* - The value of water activity at the upper
            separation point.
    - `-` *'matching_upper_seperation'* - The value of water activity at the
            matching upper separation point.

#### Signature

```python
def find_phase_separation(
    activity_water: NDArray[np.float64], activity_org: NDArray[np.float64]
) -> dict: ...
```



## organic_water_single_phase

[Show source in phase_separation.py:22](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L22)

Convert the given molar mass ratio (MW water / MW organic) to a
and oxygen2carbon value were above is a single phase with water and below
phase separation is possible.

#### Arguments

- `-` *molar_mass_ratio* - The molar mass ratio with respect to water.

#### Returns

- The single phase cross point.

#### References

- Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
  Relative-humidity-dependent organic aerosol thermodynamics
  Via an efficient reduced-complexity model.
  Atmospheric Chemistry and Physics
  https://doi.org/10.5194/acp-19-13383-2019

#### Signature

```python
def organic_water_single_phase(
    molar_mass_ratio: Union[int, float, list, np.ndarray],
) -> np.ndarray: ...
```



## q_alpha

[Show source in phase_separation.py:240](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L240)

This function calculates the q_alpha value using a squeezed logistic
    function.

#### Arguments

- seperation_activity (np.array): A numpy array of values representing
    the separation activity.
- activities (np.array): A numpy array of activity values.

#### Returns

- `np.array` - The q_alpha value.

#### Notes

- The q_alpha value represents the transfer from
    q_alpha ~0 to q_alpha ~1.
- The function uses a sigmoid curve parameter to calculate the
    q_alpha value.

#### Signature

```python
def q_alpha(
    seperation_activity: NDArray[np.float64], activities: NDArray[np.float64]
) -> np.ndarray: ...
```


---
# ratio.md

# Ratio

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Ratio

> Auto-generated documentation for [particula.activity.ratio](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py) module.

## from_molar_mass_ratio

[Show source in ratio.py:22](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L22)

Convert the given molar mass ratio (MW water / MW organic) to a
molar mass with respect to the other compound.

#### Arguments

- `molar_mass_ratio` *np.array* - The molar mass ratio with respect to water.
- `other_molar_mass` *float, optional* - The molar mass of the other compound.
    Defaults to 18.01528.

#### Returns

- `np.array` - The molar mass of the organic compound.

#### Signature

```python
def from_molar_mass_ratio(molar_mass_ratio, other_molar_mass=18.01528): ...
```



## to_molar_mass_ratio

[Show source in ratio.py:4](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L4)

Convert the given molar mass to a molar mass ratio with respect to water.
(MW water / MW organic)

#### Arguments

- `molar_mass` *np.array* - The molar mass of the organic compound.
- `other_molar_mass` *float, optional* - The molar mass of the other compound.
    Defaults to 18.01528.

#### Returns

- `np.array` - The molar mass ratio with respect to water.

#### Signature

```python
def to_molar_mass_ratio(molar_mass, other_molar_mass=18.01528): ...
```


---
# species_density.md

# Species Density

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Species Density

> Auto-generated documentation for [particula.activity.species_density](https://github.com/uncscode/particula/blob/main/particula/activity/species_density.py) module.

## organic_array

[Show source in species_density.py:93](https://github.com/uncscode/particula/blob/main/particula/activity/species_density.py#L93)

Get densities for an array.

#### Signature

```python
def organic_array(
    molar_mass,
    oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
    mass_ratio_convert=False,
): ...
```



## organic_density_estimate

[Show source in species_density.py:12](https://github.com/uncscode/particula/blob/main/particula/activity/species_density.py#L12)

Function to estimate the density of organic compounds based on the simple
model by Girolami (1994). The input parameters include molar mass, O:C
and H:C ratios. If the H:C ratio is unknown at input, enter a negative
value. The actual H:C will then be estimated based on an initial assumption
of H:C = 2. The model also estimates the number of carbon atoms per
molecular structure based on molar mass, O:C, and H:C.
The density is then approximated by the formula of Girolami.

Reference:
Girolami, G. S.: A Simple 'Back of the Envelope' Method for Estimating
the Densities and Molecular Volumes of Liquids and Solids,
J. Chem. Educ., 71(11), 962, doi:10.1021/ed071p962, 1994.

#### Arguments

- `molar_mass(float)` - Molar mass.
- `oxygen2carbon` *float* - O:C ratio.
- `hydrogen2carbon` *float* - H:C ratio. If unknown, provide a negative
    value.
- `nitrogen2carbon` *float, optional* - N:C ratio. Defaults to None.

#### Returns

- `densityEst` *float* - Estimated density in g/cm^3.

#### Signature

```python
def organic_density_estimate(
    molar_mass,
    oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
    mass_ratio_convert=False,
): ...
```


---
# water_activity.md

# Water Activity

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Water Activity

> Auto-generated documentation for [particula.activity.water_activity](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py) module.

## biphasic_water_activity_point

[Show source in water_activity.py:24](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L24)

This function computes the biphasic to single phase
water activity (RH*100).

#### Arguments

- oxygen2carbon : The oxygen to carbon ratio.
- hydrogen2carbon : The hydrogen to carbon ratio.
- molar_mass_ratio : The molar mass ratio of water to organic
  matter.
- functional_group : Optional functional group(s) of the organic
  compound, if applicable.

#### Returns

- The RH cross point array.

#### Signature

```python
def biphasic_water_activity_point(
    oxygen2carbon: Union[float, NDArray[np.float64]],
    hydrogen2carbon: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    functional_group: Optional[Union[list[str], str]] = None,
) -> np.ndarray: ...
```



## fixed_water_activity

[Show source in water_activity.py:93](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L93)

Calculate the activity coefficients of water and organic matter in
organic-water mixtures.

This function assumes a fixed water activity value (e.g., RH = 75%
corresponds to 0.75 water activity in equilibrium).
It calculates the activity coefficients for different phases and
determines phase separations if they occur.

#### Arguments

- water_activity : An array of water activity values.
- molar_mass_ratio : Array of molar mass ratios of the components.
- oxygen2carbon : Array of oxygen-to-carbon ratios.
- density : Array of densities of the mixture, in kg/m^3.

#### Returns

- A tuple containing the activity coefficients for alpha and beta
  phases, and the q_alpha (phase separation) value.
  If no phase separation occurs, the beta phase values are None.

#### Signature

```python
def fixed_water_activity(
    water_activity: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Tuple[
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
]: ...
```


---
# aerosol.md

# Aerosol

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Aerosol

> Auto-generated documentation for [particula.aerosol](https://github.com/uncscode/particula/blob/main/particula/aerosol.py) module.

## Aerosol

[Show source in aerosol.py:14](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L14)

Represents a collection of Gas and Particle objects forming an aerosol
environment.

This class allows for the representation and manipulation of an aerosol,
which consists of various gases in an Atmosphere object and one
ParticleRepresentation object.

#### Attributes

- atmosphere : The atmosphere containing the gases.
- particles : The particle Representation object.

#### Methods

- `-` *iterate_gas* - Returns an iterator over the gas species in atmosphere.
- `-` *replace_atmosphere* - Replaces the current atmosphere with a new one.
- `-` *replace_particle* - Replaces a particle in the aerosol with a new one.

#### Examples

```py title="Creating an Aerosol"
aerosol_instance = Aerosol(atmosphere, particles)
print(aerosol_instance)
```

```py title="Iterating over the Aerosol"
aerosol_instance = Aerosol(atmosphere, particles)
for gas in aerosol_instance.iterate_gas():
    print(gas)
```

#### Signature

```python
class Aerosol:
    def __init__(self, atmosphere: Atmosphere, particles: ParticleRepresentation): ...
```

#### See also

- [Atmosphere](gas/atmosphere.md#atmosphere)
- [ParticleRepresentation](particles/representation.md#particlerepresentation)

### Aerosol().__str__

[Show source in aerosol.py:61](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L61)

Provide a string representation of the aerosol.

#### Returns

- str : A string summarizing the atmosphere and each particle.

```py
aerosol_instance = Aerosol(atmosphere, particles)
print(aerosol_instance)
```

#### Signature

```python
def __str__(self) -> str: ...
```

### Aerosol().iterate_gas

[Show source in aerosol.py:76](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L76)

Return an iterator over the gas species in the atmosphere.

#### Returns

- Iterator[GasSpecies] : An iterator over gas species objects.

#### Examples

```py title="Iterating over aerosol gas"
aerosol_instance = Aerosol(atmosphere, particles)
for gas in aerosol_instance.iterate_gas():
    print(gas)
```

#### Signature

```python
def iterate_gas(self) -> Iterator[GasSpecies]: ...
```

#### See also

- [GasSpecies](gas/species.md#gasspecies)

### Aerosol().replace_atmosphere

[Show source in aerosol.py:92](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L92)

Replace the current atmosphere with a new Atmosphere instance.

#### Arguments

- atmosphere : The new Atmosphere to assign.

#### Examples

```py title="Replacing the Atmosphere in the Aerosol"
aerosol_instance = Aerosol(atmosphere, particles)
new_atmosphere = Atmosphere()
aerosol_instance.replace_atmosphere(new_atmosphere)
```

#### Signature

```python
def replace_atmosphere(self, atmosphere: Atmosphere): ...
```

#### See also

- [Atmosphere](gas/atmosphere.md#atmosphere)

### Aerosol().replace_particles

[Show source in aerosol.py:108](https://github.com/uncscode/particula/blob/main/particula/aerosol.py#L108)

Replace a particles in the aerosol with a new ParticleRepresentation.

#### Arguments

- particle : The new ParticleRepresentation to assign.

#### Examples

```py title="Replacing a Particle in the Aerosol"
aerosol_instance = Aerosol(atmosphere, particles)
new_particle = ParticleRepresentation()
aerosol_instance.replace_particles(new_particle)
```

#### Signature

```python
def replace_particles(self, particles: ParticleRepresentation): ...
```

#### See also

- [ParticleRepresentation](particles/representation.md#particlerepresentation)


---
# builder_mixin.md

# Builder Mixin

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Builder Mixin

> Auto-generated documentation for [particula.builder_mixin](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py) module.

## BuilderChargeMixin

[Show source in builder_mixin.py:233](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L233)

Mixin class for setting a particle's charge.

This class provides a method to assign charge in terms of number of
elemental charges (dimensionless), ignoring units.

#### Attributes

- charge : The assigned charge.

#### Methods

- `-` *set_charge* - Assign the particle's charge.

#### References

- No references available yet.

#### Signature

```python
class BuilderChargeMixin:
    def __init__(self): ...
```

### BuilderChargeMixin().set_charge

[Show source in builder_mixin.py:253](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L253)

Set the number of elemental charges on the particle.

#### Arguments

- charge : Numeric value of the charge.
- charge_units : Optional; if provided, a warning is logged
    and ignored.

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_charge(10)
# charge is now 10 elementary charges
```

#### Signature

```python
def set_charge(
    self, charge: Union[float, NDArray[np.float64]], charge_units: Optional[str] = None
): ...
```



## BuilderConcentrationMixin

[Show source in builder_mixin.py:176](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L176)

Mixin class for setting concentration in a mixture.

This class provides a method to assign a particle or species concentration
in kg/m^3 by default, optionally converting from other units.

#### Attributes

- concentration : The concentration in the default units.
- default_units : The default concentration units (e.g., "kg/m^3").

#### Methods

- `-` *set_concentration* - Assign the concentration, converting units
    as needed.

#### Examples

```py title="Example usage"
builder = MyBuilderClass(default_units="g/m^3")
builder.set_concentration(500, "g/m^3")
```

#### Signature

```python
class BuilderConcentrationMixin:
    def __init__(self, default_units: str = "kg/m^3"): ...
```

### BuilderConcentrationMixin().set_concentration

[Show source in builder_mixin.py:202](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L202)

Set the concentration in the mixture.

#### Arguments

- concentration : Concentration value.
- concentration_units : Units of the provided concentration.

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_concentration(0.5, "kg/m^3")
# stored as 0.5 in the default_units
```

#### Signature

```python
@validate_inputs({"concentration": "nonnegative"})
def set_concentration(
    self, concentration: Union[float, NDArray[np.float64]], concentration_units: str
): ...
```



## BuilderDensityMixin

[Show source in builder_mixin.py:17](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L17)

Mixin class for setting density and density_units.

This class provides a method to assign a particle's density in kg/m^3,
optionally converting from other units.

#### Attributes

- density : Stores the density in kg/m^3 after conversion.

#### Methods

- `-` *set_density* - Assign the density attribute, converting from given
    units to kg/m^3.

#### Examples

```py title="Setting particle density"
builder = MyBuilderClass()
builder.set_density(1000, "g/m^3")
# density is now 1.0 kg/m^3
```

#### Signature

```python
class BuilderDensityMixin:
    def __init__(self): ...
```

### BuilderDensityMixin().set_density

[Show source in builder_mixin.py:42](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L42)

Set the density of the particle in kg/m^3.

#### Arguments

- density : Density value.
- density_units : Units of the provided density.
    Default is "kg/m^3".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_density(1000, "g/m^3")
# density is now 1.0 kg/m^3
```

#### Signature

```python
@validate_inputs({"density": "positive"})
def set_density(
    self, density: Union[float, NDArray[np.float64]], density_units: str
): ...
```



## BuilderLognormalMixin

[Show source in builder_mixin.py:517](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L517)

Mixin class for setting lognormal distribution parameters.

This class provides methods to assign and manage lognormal distribution
parameters for particle radius, including the mode, geometric standard
deviation, and number concentration.

#### Attributes

- mode : Array of modes in meters.
- number_concentration : Number concentration in 1/m^3.
- geometric_standard_deviation : The dimensionless geometric std. dev.

#### Methods

- `-` *set_mode* - Assign the modal radius.
- `-` *set_geometric_standard_deviation* - Assign the geometric std. dev.
    (ignored units).
- `-` *set_number_concentration* - Assign the number concentration in 1/m^3.

#### Signature

```python
class BuilderLognormalMixin:
    def __init__(self): ...
```

### BuilderLognormalMixin().set_geometric_standard_deviation

[Show source in builder_mixin.py:570](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L570)

Set the geometric standard deviation for the lognormal distribution.

#### Arguments

- geometric_standard_deviation : Dimensionless geometric std. dev.
- geometric_standard_deviation_units : Ignored
    (for interface consistency).

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_geometric_standard_deviation(np.array([1.5, 2.0]))
# geometric std dev is now [1.5, 2.0]
```

#### Signature

```python
@validate_inputs({"geometric_standard_deviation": "positive"})
def set_geometric_standard_deviation(
    self,
    geometric_standard_deviation: NDArray[np.float64],
    geometric_standard_deviation_units: Optional[str] = None,
): ...
```

### BuilderLognormalMixin().set_mode

[Show source in builder_mixin.py:542](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L542)

Set the mode for the lognormal distribution in meters.

#### Arguments

- mode : Array of modal radius values.
- mode_units : Units of the provided mode. Default is "m".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_mode(np.array([1e-8, 2e-8]), "m")
# modes are now [1e-8, 2e-8] m
```

#### Signature

```python
@validate_inputs({"mode": "positive"})
def set_mode(self, mode: NDArray[np.float64], mode_units: str): ...
```

### BuilderLognormalMixin().set_number_concentration

[Show source in builder_mixin.py:598](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L598)

Set the number concentration for the lognormal distribution in 1/m^3.

#### Arguments

- number_concentration : Array of number concentration values.
- number_concentration_units : Units of the concentration,
    must be "1/m^3" or equivalent.

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_number_concentration(np.array([1e6, 5e5]), "m^-3")
# stored as [1e6, 5e5] 1/m^3
```

#### Signature

```python
@validate_inputs({"number_concentration": "positive"})
def set_number_concentration(
    self, number_concentration: NDArray[np.float64], number_concentration_units: str
): ...
```



## BuilderMassMixin

[Show source in builder_mixin.py:281](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L281)

Mixin class for setting particle mass in kg.

This class provides a method to assign mass in kg, optionally converting
from other units.

#### Attributes

- mass : The mass of the particle in kg.

#### Methods

- `-` *set_mass* - Assign the mass, converting from specified units.

#### Signature

```python
class BuilderMassMixin:
    def __init__(self): ...
```

### BuilderMassMixin().set_mass

[Show source in builder_mixin.py:298](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L298)

Set the mass of the particle in kg.

#### Arguments

- mass : Numeric mass value.
- mass_units : Units of the provided mass. Default is "kg".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_mass(1.0, "g")
# mass is now 0.001 kg
```

#### Signature

```python
@validate_inputs({"mass": "nonnegative"})
def set_mass(self, mass: Union[float, NDArray[np.float64]], mass_units: str): ...
```



## BuilderMolarMassMixin

[Show source in builder_mixin.py:124](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L124)

Mixin class for setting molar_mass and molar_mass_units.

This class provides a method to assign a particle's molar mass in kg/mol,
optionally converting from other units.

#### Attributes

- molar_mass : Stores the molar mass in kg/mol.

#### Methods

- `-` *set_molar_mass* - Assign the molar_mass, converting units as necessary.

#### References

- No references available yet.

#### Signature

```python
class BuilderMolarMassMixin:
    def __init__(self): ...
```

### BuilderMolarMassMixin().set_molar_mass

[Show source in builder_mixin.py:144](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L144)

Set the molar mass of the particle in kg/mol.

#### Arguments

- molar_mass : Molar mass value.
- molar_mass_units : Units of the provided molar mass.
    Default is "kg/mol".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_molar_mass(18, "g/mol")
# molar_mass is now 0.018 kg/mol
```

#### Signature

```python
@validate_inputs({"molar_mass": "positive"})
def set_molar_mass(
    self, molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str
): ...
```



## BuilderParticleResolvedCountMixin

[Show source in builder_mixin.py:630](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L630)

Mixin class for setting a particle-resolved count.

This class provides a method to define how many individual particles
should be resolved in a simulation or model.

#### Attributes

- particle_resolved_count : The number of particles to resolve.

#### Methods

- `-` *set_particle_resolved_count* - Assign the particle-resolved count.

#### Signature

```python
class BuilderParticleResolvedCountMixin:
    def __init__(self): ...
```

### BuilderParticleResolvedCountMixin().set_particle_resolved_count

[Show source in builder_mixin.py:647](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L647)

Set the number of particles to resolve.

#### Arguments

- particle_resolved_count : Positive integer count of particles.
- particle_resolved_count_units : Ignored, for interface
    consistency.

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_particle_resolved_count(1000)
```

#### Signature

```python
@validate_inputs({"particle_resolved_count": "positive"})
def set_particle_resolved_count(
    self,
    particle_resolved_count: int,
    particle_resolved_count_units: Optional[str] = None,
): ...
```



## BuilderPressureMixin

[Show source in builder_mixin.py:470](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L470)

Mixin class for setting total pressure in Pa.

This class provides a method to assign the total gas mixture pressure
in pascals, optionally converting from units like 'kPa', 'MPa', 'psi',
'bar', or 'atm'.

#### Attributes

- pressure : The total pressure in Pa.

#### Methods

- `-` *set_pressure* - Assign the pressure, converting units as needed.

#### Signature

```python
class BuilderPressureMixin:
    def __init__(self): ...
```

### BuilderPressureMixin().set_pressure

[Show source in builder_mixin.py:488](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L488)

Set the total pressure of the atmosphere.

#### Arguments

- pressure : Numeric pressure value.
- pressure_units : Units of the given pressure. Default is "Pa".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_pressure(1.0, "bar")
# pressure is now 1e5 Pa
```

#### Signature

```python
@validate_inputs({"pressure": "nonnegative"})
def set_pressure(
    self, pressure: Union[float, NDArray[np.float64]], pressure_units: str
): ...
```



## BuilderRadiusMixin

[Show source in builder_mixin.py:373](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L373)

Mixin class for setting a particle's radius in meters.

This class provides a method to assign radius in meters,
optionally converting from other units.

#### Attributes

- radius : The radius in meters.

#### Methods

- `-` *set_radius* - Assign the radius, converting units as needed.

#### Signature

```python
class BuilderRadiusMixin:
    def __init__(self): ...
```

### BuilderRadiusMixin().set_radius

[Show source in builder_mixin.py:390](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L390)

Set the radius of the particle in meters.

#### Arguments

- radius : Numeric radius value.
- radius_units : Units of the provided radius. Default is "m".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_radius(1.0, "um")
# radius is now 1e-6 m
```

#### Signature

```python
@validate_inputs({"radius": "nonnegative"})
def set_radius(self, radius: Union[float, NDArray[np.float64]], radius_units: str): ...
```



## BuilderSurfaceTensionMixin

[Show source in builder_mixin.py:72](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L72)

Mixin class for setting surface_tension.

This class provides a method to assign a particle's surface tension,
in N/m units, optionally converting from other units.

#### Attributes

- surface_tension : Stores the surface tension in N/m after conversion.

#### Methods

- `-` *set_surface_tension* - Assign the surface_tension, converting from
    other units as needed.

#### References

- No references available yet.

#### Signature

```python
class BuilderSurfaceTensionMixin:
    def __init__(self): ...
```

### BuilderSurfaceTensionMixin().set_surface_tension

[Show source in builder_mixin.py:93](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L93)

Set the surface tension of the particle in N/m.

#### Arguments

- surface_tension : Surface tension value.
- surface_tension_units : Units of the provided surface tension.
    Default is "N/m".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_surface_tension(0.072, "N/m")
```

#### Signature

```python
@validate_inputs({"surface_tension": "positive"})
def set_surface_tension(
    self, surface_tension: Union[float, NDArray[np.float64]], surface_tension_units: str
): ...
```



## BuilderTemperatureMixin

[Show source in builder_mixin.py:419](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L419)

Mixin class for setting temperature in Kelvin.

This class provides a method to assign temperature in Kelvin,
optionally converting from specified units such as 'degC', 'degF',
'degR', or 'K'.

#### Attributes

- temperature : The temperature in Kelvin.

#### Methods

- `-` *set_temperature* - Assign the temperature, converting units as needed.

#### Signature

```python
class BuilderTemperatureMixin:
    def __init__(self): ...
```

### BuilderTemperatureMixin().set_temperature

[Show source in builder_mixin.py:437](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L437)

Set the temperature of the atmosphere in Kelvin.

#### Arguments

- temperature : Numeric temperature value.
- temperature_units : Units of the given temperature.
    Defaults to "K". Accepts "degC", "degF", "degR", or "K".

#### Returns

- self : The class instance for method chaining.

#### Raises

- ValueError : If the converted temperature is below absolute zero.

#### Examples

```py title="Setting temperature"
builder.set_temperature(25, "degC")
# temperature is now 298.15 K
```

#### Signature

```python
@validate_inputs({"temperature": "finite"})
def set_temperature(self, temperature: float, temperature_units: str = "K"): ...
```



## BuilderVolumeMixin

[Show source in builder_mixin.py:327](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L327)

Mixin class for setting volume in m^3.

This class provides a method to assign volume in m^3,
optionally converting from other units.

#### Attributes

- volume : The volume in m^3.

#### Methods

- `-` *set_volume* - Assign the volume, converting units as needed.

#### Signature

```python
class BuilderVolumeMixin:
    def __init__(self): ...
```

### BuilderVolumeMixin().set_volume

[Show source in builder_mixin.py:344](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L344)

Set the volume in m^3.

#### Arguments

- volume : Volume value.
- volume_units : Units of the provided volume. Default is "m^3".

#### Returns

- self : The class instance for method chaining.

#### Examples

```py
builder.set_volume(1.0, "L")
# volume is now 0.001 m^3
```

#### Signature

```python
@validate_inputs({"volume": "nonnegative"})
def set_volume(self, volume: Union[float, NDArray[np.float64]], volume_units: str): ...
```


---
# brownian_kernel.md

# Brownian Kernel

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Brownian Kernel

> Auto-generated documentation for [particula.dynamics.coagulation.brownian_kernel](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py) module.

## _brownian_diffusivity

[Show source in brownian_kernel.py:255](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L255)

Calculate the diffusivity of particles due to Brownian motion.

This function calculates the diffusivity of particles due to Brownian
motion, which is essentially the scaled aerodynamic mobility of the
particles. The equation used is:

- D = k × T × B
    - D is the diffusivity of the particles [m²/s].
    - k is the Boltzmann constant [J/K].
    - T is the temperature of the air [K].
    - B is the aerodynamic mobility of the particles [m²/s].

#### Arguments

- temperature : The temperature of the air [K].
- aerodynamic_mobility : The aerodynamic mobility of the particles
  [m²/s].

#### Returns

- The diffusivity of the particles due to Brownian motion [m²/s].

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
  Coefficient K12.

#### Signature

```python
def _brownian_diffusivity(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## _g_collection_term

[Show source in brownian_kernel.py:216](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L216)

Calculate the `g` collection term for Brownian coagulation.

This function calculates the `g` collection term for Brownian
coagulation, defined as the ratio of the mean free path of the particles
to the radius of the particles. The equation used is:

- g = ((2r + λ)³ - (4r² + λ²)^(3/2)) / (6rλ) - 2r
    - g is the collection term for Brownian coagulation [dimensionless].
    - λ is the mean free path of the particles [m].
    - r is the radius of the particles [m].

#### Arguments

- mean_free_path_particle : The mean free path of the particles [m].
- particle_radius : The radius of the particles [m].

#### Returns

- The collection term for Brownian coagulation [dimensionless].

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
  Coefficient K12.

#### Notes

The np.sqrt(2) term appears to be an error in the text, as the term is
not used in the second edition of the book. When it is used, the values
are too small, by about 2x.

#### Signature

```python
def _g_collection_term(
    mean_free_path_particle: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## _mean_free_path_l

[Show source in brownian_kernel.py:181](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L181)

Calculate the mean free path of particles for coagulation.

Calculate the mean free path of particles for coagulation.

This function calculates the mean free path of particles, defined for
Brownian coagulation as the ratio of the diffusivity of the particles
to their mean thermal speed. This parameter is crucial for understanding
particle dynamics in a fluid. The equation used is:

- λ = (8 × D) / (π × v)
    - λ is the mean free path of the particles [m].
    - D is the diffusivity of the particles [m²/s].
    - v is the mean thermal speed of the particles [m/s].

#### Arguments

- diffusivity_particle : The diffusivity of the particles [m²/s].
- mean_thermal_speed_particle : The mean thermal speed of the
  particles [m/s].

#### Returns

- The mean free path of the particles [m].

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
  Coefficient K12.

#### Signature

```python
def _mean_free_path_l(
    diffusivity_particle: Union[float, NDArray[np.float64]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_brownian_kernel

[Show source in brownian_kernel.py:17](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L17)

Calculate the Brownian coagulation kernel for aerosol particles.

This function computes the Brownian coagulation kernel, which is
defined as the product of the diffusivity of the particles, the
collection term `g`, and the radius of the particles. The equation
used is:

- K = (4π × D × r) / (r / (r + g) + 4D / (r × v × α))
    - K is the Brownian coagulation kernel [m³/s].
    - D is the diffusivity of the particles [m²/s].
    - r is the radius of the particles [m].
    - g is the collection term for Brownian coagulation [dimensionless].
    - v is the mean thermal speed of the particles [m/s].
    - α is the collision efficiency of the particles [dimensionless].

#### Arguments

- particle_radius : The radius of the particles [m].
- diffusivity_particle : The diffusivity of the particles [m²/s].
- g_collection_term_particle : The collection term for Brownian
  coagulation [dimensionless].
- mean_thermal_speed_particle : The mean thermal speed of the
  particles [m/s].
- alpha_collision_efficiency : The collision efficiency of the
  particles [dimensionless].

#### Returns

- Square matrix of Brownian coagulation kernel for aerosol particles
  [m³/s].

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
  Coefficient K12 (with alpha collision efficiency term 13.56).

#### Signature

```python
def get_brownian_kernel(
    particle_radius: Union[float, NDArray[np.float64]],
    diffusivity_particle: Union[float, NDArray[np.float64]],
    g_collection_term_particle: Union[float, NDArray[np.float64]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float64]],
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_brownian_kernel_via_system_state

[Show source in brownian_kernel.py:98](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L98)

Calculate the Brownian coagulation kernel using system state parameters.

This function calculates the Brownian coagulation kernel for aerosol
particles by determining the necessary intermediate properties such as
particle diffusivity and mean thermal speed. The equation used is:

- K = (4π × D × r) / (r / (r + g) + 4D / (r × v × α))
    - K is the Brownian coagulation kernel [m³/s].
    - D is the diffusivity of the particles [m²/s].
    - r is the radius of the particles [m].
    - g is the collection term for Brownian coagulation [dimensionless].
    - v is the mean thermal speed of the particles [m/s].
    - α is the collision efficiency of the particles [dimensionless].

#### Arguments

- particle_radius : The radius of the particles [m].
- particle_mass : The mass of the particles [kg].
- temperature : The temperature of the air [K].
- pressure : The pressure of the air [Pa].
- alpha_collision_efficiency : The collision efficiency of the
  particles [dimensionless].

#### Returns

- Square matrix of Brownian coagulation kernel for aerosol particles
  [m³/s].

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
  Coefficient K12.

#### Signature

```python
def get_brownian_kernel_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# charged_dimensional_kernel.md

# Charged Dimensional Kernel

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Charged Dimensional Kernel

> Auto-generated documentation for [particula.dynamics.coagulation.charged_dimensional_kernel](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py) module.

## _system_state_properties

[Show source in charged_dimensional_kernel.py:34](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L34)

Get the system state properties for charged particles.

#### Arguments

- particle_radius : The radius of the particles [m].
- particle_mass : The mass of the particles [kg].
- particle_charge : The charge of the particles [C].
- temperature : The temperature of the system [K].
- pressure : The pressure of the system [Pa].

#### Returns

- coulomb_potential_ratio : The Coulomb potential ratio
    [dimensionless].
- diffusive_knudsen : The diffusive knudsen number [dimensionless].
- sum_of_radii : The sum of the radii of the particles [m].
- reduced_mass : The reduced mass of the particles [kg].
- reduced_friction_factor : The reduced friction factor of the
    particles [dimensionless].

- `Examples` - (This is an internal helper and typically not called directly.)

#### Signature

```python
def _system_state_properties(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```



## get_coulomb_kernel_chahl2019_via_system_state

[Show source in charged_dimensional_kernel.py:389](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L389)

The dimensioned coagulation kernel via system state
using Chahl (2019).

#### Arguments

- particle_radius : The radius of the particles [m].
- particle_mass : The mass of the particles [kg].
- particle_charge : The charge of the particles [C].
- temperature : The temperature of the system [K].
- pressure : The pressure of the system [Pa].

#### Returns

- The dimensioned coagulation kernel, as a square matrix,
  [m^3/s].

#### Examples

``` py title="Example Usage"
import particula as par
kernel_chahl = ()
    par.dynamics.get_coulomb_kernel_chahl2019_via_system_state(
    p_radius, p_mass, p_charge, 298.15, 101325
    )
)
print(kernel_chahl)
```

#### References

- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
    molecular regime Coulombic collisions in aerosols and dusty plasmas.
    Aerosol Science and Technology, 53(8), 933-957.
    https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
def get_coulomb_kernel_chahl2019_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```



## get_coulomb_kernel_dyachkov2007_via_system_state

[Show source in charged_dimensional_kernel.py:196](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L196)

The dimensioned coagulation kernel via system state using Dyachkov (2007).

#### Arguments

- particle_radius : The radius of the particles [m].
- particle_mass : The mass of the particles [kg].
- particle_charge : The charge of the particles [C].
- temperature : The temperature of the system [K].
- pressure : The pressure of the system [Pa].

#### Returns

- The dimensioned coagulation kernel, as a square matrix,
  [m^3/s].

#### Examples

``` py title="Example Usage"
import particula as par
kernel_dyachkov = (
    par.dynamics.get_coulomb_kernel_dyachkov2007_via_system_state(
        p_radius, p_mass, p_charge, 298.15, 101325
    )
)
print(kernel_dyachkov.shape)
```

#### References

- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

#### Signature

```python
def get_coulomb_kernel_dyachkov2007_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```



## get_coulomb_kernel_gatti2008_via_system_state

[Show source in charged_dimensional_kernel.py:260](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L260)

The dimensioned coagulation kernel via system state using Gatti (2008).

#### Arguments

- particle_radius : The radius of the particles [m].
- particle_mass : The mass of the particles [kg].
- particle_charge : The charge of the particles [C].
- temperature : The temperature of the system [K].
- pressure : The pressure of the system [Pa].

#### Returns

- The dimensioned coagulation kernel, as a square matrix,
  [m^3/s].

#### Examples

``` py title="Example Usage"
import particula as par
kernel_gatti = (
    par.dynamics.get_coulomb_kernel_gatti2008_via_system_state(
        p_radius, p_mass, p_charge, 298.15, 101325
    )
)
print(kernel_gatti)
```

#### References

- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
  charging in plasmas over a wide range of collisionality. Physical Review
  E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
  https://doi.org/10.1103/PhysRevE.78.046402

#### Signature

```python
def get_coulomb_kernel_gatti2008_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```



## get_coulomb_kernel_gopalakrishnan2012_via_system_state

[Show source in charged_dimensional_kernel.py:324](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L324)

The dimensioned coagulation kernel via system state
using Gopalakrishnan (2012).

#### Arguments

- particle_radius : The radius of the particles [m].
- particle_mass : The mass of the particles [kg].
- particle_charge : The charge of the particles [C].
- temperature : The temperature of the system [K].
- pressure : The pressure of the system [Pa].

#### Returns

- The dimensioned coagulation kernel, as a square matrix,
  [m^3/s].

#### Examples

``` py title="Example Usage"
import particula as par
kernel_gopal = (
    par.dyanmics.get_coulomb_kernel_gopalakrishnan2012_via_system_state(
        p_radius, p_mass, p_charge, 298.15, 101325
    )
)
# kernel_gopal is an N×N matrix [m^3/s]
```

#### References

- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
  in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear
  and Soft Matter Physics, 85(2).
  https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
def get_coulomb_kernel_gopalakrishnan2012_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```



## get_hard_sphere_kernel_via_system_state

[Show source in charged_dimensional_kernel.py:122](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L122)

The hard sphere dimensioned coagulation kernel via system state.

For the hard sphere kernel, the dimensionless kernel is computed
internally based on the diffusive Knudsen number, then converted
to the dimensioned form using the system state properties
(particle radius, mass, charge, temperature, pressure).
coulomb potential ratio, sum of radii, reduced mass...etc are all
calculated from the system state properties (temperature, pressure, etc.).
These are used to calculate the dimensionless kernel, which is then
converted to the dimensioned kernel.

#### Arguments

- particle_radius : The radius of the particles [m].
- particle_mass : The mass of the particles [kg].
- particle_charge : The charge of the particles [C].
- temperature : The temperature of the system [K].
- pressure : The pressure of the system [Pa].

#### Returns

- The dimensioned coagulation kernel, as a square matrix, of all
    particle-particle interactions [m^3/s].

#### Examples

``` py title="Example Usage"
kernel_matrix = get_hard_sphere_kernel_via_system_state(
    p_radius, p_mass, p_charge, 298.15, 101325
)
# kernel_matrix is an N×N array of rates
```

#### References

- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

#### Signature

```python
def get_hard_sphere_kernel_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```


---
# charged_dimensionless_kernel.md

# Charged Dimensionless Kernel

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Charged Dimensionless Kernel

> Auto-generated documentation for [particula.dynamics.coagulation.charged_dimensionless_kernel](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py) module.

## get_coulomb_kernel_chahl2019

[Show source in charged_dimensionless_kernel.py:291](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L291)

Chahl and Gopalakrishnan (2019) approximation for the dimensionless
coagulation kernel.

This function accounts for the Coulomb potential between particles using
the Chahl and Gopalakrishnan (2019) approximation.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number (K_nD)
    [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio (phi_E)
  [dimensionless].

#### Returns

- The dimensionless coagulation kernel (H) [dimensionless].

#### References

- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
  molecular regime Coulombic collisions in aerosols and dusty plasmas.
  Aerosol Science and Technology, 53(8), 933-957.
  https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
def get_coulomb_kernel_chahl2019(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coulomb_kernel_dyachkov2007

[Show source in charged_dimensionless_kernel.py:112](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L112)

Dyachkov et al. (2007) approximation for the dimensionless coagulation
kernel.

This function accounts for the Coulomb potential between particles using
the Dyachkov et al. (2007) approximation.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number (K_nD)
    [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio (phi_E)
  [dimensionless].

#### Returns

- The dimensionless coagulation kernel (H) [dimensionless].

#### References

- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

#### Signature

```python
def get_coulomb_kernel_dyachkov2007(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coulomb_kernel_gatti2008

[Show source in charged_dimensionless_kernel.py:173](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L173)

Gatti et al. (2008) approximation for the dimensionless coagulation
kernel.

This function accounts for the Coulomb potential between particles using
the Gatti et al. (2008) approximation.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number (K_nD)
    [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio (phi_E)
  [dimensionless].

#### Returns

- The dimensionless coagulation kernel (H) [dimensionless].

#### References

- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
  charging in plasmas over a wide range of collisionality. Physical Review
  E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
  https://doi.org/10.1103/PhysRevE.78.046402

#### Signature

```python
def get_coulomb_kernel_gatti2008(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coulomb_kernel_gopalakrishnan2012

[Show source in charged_dimensionless_kernel.py:243](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L243)

Gopalakrishnan and Hogan (2012) approximation for the dimensionless
coagulation kernel.

This function accounts for the Coulomb potential between particles using
the Gopalakrishnan and Hogan (2012) approximation.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number (K_nD)
    [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio (phi_E)
  [dimensionless].

#### Returns

- The dimensionless coagulation kernel (H) [dimensionless].

#### References

- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
  in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear
  and Soft Matter Physics, 85(2).
  https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
def get_coulomb_kernel_gopalakrishnan2012(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_dimensional_kernel

[Show source in charged_dimensionless_kernel.py:13](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L13)

Calculate the dimensioned coagulation kernel for each particle pair.

This function computes the dimensioned coagulation kernel from the
dimensionless coagulation kernel and the reduced quantities. All inputs
are square matrices, representing all particle-particle interactions.

#### Arguments

- dimensionless_kernel : The dimensionless coagulation kernel (H)
  [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio [dimensionless].
- sum_of_radii : The sum of the radii of the particles [m].
- reduced_mass : The reduced mass of the particles [kg].
- reduced_friction_factor : The reduced friction factor of the
  particles [dimensionless].

#### Returns

- The dimensioned coagulation kernel, as a square matrix, of all
  particle-particle interactions [m³/s].

#### References

- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
  molecular regime Coulombic collisions in aerosols and dusty plasmas.
  Aerosol Science and Technology, 53(8), 933-957.
  https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
def get_dimensional_kernel(
    dimensionless_kernel: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
    sum_of_radii: NDArray[np.float64],
    reduced_mass: NDArray[np.float64],
    reduced_friction_factor: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_hard_sphere_kernel

[Show source in charged_dimensionless_kernel.py:61](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L61)

Hard sphere approximation for the dimensionless coagulation kernel.

This function provides a hard sphere approximation for the dimensionless
coagulation kernel based on the diffusive Knudsen number.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number (K_nD)
  [dimensionless].

#### Returns

- The dimensionless coagulation kernel (H) [dimensionless].

#### Raises

- ValueError : If diffusive_knudsen contains negative values, NaN, or
  infinity.

#### References

- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

#### Signature

```python
def get_hard_sphere_kernel(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# charged_kernel_strategy.md

# Charged Kernel Strategy

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Charged Kernel Strategy

> Auto-generated documentation for [particula.dynamics.coagulation.charged_kernel_strategy](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py) module.

## ChargedKernelStrategyABC

[Show source in charged_kernel_strategy.py:28](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L28)

Abstract base class for dimensionless coagulation strategies.

This class defines the dimensionless kernel (H) method, which must be
implemented by subclasses, and the `kernel` method that converts the
dimensionless kernel into a dimensioned coagulation kernel.

#### Methods

- dimensionless : Compute the dimensionless coagulation kernel (H).
- kernel : Convert a dimensionless kernel into a dimensioned kernel.

#### Examples

```py
class CustomKernel(ChargedKernelStrategyABC):
    def dimensionless(self, diff_kn, phi):
        # user-defined approaches
        return np.ones_like(diff_kn)

kernel_strategy = CustomKernel()
dim_kernel = kernel_strategy.kernel(
    dimensionless_kernel=kernel_strategy.dimensionless(...),
    coulomb_potential_ratio=...,
    sum_of_radii=...,
    reduced_mass=...,
    reduced_friction_factor=...
)
```

#### References

- See references in the individual subclasses for details on specific
  Coulomb approximations.

#### Signature

```python
class ChargedKernelStrategyABC(ABC): ...
```

### ChargedKernelStrategyABC().dimensionless

[Show source in charged_kernel_strategy.py:62](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L62)

Return the dimensionless coagulation kernel (H).

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number (KₙD)
  [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio (φ_E)
  [dimensionless].

#### Returns

- NDArray[np.float64] : The dimensionless coagulation kernel (H).

#### References

- Dyachkov, S. A., et al. (2007).
- Gatti, M., & Kortshagen, U. (2008).
- Gopalakrishnan, R., & Hogan, C. J. (2012).
- Chahl, H. S., & Gopalakrishnan, R. (2019).

#### Signature

```python
@abstractmethod
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### ChargedKernelStrategyABC().kernel

[Show source in charged_kernel_strategy.py:87](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L87)

Convert a dimensionless kernel into a dimensioned coagulation kernel.

Uses reduced mass, friction factors, and particle radii to obtain units
of [m³/s] for each particle-particle interaction.

#### Arguments

- dimensionless_kernel : The dimensionless coagulation kernel (H)
  [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio
  [dimensionless].
- sum_of_radii : The sum of the two particle radii [m].
- reduced_mass : The reduced mass of the two particles [kg].
- reduced_friction_factor : The reduced friction factor
  [dimensionless].

#### Returns

- The dimensioned coagulation kernel [m³/s].

#### Examples

```py title="Kernel Conversion Example"
dim_kernel = kernel_strategy.kernel(
    dimensionless_kernel=H,
    coulomb_potential_ratio=phi,
    sum_of_radii=r_sum,
    reduced_mass=m_reduced,
    reduced_friction_factor=zeta
)
```

#### Signature

```python
def kernel(
    self,
    dimensionless_kernel: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
    sum_of_radii: NDArray[np.float64],
    reduced_mass: NDArray[np.float64],
    reduced_friction_factor: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## CoulombDyachkov2007KernelStrategy

[Show source in charged_kernel_strategy.py:174](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L174)

Dyachkov et al. (2007) approximation for the dimensionless coagulation
kernel.

Accounts for Coulomb potential between particles, suitable for
transition regime calculations.

#### Methods

- dimensionless : Return the dimensionless kernel (H) following Dyachkov
  et al. (2007).

#### Examples

```py title="Use Dyachkov Kernel Strategy"
import particula as par
strategy = par.dynamics.CoulombDyachkov2007KernelStrategy()
H = strategy.dimensionless(diffusive_knudsen, coulomb_potential_ratio)
```

#### References

- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation
  of particles in the transition regime: The effect of the Coulomb
  potential. J. Chem. Phys., 126(12).
  [DOI](https://doi.org/10.1063/1.2713719)

#### Signature

```python
class CoulombDyachkov2007KernelStrategy(ChargedKernelStrategyABC): ...
```

#### See also

- [ChargedKernelStrategyABC](#chargedkernelstrategyabc)

### CoulombDyachkov2007KernelStrategy().dimensionless

[Show source in charged_kernel_strategy.py:200](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L200)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## CoulombGatti2008KernelStrategy

[Show source in charged_kernel_strategy.py:210](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L210)

Gatti & Kortshagen (2008) approximation for the dimensionless coagulation
kernel.

Captures Coulomb potential effects for a broad range of charge and
collisionality conditions.

#### Methods

- dimensionless : Return the dimensionless kernel (H) following Gatti
  and Kortshagen (2008).

#### Examples

```py title="Use Gatti Kernel Strategy"
import particula as par
strategy = par.dynamics.CoulombGatti2008KernelStrategy()
H = strategy.dimensionless(diff_kn, phi_ratio)
```

#### References

- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
  charging in plasmas over a wide range of collisionality. Phys. Rev. E,
  78(4).
  [DOI](https://doi.org/10.1103/PhysRevE.78.046402)

#### Signature

```python
class CoulombGatti2008KernelStrategy(ChargedKernelStrategyABC): ...
```

#### See also

- [ChargedKernelStrategyABC](#chargedkernelstrategyabc)

### CoulombGatti2008KernelStrategy().dimensionless

[Show source in charged_kernel_strategy.py:236](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L236)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## CoulombGopalakrishnan2012KernelStrategy

[Show source in charged_kernel_strategy.py:246](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L246)

Gopalakrishnan & Hogan (2012) approximation for the dimensionless
coagulation kernel.

Incorporates Coulomb-influenced collisions in aerosol and dusty plasma
environments.

#### Methods

- dimensionless : Return the dimensionless kernel (H) following
  Gopalakrishnan & Hogan (2012).

#### Examples

```py title="Use Gopalakrishnan Kernel Strategy"
import particula as par
strategy = par.dynamics.CoulombGopalakrishnan2012KernelStrategy()
H = strategy.dimensionless(kn, phi_ratio)
```

#### References

- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
  in aerosols and dusty plasmas. Phys. Rev. E, 85(2).
  [DOI](https://doi.org/10.1103/PhysRevE.85.026410)

#### Signature

```python
class CoulombGopalakrishnan2012KernelStrategy(ChargedKernelStrategyABC): ...
```

#### See also

- [ChargedKernelStrategyABC](#chargedkernelstrategyabc)

### CoulombGopalakrishnan2012KernelStrategy().dimensionless

[Show source in charged_kernel_strategy.py:271](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L271)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## CoulumbChahl2019KernelStrategy

[Show source in charged_kernel_strategy.py:283](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L283)

Chahl & Gopalakrishnan (2019) approximation for the dimensionless
coagulation kernel.

Focuses on high-potential, near-free molecular regime Coulombic collisions
in aerosols and dusty plasmas.

#### Methods

- dimensionless : Return the dimensionless kernel (H) following
  Chahl & Gopalakrishnan (2019).

#### Examples

```py title="Use Chahl 2019 Kernel Strategy"
import particula as par
strategy = CoulumbChahl2019KernelStrategy()
H = strategy.dimensionless(kn, phi)
```

#### References

- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
  molecular regime Coulombic collisions in aerosols and dusty plasmas.
  Aerosol Sci. Technol., 53(8).
  [DOI](https://doi.org/10.1080/02786826.2019.1614522)

#### Signature

```python
class CoulumbChahl2019KernelStrategy(ChargedKernelStrategyABC): ...
```

#### See also

- [ChargedKernelStrategyABC](#chargedkernelstrategyabc)

### CoulumbChahl2019KernelStrategy().dimensionless

[Show source in charged_kernel_strategy.py:309](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L309)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## HardSphereKernelStrategy

[Show source in charged_kernel_strategy.py:137](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L137)

Hard sphere dimensionless coagulation strategy.

Idealized Coulomb interactions and assumes particles interact as
perfectly charged spheres.

#### Methods

- dimensionless : Compute the dimensionless kernel under hard sphere
  assumptions.

#### Examples

```py title="Hard Sphere Kernel Strategy"
import particula as par
hs_strategy = par.dynamics.HardSphereKernelStrategy()
H = hs_strategy.dimensionless(
    diffusive_knudsen, coulomb_potential_ratio
)
# H is the hard-sphere dimensionless kernel
```

#### References

- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
  in aerosols and dusty plasmas. Phys. Rev. E, 85(2).
  [DOI](https://doi.org/10.1103/PhysRevE.85.026410)

#### Signature

```python
class HardSphereKernelStrategy(ChargedKernelStrategyABC): ...
```

#### See also

- [ChargedKernelStrategyABC](#chargedkernelstrategyabc)

### HardSphereKernelStrategy().dimensionless

[Show source in charged_kernel_strategy.py:164](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_kernel_strategy.py#L164)

#### Signature

```python
def dimensionless(
    self, diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: ignore
) -> NDArray[np.float64]: ...
```


---
# brownian_coagulation_builder.md

# BrownianCoagulationBuilder

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Builder](./index.md#coagulation-builder) / BrownianCoagulationBuilder

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.py) module.

## BrownianCoagulationBuilder

[Show source in brownian_coagulation_builder.py:27](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.py#L27)

Brownian Coagulation builder class.

Creates a `BrownianCoagulationStrategy` given a distribution type
(e.g., "discrete", "continuous_pdf", or "particle_resolved"). Ensures
the required parameters are set before building the strategy.

#### Attributes

- distribution_type : Representation of the particle
  size distribution.

#### Methods

- set_distribution_type : Assign the distribution type.
- set_parameters : Inherited from BuilderABC to set multiple
  parameters via a dict.
- build : Validate and return a `BrownianCoagulationStrategy`.

#### Examples

```py title="Example of using BrownianCoagulationBuilder"
import particula as par
builder = BrownianCoagulationBuilder()
builder.set_distribution_type("discrete")
strategy = builder.build()
# strategy is now a BrownianCoagulationStrategy instance
```

#### Signature

```python
class BrownianCoagulationBuilder(BuilderABC, BuilderDistributionTypeMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../../../abc_builder.md#builderabc)
- [BuilderDistributionTypeMixin](./coagulation_builder_mixin.md#builderdistributiontypemixin)

### BrownianCoagulationBuilder().build

[Show source in brownian_coagulation_builder.py:63](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.py#L63)

Validate and return the BrownianCoagulationStrategy object.

Checks that all required parameters (e.g., distribution_type) are set
before creating and returning a `BrownianCoagulationStrategy`.

#### Returns

BrownianCoagulationStrategy : The newly created
Brownian coagulation strategy.

#### Signature

```python
def build(self) -> CoagulationStrategyABC: ...
```

#### See also

- [CoagulationStrategyABC](../coagulation_strategy/coagulation_strategy_abc.md#coagulationstrategyabc)


---
# charged_coagulation_builder.md

# ChargedCoagulationBuilder

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Builder](./index.md#coagulation-builder) / ChargedCoagulationBuilder

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py) module.

## ChargedCoagulationBuilder

[Show source in charged_coagulation_builder.py:29](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L29)

Charged Coagulation builder class.

Creates a `ChargedCoagulationStrategy` based on a specified distribution
type and a `ChargedKernelStrategyABC` instance, enforcing the correct
parameters for modeling electrostatic interactions in aerosol coagulation.

#### Attributes

- distribution_type : Distribution representation
  ("discrete", "continuous_pdf", or "particle_resolved").
- charged_kernel_strategy : Instance of `ChargedKernelStrategyABC`
  for electrostatic kernel calculations.

#### Methods

- set_distribution_type : Set the distribution type.
- set_charged_kernel_strategy : Set the charged kernel strategy.
- set_parameters : Configure parameters from a dictionary.
- build : Validate inputs and return a `ChargedCoagulationStrategy`.

#### Examples

```py title="Example of using ChargedCoagulationBuilder"
import particula as par
builder = par.dynamics.ChargedCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_charged_kernel_strategy(charged_kernel_strategy)
coagulation_strategy = builder.build()
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric Chemistry
  and Physics." Wiley.

#### Signature

```python
class ChargedCoagulationBuilder(BuilderABC, BuilderDistributionTypeMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../../../abc_builder.md#builderabc)
- [BuilderDistributionTypeMixin](./coagulation_builder_mixin.md#builderdistributiontypemixin)

### ChargedCoagulationBuilder().build

[Show source in charged_coagulation_builder.py:107](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L107)

Validate and return the ChargedCoagulationStrategy object.

This method checks whether all required parameters have been
specified (e.g., distribution type, charged kernel strategy)
before creating a `ChargedCoagulationStrategy`.

#### Returns

- CoagulationStrategyABC : The properly configured
  charged coagulation strategy.

#### Examples

```py title="Example of using ChargedCoagulationBuilder build"
import particula as par
builder = ChargedCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_charged_kernel_strategy(charged_kernel_strategy)
charged_strategy = builder.build()
```

#### Signature

```python
def build(self) -> CoagulationStrategyABC: ...
```

#### See also

- [CoagulationStrategyABC](../coagulation_strategy/coagulation_strategy_abc.md#coagulationstrategyabc)

### ChargedCoagulationBuilder().set_charged_kernel_strategy

[Show source in charged_coagulation_builder.py:72](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L72)

Set the charged kernel strategy for electrostatic coagulation.

#### Arguments

- charged_kernel_strategy : An instance of
  `ChargedKernelStrategyABC`.
- charged_kernel_strategy_units : For interface consistency,
  unused.

#### Raises

- ValueError : If the kernel strategy is invalid or units passed
  are unsupported.

#### Signature

```python
def set_charged_kernel_strategy(
    self,
    charged_kernel_strategy: ChargedKernelStrategyABC,
    charged_kernel_strategy_units: Optional[str] = None,
): ...
```

#### See also

- [ChargedKernelStrategyABC](../charged_kernel_strategy.md#chargedkernelstrategyabc)


---
# coagulation_builder_mixin.md

# Coagulation Builder Mixin

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Builder](./index.md#coagulation-builder) / Coagulation Builder Mixin

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py) module.

## BuilderDistributionTypeMixin

[Show source in coagulation_builder_mixin.py:25](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L25)

Mixin class for distribution type in coagulation strategies.

Provides an interface to set the distribution type for coagulation
strategies. Ensures the chosen `distribution_type` is valid.

#### Attributes

- distribution_type : Stores the selected distribution type
  (e.g., "discrete", "continuous_pdf", "particle_resolved").

#### Methods

- set_distribution_type : Set and validate the distribution type.

#### Examples

```py title="Example of using BuilderDistributionTypeMixin"
builder = SomeCoagulationBuilder()
builder.set_distribution_type("discrete")
# builder.distribution_type -> "discrete"
```

#### Signature

```python
class BuilderDistributionTypeMixin:
    def __init__(self): ...
```

### BuilderDistributionTypeMixin().set_distribution_type

[Show source in coagulation_builder_mixin.py:49](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L49)

Set the distribution type.

#### Arguments

distribution_type : The distribution type to be set.
    Options are "discrete", "continuous_pdf", "particle_resolved".
distribution_type_units : Not used.

#### Returns

- The instance of the class with the updated
    distribution_type attribute.

#### Raises

- `ValueError` - If the distribution type is not valid.

#### Examples

```py title="Example of using set_distribution_type"
builder = SomeCoagulationBuilder()
builder.set_distribution_type("discrete")
# builder.distribution_type -> "discrete"
```

#### Signature

```python
def set_distribution_type(
    self, distribution_type: str, distribution_type_units: str = None
): ...
```



## BuilderFluidDensityMixin

[Show source in coagulation_builder_mixin.py:157](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L157)

Mixin class for fluid density parameters.

Adds methods and attributes for setting and validating fluid
density in coagulation strategies.

#### Attributes

- fluid_density : Numeric value representing fluid density
  in kg/m^3 (default units).

#### Methods

- set_fluid_density : Set and validate the fluid density.

#### Examples

```py title="Example of using BuilderFluidDensityMixin"
builder.set_fluid_density(1.225, "kg/m^3")
```

#### Signature

```python
class BuilderFluidDensityMixin:
    def __init__(self): ...
```

### BuilderFluidDensityMixin().set_fluid_density

[Show source in coagulation_builder_mixin.py:179](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L179)

Set the density of the particle in kg/m^3.

#### Arguments

density : Density of the particle.
density_units : Units of the density. Default is *kg/m^3*

#### Returns

- The instance of the class with the updated
    fluid_density attribute.

#### Raises

- `ValueError` - Must be positive value.

#### Examples

```py title="Example of using set_fluid_density"
builder = SomeCoagulationBuilder()
builder.set_fluid_density(1.225, "kg/m^3")
# builder.fluid_density -> 1.225
```

#### Signature

```python
@validate_inputs({"fluid_density": "positive"})
def set_fluid_density(
    self, fluid_density: Union[float, NDArray[np.float64]], fluid_density_units: str
): ...
```



## BuilderTurbulentDissipationMixin

[Show source in coagulation_builder_mixin.py:97](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L97)

Mixin class for turbulent shear parameters.

Adds methods and attributes for setting and validating
turbulent dissipation parameters in coagulation strategies.

#### Attributes

- turbulent_dissipation : Numeric value of the energy dissipation
  rate in m^2/s^3 (default units).

#### Methods

- set_turbulent_dissipation : Set and validate the turbulent
    dissipation rate.

#### Examples

```py title="Example of using BuilderTurbulentDissipationMixin"
builder.set_turbulent_dissipation(1e-3, "m^2/s^3")
```

#### Signature

```python
class BuilderTurbulentDissipationMixin:
    def __init__(self): ...
```

### BuilderTurbulentDissipationMixin().set_turbulent_dissipation

[Show source in coagulation_builder_mixin.py:120](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L120)

Set the turbulent dissipation rate.

#### Arguments

turbulent_dissipation : Turbulent dissipation rate.
turbulent_dissipation_units : Units of the turbulent dissipation
    rate. Default is *m^2/s^3*.

#### Returns

- The instance of the class with the updated
    turbulent_dissipation attribute.

#### Raises

- `ValueError` - Must be non-negative value.

#### Examples

```py title="Example of using set_turbulent_dissipation"
builder = SomeCoagulationBuilder()
builder.set_turbulent_dissipation(1e-3, "m^2/s^3")
# builder.turbulent_dissipation -> 1e-3
```

#### Signature

```python
@validate_inputs({"turbulent_dissipation": "nonnegative"})
def set_turbulent_dissipation(
    self, turbulent_dissipation: float, turbulent_dissipation_units: str
): ...
```


---
# combine_coagulation_strategy_builder.md

# CombineCoagulationStrategyBuilder

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Builder](./index.md#coagulation-builder) / CombineCoagulationStrategyBuilder

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_builder.combine_coagulation_strategy_builder](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py) module.

## CombineCoagulationStrategyBuilder

[Show source in combine_coagulation_strategy_builder.py:27](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L27)

Builder for a combined coagulation strategy.

This class constructs a `CombineCoagulationStrategy` from multiple
sub-strategies (instances of `CoagulationStrategyABC`), enabling
advanced modeling scenarios where different coagulation mechanisms
act concurrently. Each sub-strategy's rate calculations are effectively
merged to act on the same particle population.

#### Attributes

- strategies : List of `CoagulationStrategyABC` objects to combine.

#### Methods

- set_strategies : Set the list of coagulation strategies to combine.
- build : Create and return the combined coagulation strategy.

#### Examples

```py title="Combine Coagulation Strategy Example"
import particula as par
builder = par.dynamics.CombineCoagulationStrategyBuilder()
builder.set_strategies([brownian_strategy, turbulent_strategy])
combined_strategy = builder.build()
```

#### Signature

```python
class CombineCoagulationStrategyBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../../../abc_builder.md#builderabc)

### CombineCoagulationStrategyBuilder().build

[Show source in combine_coagulation_strategy_builder.py:98](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L98)

Builds and returns the combined coagulation strategy.

#### Returns

CombineCoagulationStrategy :
    A strategy that combines all the previously added
    sub-strategies.

#### Examples

```py title="Build Example with CombineCoagulationStrategy"
combined_strategy = builder.build()
# Now you can use `combined_strategy.kernel(...)` to calculate
# combined coagulation effects from each sub-strategy.
```

#### Signature

```python
def build(self) -> CombineCoagulationStrategy: ...
```

#### See also

- [CombineCoagulationStrategy](../coagulation_strategy/combine_coagulation_strategy.md#combinecoagulationstrategy)

### CombineCoagulationStrategyBuilder().set_strategies

[Show source in combine_coagulation_strategy_builder.py:69](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/combine_coagulation_strategy_builder.py#L69)

Sets a list of CoagulationStrategyABC objects to be combined.

#### Arguments

strategies : A list of coagulation strategies to be combined.
strategies_units : For interface consistency, not used.

#### Examples

```py title="Set Strategies Example"
builder = CombineCoagulationStrategyBuilder()
builder.set_strategies([brownian_strategy, turbulent_strategy])
```

#### Returns

CombineCoagulationStrategyBuilder:
    The builder instance, for fluent chaining.

#### Signature

```python
def set_strategies(
    self,
    strategies: List[CoagulationStrategyABC],
    strategies_units: Optional[str] = None,
): ...
```


---
# turbulent_dns_coagulation_builder.md

# TurbulentDNSCoagulationBuilder

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Builder](./index.md#coagulation-builder) / TurbulentDNSCoagulationBuilder

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py) module.

## TurbulentDNSCoagulationBuilder

[Show source in turbulent_dns_coagulation_builder.py:36](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L36)

Turbulent DNS coagulation builder class.

Creates and configures a `TurbulentDNSCoagulationStrategy` to simulate
coagulation in turbulent flow fields using Direct Numerical Simulation
parameters. This builder enforces that the required parameters
(distribution_type, turbulent_dissipation, fluid_density, reynolds_lambda,
relative_velocity) are set prior to building the strategy.

#### Attributes

- distribution_type : The particle distribution type
  ("discrete", "continuous_pdf", or "particle_resolved").
- turbulent_dissipation : Rate of turbulent energy dissipation (m²/s³).
- fluid_density : Fluid density in kg/m³.
- reynolds_lambda : Taylor-scale Reynolds number (dimensionless).
- relative_velocity : Relative velocity in m/s (particle vs. airflow).

#### Methods

- set_distribution_type : Set the distribution type.
- set_turbulent_dissipation : Set the turbulent dissipation rate.
- set_fluid_density : Set the fluid density.
- set_reynolds_lambda : Set the Taylor-scale Reynolds number.
- set_relative_velocity : Set the relative velocity.
- build : Validate parameters and return a
  `TurbulentDNSCoagulationStrategy`.

#### Examples

```py title="Turbulent DNS Builder Example"
builder = TurbulentDNSCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_turbulent_dissipation(1e-3)
builder.set_fluid_density(1.225)
builder.set_reynolds_lambda(250.)
builder.set_relative_velocity(0.5, "m/s")
strategy = builder.build()
# Now 'strategy' can be used to compute DNS-based coagulation rates.

References:
- Saffman, P. G., & Turner, J. S. (1956) "On the collision of drops
  in turbulent clouds." Journal of Fluid Mechanics, 1(1): 16–30.

#### Signature

```python
class TurbulentDNSCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../../../abc_builder.md#builderabc)
- [BuilderDistributionTypeMixin](./coagulation_builder_mixin.md#builderdistributiontypemixin)
- [BuilderFluidDensityMixin](./coagulation_builder_mixin.md#builderfluiddensitymixin)
- [BuilderTurbulentDissipationMixin](./coagulation_builder_mixin.md#builderturbulentdissipationmixin)

### TurbulentDNSCoagulationBuilder().build

[Show source in turbulent_dns_coagulation_builder.py:161](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L161)

Construct a TurbulentDNSCoagulationStrategy.

Validates the required parameters, then instantiates a
`TurbulentDNSCoagulationStrategy` for DNS-based coagulation
calculations.

#### Returns

- CoagulationStrategyABC : The configured DNS coagulation strategy.

#### Signature

```python
def build(self) -> CoagulationStrategyABC: ...
```

#### See also

- [CoagulationStrategyABC](../coagulation_strategy/coagulation_strategy_abc.md#coagulationstrategyabc)

### TurbulentDNSCoagulationBuilder().set_relative_velocity

[Show source in turbulent_dns_coagulation_builder.py:133](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L133)

Set the relative particle-airflow velocity for DNS coagulation.

This value is typically a background flow velocity or a
sedimentation-adjusted velocity, excluding turbulence.

#### Arguments

- relative_velocity : Numeric value of velocity.
- relative_velocity_units : Units of the velocity
  (e.g., "m/s").

#### Returns

- self : The builder instance for chaining.

#### Signature

```python
@validate_inputs({"relative_velocity": "finite"})
def set_relative_velocity(
    self, relative_velocity: float, relative_velocity_units: str
): ...
```

### TurbulentDNSCoagulationBuilder().set_reynolds_lambda

[Show source in turbulent_dns_coagulation_builder.py:99](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L99)

Set the Taylor-scale Reynolds number (Reλ).

Represents a measure of turbulence intensity in DNS flows.
When specifying units, only "dimensionless" is recognized here.
Any other unit triggers a warning and is treated as dimensionless.

#### Arguments

- reynolds_lambda : Numeric value for Reλ.
- reynolds_lambda_units : String indicating units
  (default "dimensionless").

#### Returns

- self : The builder instance for chaining.

#### Examples

```py
builder.set_reynolds_lambda(250.)
```

#### Signature

```python
@validate_inputs({"reynolds_lambda": "nonnegative"})
def set_reynolds_lambda(
    self, reynolds_lambda: float, reynolds_lambda_units: Optional[str] = None
): ...
```


---
# turbulent_shear_coagulation_builder.md

# TurbulentShearCoagulationBuilder

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Builder](./index.md#coagulation-builder) / TurbulentShearCoagulationBuilder

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.py) module.

## TurbulentShearCoagulationBuilder

[Show source in turbulent_shear_coagulation_builder.py:23](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.py#L23)

Turbulent shear coagulation builder.

Creates a TurbulentShearCoagulationStrategy that calculates coagulation
rates under turbulent flow conditions. Ensures the correct distribution
type, turbulent dissipation, and fluid density values are provided.

#### Attributes

- distribution_type : Type of the particle distribution
  ("discrete", "continuous_pdf", or "particle_resolved").
- turbulent_dissipation : Turbulent energy dissipation rate (m²/s³).
- fluid_density : Fluid density (kg/m³) for the coagulation medium.

#### Methods

- set_distribution_type : Set the distribution type.
- set_turbulent_dissipation : Set turbulent dissipation rate.
- set_fluid_density : Set fluid density.
- build : Validate parameters and return a
  TurbulentShearCoagulationStrategy.

#### Examples

```py title="Turbulent Shear Coagulation Builder Example"
import particula as par
builder = par.dynamics.TurbulentShearCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_turbulent_dissipation(1e-3)
builder.set_fluid_density(1000.)
strategy = builder.build()
# Now 'strategy' can be used to compute turbulent shear coagulation
# rates.
```

#### References

- Saffman, P. G., & Turner, J. S. (1956). "On the collision of drops
  in turbulent clouds." J. Fluid Mech., 1, 16-30.

#### Signature

```python
class TurbulentShearCoagulationBuilder(
    BuilderABC,
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../../../abc_builder.md#builderabc)
- [BuilderDistributionTypeMixin](./coagulation_builder_mixin.md#builderdistributiontypemixin)
- [BuilderFluidDensityMixin](./coagulation_builder_mixin.md#builderfluiddensitymixin)
- [BuilderTurbulentDissipationMixin](./coagulation_builder_mixin.md#builderturbulentdissipationmixin)

### TurbulentShearCoagulationBuilder().build

[Show source in turbulent_shear_coagulation_builder.py:87](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.py#L87)

Construct a TurbulentShearCoagulationStrategy.

This method performs a final check to ensure all required parameters
have been set. It then creates and returns an instance of
TurbulentShearCoagulationStrategy.

#### Returns

- The resulting turbulent shear coagulation strategy object.

#### Signature

```python
def build(self) -> CoagulationStrategyABC: ...
```

#### See also

- [CoagulationStrategyABC](../coagulation_strategy/coagulation_strategy_abc.md#coagulationstrategyabc)


---
# coagulation_factories.md

# Coagulation Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Coagulation Factories

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_factories](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_factories.py) module.

## CoagulationFactory

[Show source in coagulation_factories.py:29](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_factories.py#L29)

Factory class for creating coagulation strategy instances
based on a given type string. Supported types include:
    - 'brownian'
    - 'charged'
    - 'turbulent_shear'
    - 'turbulent_dns'
    - 'combine'

#### Methods

- get_builders() : Returns the mapping of strategy types to builder
    instances.
- get_strategy(strategy_type, parameters): Gets the strategy instance
    for the specified strategy type.
    - `-` *strategy_type* - Type of coagulation strategy to use, can be
        'brownian', 'charged', 'turbulent_shear', 'turbulent_dns', or
        'combine'.

#### Signature

```python
class CoagulationFactory(
    StrategyFactoryABC[
        Union[
            BrownianCoagulationBuilder,
            ChargedCoagulationBuilder,
            TurbulentShearCoagulationBuilder,
            TurbulentDNSCoagulationBuilder,
            CombineCoagulationStrategyBuilder,
        ],
        CoagulationStrategyABC,
    ]
): ...
```

#### See also

- [BrownianCoagulationBuilder](coagulation_builder/brownian_coagulation_builder.md#browniancoagulationbuilder)
- [ChargedCoagulationBuilder](coagulation_builder/charged_coagulation_builder.md#chargedcoagulationbuilder)
- [CoagulationStrategyABC](coagulation_strategy/coagulation_strategy_abc.md#coagulationstrategyabc)
- [CombineCoagulationStrategyBuilder](coagulation_builder/combine_coagulation_strategy_builder.md#combinecoagulationstrategybuilder)
- [TurbulentDNSCoagulationBuilder](coagulation_builder/turbulent_dns_coagulation_builder.md#turbulentdnscoagulationbuilder)
- [TurbulentShearCoagulationBuilder](coagulation_builder/turbulent_shear_coagulation_builder.md#turbulentshearcoagulationbuilder)

### CoagulationFactory().get_builders

[Show source in coagulation_factories.py:60](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_factories.py#L60)

#### Signature

```python
def get_builders(self) -> Dict[str, Any]: ...
```


---
# coagulation_rate.md

# Coagulation Rate

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Coagulation Rate

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_rate](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_rate.py) module.

## get_coagulation_gain_rate_continuous

[Show source in coagulation_rate.py:172](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_rate.py#L172)

Calculate the coagulation gain rate via continuous integration.

This function converts the distribution to a continuous form, then
uses RectBivariateSpline to interpolate and integrate:

- gain_rate(r) = ∫ kernel(r, r') × concentration(r) × concentration(r') dr'

#### Arguments

- radius : The particle radius array [m].
- concentration : The particle distribution.
- kernel : Coagulation kernel matrix.

#### Returns

- The coagulation gain rate, in the shape of radius.

#### Examples

```py
import numpy as np
import particula as par

r = np.array([1e-7, 2e-7, 3e-7])
conc = np.array([1.0, 0.5, 0.2])
kern = np.ones((3, 3)) * 1e-9

gain_cont = par.dynamics.get_coagulation_gain_rate_continuous(
    r, conc, kern
)
print(gain_cont)
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Chapter 13, Equation 13.61.

#### Signature

```python
def get_coagulation_gain_rate_continuous(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coagulation_gain_rate_discrete

[Show source in coagulation_rate.py:58](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_rate.py#L58)

Calculate the coagulation gain rate (using a quasi-continuous approach).

Though named "discrete," this function converts the discrete distribution
to a PDF and uses interpolation (RectBivariateSpline) to approximate the
gain term. The concept is:

- gain_rate(r) = ∫ kernel(r, r') × PDF(r) × PDF(r') dr'
  (implemented via numeric integration)

#### Arguments

- radius : The particle radius array [m].
- concentration : The particle distribution.
- kernel : Coagulation kernel matrix.

#### Returns

- The coagulation gain rate, matched to the shape of radius.

#### Examples

```py
import numpy as np
import particula as par

r = np.array([1e-7, 2e-7, 3e-7])
conc = np.array([1.0, 0.5, 0.2])
kern = np.ones((3, 3)) * 1e-9

gain_val = par.dynamics.get_coagulation_gain_rate_discrete(
    r, conc, kern
)
print(gain_val)
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Chapter 13, Equation 13.61.

#### Signature

```python
def get_coagulation_gain_rate_discrete(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coagulation_loss_rate_continuous

[Show source in coagulation_rate.py:128](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_rate.py#L128)

Calculate the coagulation loss rate via continuous integration.

This method integrates the product of kernel and concentration over
the radius grid. The equation is:

- loss_rate(r) = concentration(r) × ∫ kernel(r, r') × concentration(r') dr'

#### Arguments

- radius : The particle radius array [m].
- concentration : The particle distribution.
- kernel : Coagulation kernel matrix (NDArray[np.float64]).

#### Returns

- The coagulation loss rate.

#### Examples

```py
import numpy as np
import particula as par

r = np.array([1e-7, 2e-7, 3e-7])
conc = np.array([1.0, 0.5, 0.2])
kern = np.ones((3, 3)) * 1e-9

loss_cont = par.dynamics.get_coagulation_loss_rate_continuous(
    r, conc, kern
)
print(loss_cont)
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Chapter 13, Equation 13.61.

#### Signature

```python
def get_coagulation_loss_rate_continuous(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coagulation_loss_rate_discrete

[Show source in coagulation_rate.py:20](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_rate.py#L20)

Calculate the coagulation loss rate via a discrete summation approach.

This function computes the loss rate of particles from collisions by
summing over all size classes. The equation is:

- loss_rate = ΣᵢΣⱼ [kernel(i, j) × concentration(i) × concentration(j)]

#### Arguments

- concentration : The distribution of particles.
- kernel : The coagulation kernel matrix (NDArray[np.float64]).

#### Returns

- The coagulation loss rate (float or NDArray[np.float64]).

#### Examples

```py
import numpy as np
import particula as par

conc = np.array([1.0, 2.0, 3.0])
kern = np.ones((3, 3))
loss = par.dynamics.get_coagulation_loss_rate_discrete(conc, kern)
print(loss)
# Example output: 36.0
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics, Chapter 13, Equation 13.61.

#### Signature

```python
def get_coagulation_loss_rate_discrete(
    concentration: Union[float, NDArray[np.float64]], kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```


---
# brownian_coagulation_strategy.md

# BrownianCoagulationStrategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Strategy](./index.md#coagulation-strategy) / BrownianCoagulationStrategy

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py) module.

## BrownianCoagulationStrategy

[Show source in brownian_coagulation_strategy.py:22](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L22)

Discrete Brownian coagulation strategy class for aerosol simulations.

This class implements methods defined in CoagulationStrategyABC
to simulate Brownian coagulation in particle populations. It calculates
coagulation rates via a Brownian kernel that depends on properties such
as temperature, pressure, and particle radius.

#### Attributes

- distribution_type : Defines how particles are represented
  (e.g., "discrete", "continuous_pdf", or "particle_resolved").

#### Methods

- kernel : Calculate the Brownian coagulation kernel (dimensioned).
- loss_rate : Calculate the coagulation loss rate (not shown here).
- gain_rate : Calculate the coagulation gain rate (not shown here).
- net_rate : Calculate the net coagulation rate (not shown here).
- dimensionless_kernel : Not implemented, raises NotImplementedError.
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

#### Examples

```py title="Example Usage of BrownianCoagulationStrategy"
import particula as par
brownian_strat = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete"
)
# Suppose we have a ParticleRepresentation object called 'particle_rep'
# kernel_values = brownian_strat.kernel(
#   particle_rep, temperature=298, pressure=101325
# )
# ...
```

#### References

- `get_brownian_kernel_via_system_state`
- Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric chemistry
  and physics," Section 13, Table 13.1.

#### Signature

```python
class BrownianCoagulationStrategy(CoagulationStrategyABC):
    def __init__(self, distribution_type: str): ...
```

#### See also

- [CoagulationStrategyABC](./coagulation_strategy_abc.md#coagulationstrategyabc)

### BrownianCoagulationStrategy().dimensionless_kernel

[Show source in brownian_coagulation_strategy.py:77](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L77)

Not implemented for BrownianCoagulationStrategy.

This method raises NotImplementedError since dimensionless
Brownian kernels are not defined here.

#### Arguments

- diffusive_knudsen : Knudsen number array (unused).
- coulomb_potential_ratio : Coulomb ratio array (unused).

#### Raises

- NotImplementedError : Always, as no dimensionless kernel is
provided.

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### BrownianCoagulationStrategy().kernel

[Show source in brownian_coagulation_strategy.py:103](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L103)

Calculate the dimensioned Brownian coagulation kernel.

Leverages the `get_brownian_kernel_via_system_state` function to
compute the kernel, which accounts for particle size, temperature,
and pressure. The kernel typically has units of volume per time.

#### Arguments

- particle : ParticleRepresentation containing the distribution
  and density or mass data needed for the kernel calculation.
- temperature : System temperature in Kelvin.
- pressure : System pressure in Pascals.

#### Returns

- Brownian coagulation kernel values. Shape depends on the
  underlying distribution.

#### Examples

```py
kernel_matrix = brownian_strat.kernel(particle_rep, 300, 101325)
```

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)


---
# charged_coagulation_strategy.md

# ChargedCoagulationStrategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Strategy](./index.md#coagulation-strategy) / ChargedCoagulationStrategy

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py) module.

## ChargedCoagulationStrategy

[Show source in charged_coagulation_strategy.py:23](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L23)

Charged Brownian coagulation strategy using a dimensionless kernel.

This class implements the methods defined in the CoagulationStrategyABC
abstract class. A ChargedKernelStrategyABC instance is passed to define
how the dimensionless kernel is calculated. This approach allows flexible
handling of Coulomb interactions under various regimes.

#### Attributes

- kernel_strategy : Instance of ChargedKernelStrategyABC used to
  calculate dimensionless and dimensioned kernels.

#### Methods

- dimensionless_kernel : Compute dimensionless kernel values for
  charged coagulation.
- kernel : Convert dimensionless kernel values into a dimensioned
  coagulation kernel.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

#### Examples

```py title="Example Usage"
import numpy as np
import particula as par
kernel_strategy = par.dynamics.HardSphereKernelStrategy()
charged_coag = par.dynamics.ChargedCoagulationStrategy(
    distribution_type="discrete", kernel_strategy=kernel_strategy
)
# Now the charged_coag object can compute dimensionless and
# dimensioned kernels given a ParticleRepresentation object.
```

#### References

- Seinfeld, J. H., & Pandis, S. N. "Atmospheric Chemistry and
  - `Physics` - From Air Pollution to Climate Change." Wiley, 2016.

#### Signature

```python
class ChargedCoagulationStrategy(CoagulationStrategyABC):
    def __init__(
        self, distribution_type: str, kernel_strategy: ChargedKernelStrategyABC
    ): ...
```

#### See also

- [ChargedKernelStrategyABC](../charged_kernel_strategy.md#chargedkernelstrategyabc)
- [CoagulationStrategyABC](./coagulation_strategy_abc.md#coagulationstrategyabc)

### ChargedCoagulationStrategy().dimensionless_kernel

[Show source in charged_coagulation_strategy.py:88](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L88)

Compute the dimensionless kernel for charged coagulation.

This method delegates computation to the provided kernel strategy. It
returns the dimensionless kernel (H) as a function of the diffusive
Knudsen number and the Coulomb potential ratio.

#### Arguments

- diffusive_knudsen : Dimensionless Knudsen number(s) describing
  particle diffusive behavior.
- coulomb_potential_ratio : Dimensionless ratio(s) incorporating
  electrostatic interactions.

#### Returns

- NDArray[np.float64] : Array of dimensionless kernel values.

#### Examples

```py title="Dimensionless Kernel Example"
kn = np.array([0.1, 0.2])
phi = np.array([1.0, 2.0])
dim_kernel = charged_coag.dimensionless_kernel(kn, phi)
# dim_kernel -> array of dimensionless kernel values
```

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### ChargedCoagulationStrategy().kernel

[Show source in charged_coagulation_strategy.py:122](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L122)

Compute the dimensioned coagulation kernel for charged particles.

This method converts the dimensionless kernel into a dimensioned
coagulation kernel by combining Coulomb parameters, the pairwise
radii of particles, reduced mass, and friction factors.

#### Arguments

- particle : A ParticleRepresentation instance containing
  distribution, density, and concentration data.
- temperature : Float specifying the system temperature (K).
- pressure : Float specifying the system pressure (Pa).

#### Returns

- float or NDArray[np.float64] : The dimensioned coagulation
  kernel value(s).

#### Examples

```py title="Dimensioned Kernel Example"
kernel_matrix = charged_coag.kernel(
    particle=my_particle, temperature=300, pressure=101325
)
# kernel_matrix -> 2D array of size (n_particles, n_particles)
```

#### References

- Gopalakrishnan, R. & Hogan, C. J. "Determination of the Transition
  Regime Collision Kernel from Mean First Passage Times." Aerosol
  Science and Technology, 46: 887-899, 2012.

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)


---
# coagulation_strategy_abc.md

# CoagulationStrategyABC

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Strategy](./index.md#coagulation-strategy) / CoagulationStrategyABC

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py) module.

## CoagulationStrategyABC

[Show source in coagulation_strategy_abc.py:28](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L28)

Abstract base class for defining a coagulation strategy.

This class defines the methods that must be implemented by any coagulation
strategy (e.g., for discrete, continuous, or particle-resolved distributions).

#### Attributes

- distribution_type : The type of distribution to be used, one of
  ("discrete", "continuous_pdf", or "particle_resolved").

#### Methods

- dimensionless_kernel : Calculate the dimensionless coagulation kernel.
- kernel : Obtain the dimensioned coagulation kernel [m^3/s].
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

#### Examples

```py
class ExampleCoagulation(CoagulationStrategyABC):
    def dimensionless_kernel(self, diff_kn, coulomb_phi):
        return diff_kn + coulomb_phi
    def kernel(self, particle, temperature, pressure):
        return 1.0
strategy = ExampleCoagulation("discrete")
```

#### References

- Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and Physics:
  From Air Pollution to Climate Change (3rd ed.). Wiley.

#### Signature

```python
class CoagulationStrategyABC(ABC):
    def __init__(
        self,
        distribution_type: str,
        particle_resolved_kernel_radius: Optional[NDArray[np.float64]] = None,
        particle_resolved_kernel_bins_number: Optional[int] = None,
        particle_resolved_kernel_bins_per_decade: int = 10,
    ): ...
```

### CoagulationStrategyABC().coulomb_potential_ratio

[Show source in coagulation_strategy_abc.py:380](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L380)

Calculate the Coulomb potential ratio for each particle.

This ratio characterizes the influence of electrostatic forces on
coagulation processes.

#### Arguments

- particle : The ParticleRepresentation.
- temperature : The gas-phase temperature [K].

#### Returns

- NDArray[np.float64] : Coulomb potential ratio(s) [dimensionless].

#### Examples

```py
phi = strategy.coulomb_potential_ratio(particle, 298.15)
```

#### Signature

```python
def coulomb_potential_ratio(
    self, particle: ParticleRepresentation, temperature: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### CoagulationStrategyABC().diffusive_knudsen

[Show source in coagulation_strategy_abc.py:339](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L339)

Calculate the diffusive Knudsen number for each particle.

The Knudsen number is used to characterize the relative importance of
diffusion-controlled processes.

#### Arguments

- particle : The ParticleRepresentation.
- temperature : The gas-phase temperature [K].
- pressure : The gas-phase pressure [Pa].

#### Returns

- NDArray[np.float64] : Diffusive Knudsen number(s) [dimensionless].

#### Examples

```py
knudsen_nums = strategy.diffusive_knudsen(particle, 298.15, 101325)
```

#### Signature

```python
def diffusive_knudsen(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### CoagulationStrategyABC().dimensionless_kernel

[Show source in coagulation_strategy_abc.py:92](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L92)

Calculate the dimensionless coagulation kernel.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio [dimensionless].

#### Returns

- NDArray[np.float64] : Dimensionless kernel for particle coagulation.

#### Examples

```py
H = strategy.dimensionless_kernel(kn_array, phi_array)
# H might be array([...]) representing the dimensionless kernel
```

#### Signature

```python
@abstractmethod
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### CoagulationStrategyABC().friction_factor

[Show source in coagulation_strategy_abc.py:407](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L407)

Compute the friction factor for each particle in the aerosol.

Considers dynamic viscosity, mean free path, and slip correction to
determine the friction factor [dimensionless].

#### Arguments

- particle : The ParticleRepresentation for which to compute friction factor.
- temperature : Gas temperature [K].
- pressure : Gas pressure [Pa].

#### Returns

- NDArray[np.float64] : Friction factor(s) [dimensionless].

#### Examples

```py
fr = strategy.friction_factor(particle, 298.15, 101325)
```

#### Signature

```python
def friction_factor(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### CoagulationStrategyABC().gain_rate

[Show source in coagulation_strategy_abc.py:182](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L182)

Calculate the coagulation gain rate [kg/s].

#### Arguments

- particle : The particle representation used in the calculation.
- kernel : The coagulation kernel [m^3/s].

#### Returns

- float or NDArray[np.float64] : The gain rate [kg/s].

#### Raises

- ValueError : If the distribution type is invalid.

#### Examples

```py
gain = strategy.gain_rate(particle, k_val)
```

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### CoagulationStrategyABC().kernel

[Show source in coagulation_strategy_abc.py:115](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L115)

Calculate the coagulation kernel [m^3/s].

Uses particle attributes (e.g., radius, mass) along with temperature
and pressure to return a dimensional kernel for coagulation.

#### Arguments

- particle : The ParticleRepresentation object, providing radius and concentration.
- temperature : The temperature in Kelvin [K].
- pressure : The pressure in Pascals [Pa].

#### Returns

- float or NDArray[np.float64] : The coagulation kernel [m^3/s].

#### Examples

```py
k_val = strategy.kernel(particle, 298.15, 101325)
# k_val can be a scalar or array
```

#### Signature

```python
@abstractmethod
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### CoagulationStrategyABC().loss_rate

[Show source in coagulation_strategy_abc.py:143](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L143)

Calculate the coagulation loss rate [kg/s].

#### Arguments

- particle : The particle representation for which the loss rate is calculated.
- kernel : The coagulation kernel [m^3/s].

#### Returns

- float or NDArray[np.float64] : The loss rate [kg/s].

#### Raises

- ValueError : If the distribution type is invalid.

#### Examples

```py
loss = strategy.loss_rate(particle, k_val)
```

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### CoagulationStrategyABC().net_rate

[Show source in coagulation_strategy_abc.py:222](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L222)

Compute the net coagulation rate = gain - loss [kg/s].

#### Arguments

- particle : The particle representation.
- temperature : The gas-phase temperature [K].
- pressure : The gas-phase pressure [Pa].

#### Returns

- float or NDArray[np.float64] : The net coagulation rate [kg/s].
    (positive => net gain, negative => net loss).

#### Examples

```py
net = strategy.net_rate(particle, 298.15, 101325)
```

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### CoagulationStrategyABC().step

[Show source in coagulation_strategy_abc.py:253](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L253)

Perform a single coagulation step over a specified time interval.

Updates the particle distribution or representation based on the net_rate
calculated for the given time_step.

#### Arguments

- particle : The particle representation to update.
- temperature : The gas-phase temperature [K].
- pressure : The gas-phase pressure [Pa].
- time_step : The timestep over which to integrate [s].

#### Returns

- ParticleRepresentation : Updated particle representation after this step.

#### Raises

- ValueError : If the distribution type is invalid or unsupported.

#### Examples

```py
updated_particle = strategy.step(particle, 298.15, 101325, 1.0)
```

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)


---
# combine_coagulation_strategy.md

# CombineCoagulationStrategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Strategy](./index.md#coagulation-strategy) / CombineCoagulationStrategy

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py) module.

## CombineCoagulationStrategy

[Show source in combine_coagulation_strategy.py:26](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L26)

Combine multiple coagulation strategies into one.

This class takes a list of coagulation strategies and merges their
kernels by summing them. Each included strategy must share the same
distribution type.

#### Attributes

- distribution_type : Matches the distribution_type of the first
  strategy.
- strategies : A list of individual CoagulationStrategyABC instances.

#### Methods

- dimensionless_kernel : Raises NotImplementedError, as not supported here.
- kernel : Compute the sum of all strategy kernels.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

#### Examples

```py title="Example Usage of CombineCoagulationStrategy"
import particula as par
combined_strategy = par.dynamics.CombineCoagulationStrategy(
    [strategy1, strategy2]
)
k_value = combined_strategy.kernel(
    particle=aerosol, temperature=300, pressure=101325
) # combined kernel value
```

#### References

- No specific references. Summation approach is straightforward.

#### Signature

```python
class CombineCoagulationStrategy(CoagulationStrategyABC):
    def __init__(self, strategies: List[CoagulationStrategyABC]): ...
```

#### See also

- [CoagulationStrategyABC](./coagulation_strategy_abc.md#coagulationstrategyabc)

### CombineCoagulationStrategy().dimensionless_kernel

[Show source in combine_coagulation_strategy.py:94](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L94)

Raise NotImplementedError for dimensionless kernel in combined
strategy.

Dimensionless kernels must be handled individually by the underlying
strategies. This method logs an error and raises NotImplementedError.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number(s)
  [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio(s)
  [dimensionless].

#### Raises

- NotImplementedError : This method is not supported in the
  combined strategy.

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### CombineCoagulationStrategy().kernel

[Show source in combine_coagulation_strategy.py:123](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L123)

Compute the total coagulation kernel by summing the kernels from all
underlying strategies.

#### Arguments

- particle : The ParticleRepresentation instance containing
  particle data (radii, distribution, etc.).
- temperature : The temperature in Kelvin [K].
- pressure : The pressure in Pascals [Pa].

#### Returns

- float or NDArray[np.float64] : The combined coagulation kernel,
  equal to the sum of each strategy's kernel.

#### Examples

```py
k_combined = combined_strategy.kernel(
    particle=my_particle,
    temperature=300.0,
    pressure=101325
)
# k_combined is the sum of kernels from each strategy in
# combined_strategy
```

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)


---
# sedimentation_coagulation_strategy.md

# SedimentationCoagulationStrategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Strategy](./index.md#coagulation-strategy) / SedimentationCoagulationStrategy

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_strategy.sedimentation_coagulation_strategy](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/sedimentation_coagulation_strategy.py) module.

## SedimentationCoagulationStrategy

[Show source in sedimentation_coagulation_strategy.py:25](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/sedimentation_coagulation_strategy.py#L25)

Sedimentation coagulation strategy for aerosol particles.

Implements the Seinfeld & Pandis (2016) sedimentation kernel as part of
the CoagulationStrategyABC. This approach models collisions driven by
gravitational settling.

#### Attributes

- distribution_type : The particle distribution type ("discrete",
  "continuous_pdf", or "particle_resolved").

#### Methods

- dimensionless_kernel : Raises NotImplementedError for this strategy.
- kernel : Return the sedimentation coagulation kernel [m^3/s].
- loss_rate : (Inherited) Calculate coagulation loss rate.
- gain_rate : (Inherited) Calculate coagulation gain rate.
- net_rate : (Inherited) Calculate net coagulation rate.
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

#### Examples

```py title="Sedimentation Coagulation Strategy Example"
import particula as par
strategy = SedimentationCoagulationStrategy(
    distribution_type="discrete"
)
# Use strategy.kernel(aerosol_particle, 298.15, 101325) to get the
# sedimentation kernel.
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
  Physics, Chapter 13, Equation 13A.4, Wiley.

#### Signature

```python
class SedimentationCoagulationStrategy(CoagulationStrategyABC):
    def __init__(self, distribution_type: str): ...
```

#### See also

- [CoagulationStrategyABC](./coagulation_strategy_abc.md#coagulationstrategyabc)

### SedimentationCoagulationStrategy().dimensionless_kernel

[Show source in sedimentation_coagulation_strategy.py:71](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/sedimentation_coagulation_strategy.py#L71)

Raise NotImplementedError for dimensionless kernel in sedimentation
strategy.

This method is not applicable to sedimentation-based collisions.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio
  [dimensionless].

#### Raises

- NotImplementedError : Always raised for this strategy.

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### SedimentationCoagulationStrategy().kernel

[Show source in sedimentation_coagulation_strategy.py:97](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/sedimentation_coagulation_strategy.py#L97)

Compute the sedimentation coagulation kernel [m^3/s].

Uses the Seinfeld & Pandis (2016) sedimentation kernel via
`get_sedimentation_kernel_sp2016_via_system_state`.

#### Arguments

- particle : The ParticleRepresentation providing particle radius
  and density.
- temperature : The system temperature [K].
- pressure : The system pressure [Pa].

#### Returns

- The sedimentation coagulation kernel.

#### Examples

```py
k_values = strategy.kernel(
    ParticleRepresentation, temperature=298.15, pressure=101325
)
# k_values may be a single float or an array
```

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)


---
# turbulent_dns_coagulation_strategy.md

# TurbulentDNSCoagulationStrategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Strategy](./index.md#coagulation-strategy) / TurbulentDNSCoagulationStrategy

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py) module.

## TurbulentDNSCoagulationStrategy

[Show source in turbulent_dns_coagulation_strategy.py:26](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L26)

Turbulent DNS coagulation strategy for aerosols.

Implements methods from `CoagulationStrategyABC`, applying the
turbulent DNS kernel following Ayala et al. (2008). Suitable for
coagulation of particles larger than 1 µm in turbulent flow fields.

#### Attributes

- distribution_type : The particle distribution type ("discrete",
  "continuous_pdf", or "particle_resolved").
- turbulent_dissipation : Turbulent kinetic energy dissipation
  [m^2/s^3] used in DNS fits (examples: 0.001, 0.01, 0.04).
- fluid_density : The fluid (air) density [kg/m^3].
- reynolds_lambda : Reynolds lambda of air (e.g., 23 or 74).
- relative_velocity : Relative velocity of the air [m/s] for
  collisions.

#### Methods

- set_turbulent_dissipation : Change turbulent dissipation rate.
- set_reynolds_lambda : Update the Reynolds lambda.
- set_relative_velocity : Update the relative velocity.
- dimensionless_kernel : Raise NotImplementedError for DNS approach.
- kernel : Return the DNS-based coagulation kernel.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

#### Examples

```py title="Example usage of TurbulentDNSCoagulationStrategy"
import particula as par
strategy = par.dynamics.TurbulentDNSCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=0.01,
    fluid_density=1.225,
    reynolds_lambda=23,
    relative_velocity=0.5
)
# Use strategy.kernel(...) to compute the DNS-based kernel
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the
  geometric collision rate of sedimenting droplets. Part 2. Theory and
  parameterization. New Journal of Physics, 10.
  [DOI](https://doi.org/10.1088/1367-2630/10/7/075016)

#### Signature

```python
class TurbulentDNSCoagulationStrategy(CoagulationStrategyABC):
    def __init__(
        self,
        distribution_type: str,
        turbulent_dissipation: float,
        fluid_density: float,
        reynolds_lambda: float,
        relative_velocity: float,
    ): ...
```

#### See also

- [CoagulationStrategyABC](./coagulation_strategy_abc.md#coagulationstrategyabc)

### TurbulentDNSCoagulationStrategy().dimensionless_kernel

[Show source in turbulent_dns_coagulation_strategy.py:165](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L165)

Compute or return the dimensionless kernel (H).

Not implemented for DNS-based approaches, so raises
NotImplementedError.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number(s)
  [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio(s)
  [dimensionless].

#### Returns

- None : Raises NotImplementedError instead.

#### Raises

- NotImplementedError : This strategy does not support
dimensionless kernels.

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### TurbulentDNSCoagulationStrategy().kernel

[Show source in turbulent_dns_coagulation_strategy.py:196](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L196)

Compute the DNS-based coagulation kernel [m^3/s].

Uses the `get_turbulent_dns_kernel_ao2008_via_system_state` function to
calculate collision rates following Ayala et al. (2008). This approach
accounts for turbulent dissipation, fluid density, Reynolds lambda,
and relative velocity.

#### Arguments

- particle : The ParticleRepresentation whose radii and density
  are needed.
- temperature : The temperature of the system [K].
- pressure : The system pressure [Pa] (unused here, but included
  for interface consistency).

#### Returns

- The DNS-based coagulation kernel(s).

#### Examples

```py
kernel_values = strategy.kernel(
    particle=my_particle,
    temperature=298.15,
    pressure=101325
)
# kernel_values may be a float or array, depending on the
# distribution
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence
  on the geometric collision rate of sedimenting droplets. Part 2.
  New Journal of Physics, 10.
  [DOI](https://doi.org/10.1088/1367-2630/10/7/075016)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### TurbulentDNSCoagulationStrategy().set_relative_velocity

[Show source in turbulent_dns_coagulation_strategy.py:147](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L147)

Set the relative velocity of the flow [m/s].

#### Arguments

- relative_velocity : Relative velocity in [m/s].

#### Returns

- TurbulentDNSCoagulationStrategy : Self, for method chaining.

#### Examples

```py
strategy.set_relative_velocity(0.8)
```

#### Signature

```python
def set_relative_velocity(self, relative_velocity: float): ...
```

### TurbulentDNSCoagulationStrategy().set_reynolds_lambda

[Show source in turbulent_dns_coagulation_strategy.py:129](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L129)

Set the Reynolds lambda value.

#### Arguments

- reynolds_lambda : Reynolds lambda [dimensionless].

#### Returns

- TurbulentDNSCoagulationStrategy : Self, for method chaining.

#### Examples

```py
strategy.set_reynolds_lambda(74)
```

#### Signature

```python
def set_reynolds_lambda(self, reynolds_lambda: float): ...
```

### TurbulentDNSCoagulationStrategy().set_turbulent_dissipation

[Show source in turbulent_dns_coagulation_strategy.py:111](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L111)

Set the turbulent kinetic energy dissipation rate.

#### Arguments

- turbulent_dissipation : Turbulent dissipation [m^2/s^3].

#### Returns

- TurbulentDNSCoagulationStrategy : Self, allowing method chaining.

#### Examples

```py
strategy.set_turbulent_dissipation(0.02)
```

#### Signature

```python
def set_turbulent_dissipation(self, turbulent_dissipation: float): ...
```


---
# turbulent_shear_coagulation_strategy.md

# TurbulentShearCoagulationStrategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Coagulation Strategy](./index.md#coagulation-strategy) / TurbulentShearCoagulationStrategy

> Auto-generated documentation for [particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py) module.

## TurbulentShearCoagulationStrategy

[Show source in turbulent_shear_coagulation_strategy.py:29](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L29)

Turbulent shear coagulation strategy for aerosol particles.

Implements the Saffman & Turner (1956) turbulent shear coagulation kernel,
extending the base `CoagulationStrategyABC` class to provide a physically
consistent model of coagulation in turbulent flow.

#### Attributes

- distribution_type : The type of particle distribution for coagulation
  ("discrete", "continuous_pdf", or "particle_resolved").
- turbulent_dissipation : Turbulent kinetic energy dissipation
  rate [m^2/s^3].
- fluid_density : Fluid density [kg/m^3].

#### Methods

- set_turbulent_dissipation : Set the turbulent kinetic energy dissipation
  rate.
- dimensionless_kernel : (Not implemented here) Raise NotImplementedError.
- kernel : Compute the turbulent shear coagulation kernel via
  Saffman-Turner approach.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

#### Examples

```py title="Example usage of TurbulentShearCoagulationStrategy"
import particula as par
strategy = par.dynamics.TurbulentShearCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=0.01,
    fluid_density=1.225,
)
# Use strategy.kernel(...) to get the coagulation kernel
```

#### References

- Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
  turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
  https://doi.org/10.1017/S0022112056000020

#### Signature

```python
class TurbulentShearCoagulationStrategy(CoagulationStrategyABC):
    def __init__(
        self, distribution_type: str, turbulent_dissipation: float, fluid_density: float
    ): ...
```

#### See also

- [CoagulationStrategyABC](./coagulation_strategy_abc.md#coagulationstrategyabc)

### TurbulentShearCoagulationStrategy().dimensionless_kernel

[Show source in turbulent_shear_coagulation_strategy.py:119](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L119)

Compute a dimensionless kernel (H).

Not implemented for turbulent shear; raises NotImplementedError.

#### Arguments

- diffusive_knudsen : The diffusive Knudsen number [dimensionless].
- coulomb_potential_ratio : The Coulomb potential ratio
  [dimensionless].

#### Returns

- NDArray[np.float64] : Not returned; raises error instead.

#### Examples

```py
# This method is not supported here
try:
    result = strategy.dimensionless_kernel(diff_kn, phi_ratio)
except NotImplementedError:
    print("Not implemented for turbulent shear strategy.")
```

#### References

- Saffman & Turner (1956) used dimensional forms; dimensionless
  form is not covered.

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### TurbulentShearCoagulationStrategy().kernel

[Show source in turbulent_shear_coagulation_strategy.py:157](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L157)

Compute the dimensioned turbulent shear coagulation kernel [m^3/s].

Uses the system state to calculate the Saffman-Turner (1956) kernel,
which depends on the dissipation rate of turbulent kinetic energy,
fluid density, and particle radius.

#### Arguments

- particle : The ParticleRepresentation instance to retrieve
  particle radii.
- temperature : The system temperature [K].
- pressure : The system pressure [Pa].

#### Returns

- float or NDArray[np.float64] : The coagulation kernel(s) [m^3/s].

#### Examples

```py title="Example usage of kernel method"
kernel_value = strategy.kernel(
    particle=ParticleRepresentation(...),
    temperature=298.15,
    pressure=101325
)
# kernel_value could be a float or array depending on the
# particle representation
```

#### References

- Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
  in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
  https://doi.org/10.1017/S0022112056000020

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../../particles/representation.md#particlerepresentation)

### TurbulentShearCoagulationStrategy().set_turbulent_dissipation

[Show source in turbulent_shear_coagulation_strategy.py:100](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L100)

Set the turbulent kinetic energy dissipation rate.

#### Arguments

- turbulent_dissipation : Turbulent kinetic energy dissipation
  rate [m^2/s^3].

#### Returns

- Self (TurbulentShearCoagulationStrategy)

#### Examples

```py
strategy.set_turbulent_dissipation(0.02)
```

#### Signature

```python
def set_turbulent_dissipation(self, turbulent_dissipation: float): ...
```


---
# particle_resolved_method.md

# Particle Resolved Method

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Particle Resolved Step](./index.md#particle-resolved-step) / Particle Resolved Method

> Auto-generated documentation for [particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py) module.

## _calculate_probabilities

[Show source in particle_resolved_method.py:312](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L312)

Calculate coagulation probabilities based on kernel values and system
parameters.

This function multiplies the kernel values by the time step and a factor
derived from the ratio of (events / tests) over the volume to obtain the
probability of coagulation.

#### Arguments

- kernel_values : Interpolated kernel values for a given particle pair,
  may be scalar or array.
- time_step : Duration of one coagulation step in seconds.
- events : Number of possible collisions for the pair(s).
- tests : Number of trials for the random selection procedure.
- volume : System volume in m³.

#### Returns

- The probability (or array of probabilities) that a collision occurs
  during this time step.

#### Examples

```py
prob = _calculate_probabilities(0.5, 1.0, 20, 10, 1e-3)
# prob ~ 0.5 * 1.0 * 20 / (10 * 1e-3) = 1000
```

#### Signature

```python
def _calculate_probabilities(
    kernel_values: Union[float, NDArray[np.float64]],
    time_step: float,
    events: int,
    tests: int,
    volume: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## _final_coagulation_state

[Show source in particle_resolved_method.py:348](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L348)

Resolve the final state of particles that have undergone multiple
coagulation events.

This function ensures that each small particle index merges correctly to
a final large particle index, preventing logical conflicts (e.g., a single
particle merging into multiple large particles in the same step).

#### Arguments

- small_indices : Array of smaller particle indices in coagulation.
- large_indices : Array of larger particle indices in coagulation.
- particle_radius : Array of current particle radii.

#### Returns

- A tuple (updated_small_indices, updated_large_indices) that resolves
  multiple merges for the same particle.

#### Examples

```py
import numpy as np
small = np.array([0, 1, 2])
large = np.array([2, 3, 4])
r = np.array([1e-9, 1.5e-9, 2e-9, 3e-9, 4e-9])
s_final, l_final = _final_coagulation_state(small, large, r)
# ensures each index in s_final merges to a single large index
```

#### Signature

```python
def _final_coagulation_state(
    small_indices: NDArray[np.int64],
    large_indices: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## _interpolate_kernel

[Show source in particle_resolved_method.py:267](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L267)

Create an interpolation function for the coagulation kernel with
out-of-bounds handling.

This function returns a RegularGridInterpolator that performs linear
interpolation for values within the domain of the kernel and clamps to the
nearest value outside of it.

#### Arguments

- kernel : 2D coagulation kernel values.
- kernel_radius : Radii corresponding to kernel bins.

#### Returns

- A RegularGridInterpolator object for retrieving kernel values based
    on radius pairs.

#### Examples

```py
import numpy as np
from particula.dynamics.coagulation.particle_resolved_step import
    particle_resolved_method
kernel_vals = np.random.rand(10,10)
rad = np.linspace(1e-9, 1e-7, 10)
interpolator = particle_resolved_method._interpolate_kernel(
    kernel_vals, rad
)
# Use interpolator([[r_small, r_large]]) to get kernel value
```

#### Signature

```python
def _interpolate_kernel(
    kernel: NDArray[np.float64], kernel_radius: NDArray[np.float64]
) -> RegularGridInterpolator: ...
```



## get_particle_resolved_coagulation_step

[Show source in particle_resolved_method.py:88](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L88)

Perform a single step of particle coagulation, updating particle radii
with a stochastic approach.

This function models collisions between particles based on a given
coagulation kernel. It identifies potential collision pairs, randomly
selects which collisions occur according to a probability derived from the
kernel value, and then tracks which particles have coagulated.

The main calculation for the probability of coagulation is:

- Probability = K × Δt × (possible collisions) / (tests × volume)
    - K is the interpolated kernel value,
    - Δt is the timestep,
    - volume is the system volume.

#### Arguments

- particle_radius : Array of particle radii.
- kernel : 2D coagulation kernel matrix matching the size of
    kernel_radius.
- kernel_radius : Radii used to index or interpolate the kernel.
- volume : Volume of the system in m³.
- time_step : Time step for each coagulation iteration in seconds.
- random_generator : Random number generator for the stochastic
    approach.

#### Returns

- An array of shape (N, 2), where each row contains
    [small_index, large_index] for coagulation events.

#### Examples

```py title="Example Usage"
import numpy as np
from particula.dynamics.coagulation.particle_resolved_step import
    particle_resolved_method

r = np.array([1e-9, 2e-9, 3e-9])
kernel_values = np.ones((50, 50))
kernel_r = np.linspace(1e-10, 1e-7, 50)
vol = 1e-3
dt = 0.01
rng = np.random.default_rng(42)
event_pairs =
particle_resolved_method.get_particle_resolved_coagulation_step(
    particle_radius=r,
    kernel=kernel_values,
    kernel_radius=kernel_r,
    volume=vol,
    time_step=dt,
    random_generator=rng
)
# event_pairs contains the pairs of [small, large] indices that
# coagulated.

References:
- Seinfeld, J. H., & Pandis, S. N. *Atmospheric Chemistry and Physics*,
  Wiley, 2016.

#### Signature

```python
def get_particle_resolved_coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> NDArray[np.int64]: ...
```



## get_particle_resolved_update_step

[Show source in particle_resolved_method.py:14](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L14)

Update particle radii and track lost/gained particles after coagulation
events.

This function simulates the immediate effect of coagulation on particle
radii, marking smaller particles as lost and updating the larger particles
to the new radius computed from volume conservation. The calculation is:

- r_new = cbrt(r_small³ + r_large³)
    - r_new is the new radius in meters,
    - r_small is the smaller particle's radius in meters,
    - r_large is the larger particle's radius in meters.

#### Arguments

- particle_radius : Array of particle radii.
- loss : Array to store lost particle radii.
- gain : Array to store gained particle radii.
- small_index : Indices of smaller particles.
- large_index : Indices of larger particles.

#### Returns

- Updated array of particle radii after coagulation events.
- Updated array for the radii of particles that were lost.
- Updated array for the radii of particles that were gained.

#### Examples

```py title="Example Usage"
import numpy as np
from particula.dynamics.coagulation.particle_resolved_step import
    particle_resolved_method

r = np.array([1e-9, 2e-9, 3e-9, 1e-9])
lost = np.zeros_like(r)
gained = np.zeros_like(r)
s_idx = np.array([0, 1])
l_idx = np.array([2, 3])
updated_r, lost_r, gained_r = (
    particle_resolved_method.get_particle_resolved_update_step(
        r, lost, gained, s_idx, l_idx
    ))
# updated_r now has coagulated radii, lost_r and gained_r are tracked.

#### Signature

```python
def get_particle_resolved_update_step(
    particle_radius: NDArray[np.float64],
    loss: NDArray[np.float64],
    gain: NDArray[np.float64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
```


---
# super_droplet_method.md

# Super Droplet Method

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Particle Resolved Step](./index.md#particle-resolved-step) / Super Droplet Method

> Auto-generated documentation for [particula.dynamics.coagulation.particle_resolved_step.super_droplet_method](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py) module.

## _bin_particles

[Show source in super_droplet_method.py:541](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L541)

Divide the sorted particle radii into bins and count how many fall into
each bin.

This function uses `radius_bins` as edges and assigns each particle
radius to a bin index via `np.digitize`. The result is (1) a histogram
with the number of particles in each bin, and (2) an array of per-particle
bin indices.

#### Arguments

- particle_radius : Array of sorted particle radii.
- radius_bins : Edges used to define the bins.

#### Returns

- `number_in_bins` : Counts of how many radii lie in each bin.
- `bin_indices` : The bin index assigned to each particle.

#### Examples

```py
import numpy as np
rad = np.array([1e-9, 1.5e-9, 2e-9, 5e-9])
bin_edges = np.array([1e-9, 2e-9, 3e-9, 1e-8])
n_in_bins, bin_idx = _bin_particles(rad, bin_edges)
# n_in_bins -> [1, 2, 1]
# bin_idx might be [0, 1, 1, 2]
```

#### Signature

```python
def _bin_particles(
    particle_radius: NDArray[np.float64], radius_bins: NDArray[np.float64]
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## _bin_to_particle_indices

[Show source in super_droplet_method.py:327](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L327)

Map bin-relative indices back to absolute positions in the particle array.

This function adjusts the offsets for each bin so that the pairwise
indices used for collision are mapped onto the actual sorted particle
array. For instance, if `lower_indices` are all within bin 0, and bin 0
particles occupy positions [0..9], this method adds that offset to
each index in `lower_indices`.

#### Arguments

- lower_indices : Relative indices (local to the bin) of smaller
    particles.
- upper_indices : Relative indices (local to the bin) of larger
    particles.
- lower_bin : The bin representing the smaller particles.
- upper_bin : The bin representing the larger particles.
- bin_indices : Cumulative offsets to determine where each bin begins.

#### Returns

- `small_index` : Absolute positions of smaller particles in the
  sorted particle array.
- `large_index` : Absolute positions of the larger particles in
  the sorted array.

#### Examples

```py
bins = np.array([0, 10, 20])
lw_rel = np.array([0, 1])
up_rel = np.array([2, 3])
# Convert these local indices for bin 1 (start=10) and bin 2 (start=20)
s_idx, l_idx = _bin_to_particle_indices(lw_rel, up_rel, 1, 2, bins)
# s_idx -> [10, 11]
# l_idx -> [22, 23]

#### Signature

```python
def _bin_to_particle_indices(
    lower_indices: NDArray[np.int64],
    upper_indices: NDArray[np.int64],
    lower_bin: int,
    upper_bin: int,
    bin_indices: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## _calculate_concentration_in_bins

[Show source in super_droplet_method.py:613](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L613)

Sum the particle concentrations in each bin.

Given per-particle `bin_indices` and `particle_concentration`, this
function accumulates the total concentration of all particles that
fall into each bin. The `number_in_bins` array is used mainly for
shape reference but can also confirm the count of particles.

#### Arguments

- bin_indices : Array of bin indices for each particle.
- particle_concentration : 1D array of concentrations matching
  each particle.
- number_in_bins : Array with the count of particles in each bin.

#### Returns

- A 1D array whose length is the number of unique bins, containing
  the summed concentration per bin.

#### Examples

```py
import numpy as np
b_idx = np.array([0, 0, 1, 1, 2])
conc = np.array([10., 5., 2., 3., 4.])
n_in_bins = np.array([2, 2, 1])  # might match the bin partition
bin_c = _calculate_concentration_in_bins(b_idx, conc, n_in_bins)
# bin_c -> [15., 5., 4.]
```

#### Signature

```python
def _calculate_concentration_in_bins(
    bin_indices: NDArray[np.int64],
    particle_concentration: NDArray[np.float64],
    number_in_bins: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## _coagulation_events

[Show source in super_droplet_method.py:438](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L438)

Stochastically pick which collisions (among possible pairs) actually happen.

This function computes a collision probability for each `(small_index,
large_index)` pair by taking the ratio of `kernel_values / kernel_max`.
Next, a random uniform draw decides if each collision occurs.

#### Arguments

- small_index : Array of indices representing smaller particles.
- large_index : Array of indices representing larger particles.
- kernel_values : Collision kernel values for each pair.
- kernel_max : A maximum kernel value used for normalization.
- generator : Random generator to compare probabilities vs.
  uniform draws.

#### Returns

- Filtered `small_index` containing only those that coagulated.
- Filtered `large_index` containing only those that coagulated.

#### Examples

```py
rng = np.random.default_rng(999)
s_idx = np.array([0, 1, 2])
l_idx = np.array([3, 4, 5])
kv = np.array([0.5, 1.0, 0.1])
kmax = 1.0
s_new, l_new = _coagulation_events(s_idx, l_idx, kv, kmax, rng)
# Each pair has probability kv/kmax => [0.5, 1.0, 0.1]
# The final s_new, l_new depends on random draws
```

#### Signature

```python
def _coagulation_events(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    kernel_values: NDArray[np.float64],
    kernel_max: float,
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## _event_pairs

[Show source in super_droplet_method.py:119](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L119)

Calculate an approximate count of particle-pair interactions.

This function estimates the number of collisions or interactions
that might occur between two bins of particles, given a maximum
kernel value and the current population of each bin. When the bins
are the same, a correction factor is applied to avoid double-counting
pairs.

#### Arguments

- lower_bin : Index of the lower bin in the distribution.
- upper_bin : Index of the upper bin in the distribution.
- kernel_max : Maximum kernel value used to weight collisions.
- number_in_bins : The population of particles per bin.

#### Returns

- A float representing the expected number of particle-pair
  collision events.

#### Examples

```py
max_kernel = 1.0e-9
n_bins = np.array([100, 150, 200])
# lower_bin=0, upper_bin=1 => collisions between bin 0 and bin 1
events_est = _event_pairs(0, 1, max_kernel, n_bins)
# events_est is ~ 1.0e-9 * 100 * 150
```

#### Signature

```python
def _event_pairs(
    lower_bin: int,
    upper_bin: int,
    kernel_max: Union[float, NDArray[np.float64]],
    number_in_bins: Union[NDArray[np.float64], NDArray[np.int64]],
) -> float: ...
```



## _filter_valid_indices

[Show source in super_droplet_method.py:380](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L380)

Remove invalid pairs of particles based on radius and optional event limit.

This function checks each pair of `(small_index, large_index)` to ensure
both have radius > 0. If `single_event_counter` is provided, it further
enforces that each particle has had < 1 event so far (or you can
define your own threshold). The pairs failing these checks are removed.

#### Arguments

- small_index : Indices for the smaller particles in each pair.
- large_index : Indices for the larger particles in each pair.
- particle_radius : Array of radii for each particle.
- single_event_counter : Optional array counting how many events
  each particle has undergone. If provided, only particles with
  counter < 1 pass the filter.

#### Returns

- Filtered `small_index` with only valid pairs.
- Filtered `large_index` that corresponds to valid pairs.

#### Examples

```py
r = np.array([0.1, 0.0, 0.08, 0.02])
c = np.array([0, 0, 0, 0])
small_i = np.array([0, 1, 2])
large_i = np.array([3, 0, 1])
# Filter out pairs with radius <= 0 or event_counter >= 1
s_valid, l_valid = _filter_valid_indices(
    small_i, large_i, r, single_event_counter=c
)
# Indices with r>0 remain in s_valid, l_valid
```

#### Signature

```python
def _filter_valid_indices(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
    single_event_counter: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## _get_bin_pairs

[Show source in super_droplet_method.py:583](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L583)

Produce the list of all unique (binA, binB) pairs using combinations
with replacement.

This function is useful when we want to iterate over all bin pairs
(including binA == binB) for collision computations. The combination
ensures each pair is returned only once.

#### Arguments

- bin_indices : Array of bin indices for each particle (though
  only the unique values matter).

#### Returns

- A list of (lower_bin, upper_bin) pairs covering all unique
  bins in `bin_indices`.

#### Examples

```py
bins = np.array([0, 0, 1, 2, 2])
pairs = _get_bin_pairs(bins)
# pairs -> [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
```

#### Signature

```python
def _get_bin_pairs(bin_indices: NDArray[np.int64]) -> list[Tuple[int, int]]: ...
```



## _sample_events

[Show source in super_droplet_method.py:166](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L166)

Determine how many collisions actually occur using a Poisson draw.

This function uses the expected collision count (`events`) and normalizes
by system `volume` to compute an effective collision rate. It then
samples from a Poisson distribution to obtain the actual number of
collisions happening within the current `time_step`.

#### Arguments

- events : The calculated number of particle pairs that could
  interact.
- volume : The volume of the simulation space (m³).
- time_step : The time span (seconds) over which collisions are
  considered.
- generator : A NumPy random Generator to sample the Poisson
  distribution.

#### Returns

- The sampled number of coagulation events as an integer.

#### Examples

```py
from numpy.random import default_rng
rng = default_rng(42)
collisions = _sample_events(events=5e3, volume=0.1, time_step=0.01,
    generator=rng)
# collisions might be ~ Poisson( 5e3 / 0.1 * 0.01 ) => Poisson(5)
```

#### Signature

```python
def _sample_events(
    events: float, volume: float, time_step: float, generator: np.random.Generator
) -> int: ...
```



## _select_random_indices

[Show source in super_droplet_method.py:267](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L267)

Randomly choose indices within each bin to represent collision partners.

This function picks `events` indices from the population of the
`lower_bin` and `upper_bin`, ignoring any radius or event-limit checks
(those may happen later). The result is two arrays of equal size,
each containing random picks within the respective bins.

#### Arguments

- lower_bin : Index for the "smaller" bin.
- upper_bin : Index for the "larger" bin.
- events : How many pairs to select.
- number_in_bins : Array with the count of particles in each bin.
- generator : Random number generator to draw the indices.

#### Returns

- An array of size `events` with random picks from `lower_bin`.
- An array of size `events` with random picks from `upper_bin`.

#### Examples

```py
import numpy as np
rng = np.random.default_rng(42)
n_in_bins = np.array([5, 10, 7])
i_lw, i_up = _select_random_indices(
    lower_bin=0,
    upper_bin=2,
    events=3,
    number_in_bins=n_in_bins,
    generator=rng
)
# i_lw -> random indices in [0..4]
# i_up -> random indices in [0..6]

#### Signature

```python
def _select_random_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    number_in_bins: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## _sort_particles

[Show source in super_droplet_method.py:491](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L491)

Sort particle radii (and optionally concentrations) in ascending order.

The function returns an array of `unsort_indices` that can be used
to restore the particles to their original order after manipulations.

#### Arguments

- particle_radius : 1D NumPy array of particle radii.
- particle_concentration : Optional array of corresponding
  concentrations.

#### Returns

- `unsort_indices` : Indices to revert sorting to the original order.
- `sorted_radius` : Sorted array of radii in ascending order.
- `sorted_concentration` : Sorted array of concentrations, if
  provided; otherwise `None`.

#### Examples

```py
import numpy as np
r = np.array([0.3, 0.1, 0.5])
c = np.array([10, 30, 20])
u_idx, s_r, s_c = _sort_particles(r, c)
# s_r -> [0.1, 0.3, 0.5]
# s_c -> [30, 10, 20]
# u_idx can be used to get them back in [0.3, 0.1, 0.5] order
```

#### Signature

```python
def _sort_particles(
    particle_radius: NDArray[np.float64],
    particle_concentration: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.float64], Optional[NDArray[np.float64]]]: ...
```



## _super_droplet_update_step

[Show source in super_droplet_method.py:15](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L15)

Update particle radii and concentrations when two particles coagulate.

This function merges smaller and larger particles by combining their
volumes and redistributing particle concentrations. The resulting
radii are computed via volume conservation, and an event counter
tracks how many coagulation events each particle has undergone.

#### Arguments

- particle_radius : Array of particle radii (m).
- concentration : Array representing the concentration of each
  particle (number or mass, depending on usage).
- single_event_counter : Tracks the number of coagulation events
  each particle has experienced in the current iteration.
- small_index : Indices for smaller particles in a coagulation event.
- large_index : Indices for larger particles in a coagulation event.

#### Returns

- An updated array of particle radii (m) following coagulation.
- An updated array representing the concentration of particles.
- An updated array tracking the index-wise number of events.

#### Examples

```py
import numpy as np
r = np.array([1e-9, 2e-9, 3e-9])
conc = np.array([100., 50., 75.])
events = np.zeros_like(r, dtype=int)
s_idx = np.array([0])
l_idx = np.array([2])
out_r, out_c, out_ev = _super_droplet_update_step(
    r, conc, events, s_idx, l_idx)
# out_r[0] is updated via volume combination with out_r[2].
```

#### Signature

```python
def _super_droplet_update_step(
    particle_radius: NDArray[np.float64],
    concentration: NDArray[np.float64],
    single_event_counter: NDArray[np.int64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]: ...
```



## get_super_droplet_coagulation_step

[Show source in super_droplet_method.py:658](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L658)

Carry out one time-step of super-droplet-based coagulation.

This function sorts particles by radius, bins them, and then stochastically
computes collision events according to the coagulation kernel. It updates
the particle radii/concentrations, then unsorts them back to the original
order.

#### Arguments

- particle_radius : Array of particle radii (m).
- particle_concentration : Array of per-particle concentration.
- kernel : 2D matrix of coagulation kernel values, dimension
  ~ len(kernel_radius) × len(kernel_radius).
- kernel_radius : Array of radius points defining the kernel dimension.
- volume : System volume or domain size in m³.
- time_step : The length of this coagulation iteration in seconds.
- random_generator : Random number generator for sampling collisions.

#### Returns

- Updated radii array after processing coagulation.
- Updated concentrations array after processing coagulation.

#### Examples

```py
import numpy as np
from numpy.random import default_rng
radius = np.array([1e-9, 2e-9, 5e-9])
conc = np.array([100., 50., 10.])
ker_vals = np.ones((3,3))
ker_r = np.array([1e-9, 2e-9, 5e-9])
rng = default_rng(42)
r_new, c_new = get_super_droplet_coagulation_step(
    radius, conc, ker_vals, ker_r, 1e-3, 1.0, rng)
# r_new, c_new have updated values after one super droplet
# coagulation step.

References:
- E. W. Tedford and L. A. Perugini, "Superdroplet method
  in cloud microphysics simulations," J. Atmos. Sci., 2020.
- Seinfeld, J. H., & Pandis, S. N. *Atmospheric Chemistry and Physics*,
  Wiley, 2016.

#### Signature

```python
def get_super_droplet_coagulation_step(
    particle_radius: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```



## random_choice_indices

[Show source in super_droplet_method.py:209](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L209)

Select valid particle indices in two bins for coagulation events.

This function tries to choose `events` valid indices from
`lower_bin` and `upper_bin`, discarding any particles with radius ≤ 0.
It uses the provided random generator to perform the sampling
with replacement if needed.

#### Arguments

- lower_bin : Index of the lower bin to filter particles from.
- upper_bin : Index of the upper bin to filter particles from.
- events : Number of events (indices) to sample for each bin.
- particle_radius : Array of particle radii; only those > 0
  are considered valid.
- bin_indices : Array of bin labels corresponding to each particle.
- generator : Random number generator used for index selection.

#### Returns

- Indices of particles from the lower bin.
- Indices of particles from the upper bin.

#### Examples

```py
import numpy as np
rng = np.random.default_rng(123)
radius = np.array([0.3, 0.1, 0.0, 0.5])
bins = np.array([0, 0, 1, 1])
lw_bin, up_bin = random_choice_indices(0, 1, 2, radius, bins, rng)
# lw_bin -> array of valid picks from bin 0
# up_bin -> array of valid picks from bin 1
```

#### Signature

```python
def random_choice_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    particle_radius: NDArray[np.float64],
    bin_indices: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```


---
# sedimentation_kernel.md

# Sedimentation Kernel

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Sedimentation Kernel

> Auto-generated documentation for [particula.dynamics.coagulation.sedimentation_kernel](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py) module.

## calculate_collision_efficiency_function

[Show source in sedimentation_kernel.py:29](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py#L29)

Calculate the collision efficiency between two particles (placeholder).

This function calculates the collision efficiency E for two particles
of radii radius1 and radius2, which can depend on additional factors
(e.g., fluid flow or electrostatic forces). Currently not implemented.

#### Arguments

- radius1 : The radius of the first particle [m].
- radius2 : The radius of the second particle [m].

#### Returns

- Collision efficiency [dimensionless].

#### Examples

```py
# Not implemented
calculate_collision_efficiency_function(1e-7, 2e-7)
# Raises NotImplementedError
```

#### References

- Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
  in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.

#### Signature

```python
def calculate_collision_efficiency_function(radius1: float, radius2: float) -> float: ...
```



## get_sedimentation_kernel_sp2016

[Show source in sedimentation_kernel.py:63](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py#L63)

Calculate the sedimentation kernel for aerosol particles (Equation 13A.4).

This function computes the coagulation kernel due to gravitational
settling, where larger particles settle faster and overtake smaller
ones. The kernel is based on the combined diameters, the settling
velocity difference, and the collision efficiency.

Equation:
- K(i, j) = (π / 4) × (Dᵢ + Dⱼ)² × |vᵢ - vⱼ| × Eᵢⱼ
    - Dᵢ, Dⱼ : diameters of particle i and j [m],
    - vᵢ, vⱼ : settling velocities [m/s],
    - Eᵢⱼ : collision efficiency (dimensionless).

#### Arguments

- particle_radius : Array of particle radii [m].
- settling_velocities : Array of particle settling velocities [m/s].
- calculate_collision_efficiency : Whether to calculate collision
  efficiency or use 1. Defaults to True.

#### Returns

- Sedimentation kernel matrix [m³/s], shape (n, n).

#### Examples

```py title="Example"
import numpy as np

rads = np.array([1e-7, 2e-7])
vels = np.array([5e-3, 1e-2])
kernel = get_sedimentation_kernel_sp2016(rads, vels)
print(kernel)
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
  and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.

#### Signature

```python
def get_sedimentation_kernel_sp2016(
    particle_radius: NDArray[np.float64],
    settling_velocities: NDArray[np.float64],
    calculate_collision_efficiency: bool = True,
) -> NDArray[np.float64]: ...
```



## get_sedimentation_kernel_sp2016_via_system_state

[Show source in sedimentation_kernel.py:128](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/sedimentation_kernel.py#L128)

Calculate the sedimentation kernel (Equation 13A.4) via system state.

This function first derives settling velocities using the system state
(particle radius, density, temperature, pressure), then calls
get_sedimentation_kernel_sp2016.

#### Arguments

- particle_radius : Array of particle radii [m].
- particle_density : Array of particle densities [kg/m³].
- temperature : Temperature [K].
- pressure : Pressure [Pa].
- calculate_collision_efficiency : Whether to calculate collision
  efficiency or use 1. Defaults to True.

#### Returns

- Sedimentation kernel matrix [m³/s], shape (n, n).

#### Examples

```py title="Example"
import numpy as np
rads = np.array([1e-7, 2e-7])
dens = np.array([1000, 1200])
kernel = get_sedimentation_kernel_sp2016_via_system_state(
    particle_radius=rads,
    particle_density=dens,
    temperature=298,
    pressure=101325
)
print(kernel)
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry
  and physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.4.

#### Signature

```python
@validate_inputs({"temperature": "positive", "pressure": "positive"})
def get_sedimentation_kernel_sp2016_via_system_state(
    particle_radius: NDArray[np.float64],
    particle_density: NDArray[np.float64],
    temperature: float,
    pressure: float,
    calculate_collision_efficiency: bool = True,
) -> NDArray[np.float64]: ...
```


---
# g12_radial_distribution_ao2008.md

# G12 Radial Distribution Ao2008

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / G12 Radial Distribution Ao2008

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py) module.

## _calculate_c1

[Show source in g12_radial_distribution_ao2008.py:122](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L122)

Compute C_1 based on Stokes number and turbulence properties.

C_1 is given by:

C_1 = y(St) / (|g| / (v_k / τ_k))^f_3(R_λ)

- y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St
- f_3(R_λ) = 0.1886 * exp(20.306 / R_λ)
- |g| : Gravitational acceleration [m/s²]
- v_k : Kolmogorov velocity scale [m/s]
- τ_k : Kolmogorov timescale [s]

#### Signature

```python
def _calculate_c1(
    stokes_number: NDArray[np.float64],
    reynolds_lambda: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> NDArray[np.float64]: ...
```



## _calculate_rc

[Show source in g12_radial_distribution_ao2008.py:174](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L174)

Compute r_c, the turbulence-driven correction to the collision kernel.

The equation is:

(r_c / η)^2 = |St_2 - St_1| * F(a_Og, R_λ)

#### Signature

```python
def _calculate_rc(
    stokes_diff_matrix: NDArray[np.float64],
    kolmogorov_length_scale: float,
    normalized_accel_variance: float,
    reynolds_lambda: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> NDArray[np.float64]: ...
```



## _compute_a_og

[Show source in g12_radial_distribution_ao2008.py:201](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L201)

Compute aOg, which accounts for the effect of gravity on
turbulence-driven clustering.

- a_Og = a_o + (π / 8) * (|g| / (v_k / τ_k))^2

#### Signature

```python
def _compute_a_og(
    normalized_accel_variance: float, kolmogorov_velocity: float, kolmogorov_time: float
) -> float: ...
```



## _compute_f

[Show source in g12_radial_distribution_ao2008.py:218](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L218)

Compute F(aOg, R_lambda), an empirical scaling factor for
turbulence effects.

- F(a_Og, R_λ) = 20.115 * (a_Og / R_λ)^0.5

#### Signature

```python
def _compute_f(a_og: float, reynolds_lambda: float) -> float: ...
```



## _compute_f3_lambda

[Show source in g12_radial_distribution_ao2008.py:169](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L169)

Compute f_3(R_lambda), an empirical turbulence factor.

#### Signature

```python
def _compute_f3_lambda(reynolds_lambda: float) -> float: ...
```



## _compute_y_stokes

[Show source in g12_radial_distribution_ao2008.py:150](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L150)

Compute y(St), ensuring values remain non-negative.

y(St) = -0.1988 St^4 + 1.5275 St^3 - 4.2942 St^2 + 5.3406 St

Ensures y(St) ≥ 0 (if negative, sets to 0).

#### Signature

```python
def _compute_y_stokes(stokes_number: NDArray[np.float64]) -> NDArray[np.float64]: ...
```



## get_g12_radial_distribution_ao2008

[Show source in g12_radial_distribution_ao2008.py:14](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/g12_radial_distribution_ao2008.py#L14)

Compute the radial distribution function g₁₂ for particles in a
turbulent flow.

This function describes the clustering of particles in a turbulent flow.
The equation is:

- g₁₂ = ((η² + r_c²) / (R² + r_c²))^(C₁/2)
    - g₁₂ is the radial distribution function (dimensionless),
    - η is the Kolmogorov length scale (m),
    - r_c is the turbulence-driven correction length (m),
    - R is the collision radius (sum of the two particle radii) (m),
    - C₁ is a dimensionless function dependent on the Stokes numbers,
      Reynolds number, etc.

#### Arguments

- particle_radius : Array of particle radii in meters.
- stokes_number : Array of particle Stokes numbers (dimensionless).
- kolmogorov_length_scale : Kolmogorov length scale in meters.
- reynolds_lambda : Taylor-microscale Reynolds number (dimensionless).
- normalized_accel_variance : Normalized acceleration variance
    (dimensionless).
- kolmogorov_velocity : Kolmogorov velocity scale in m/s.
- kolmogorov_time : Kolmogorov timescale in seconds.

#### Returns

- The radial distribution function g₁₂ (dimensionless).

#### Examples

```py title="Example Usage"
import numpy as np
from particula.dynamics.coagulation.turbulent_dns_kernel
    .g12_radial_distribution_ao2008 import (
        get_g12_radial_distribution_ao2008,
    )

radii = np.array([1e-7, 1e-6])
stks = np.array([0.1, 0.2])
result = get_g12_radial_distribution_ao2008(
    particle_radius=radii,
    stokes_number=stks,
    kolmogorov_length_scale=1e-4,
    reynolds_lambda=100,
    normalized_accel_variance=0.5,
    kolmogorov_velocity=0.1,
    kolmogorov_time=0.001,
)
print(result)
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2.
  Theory and parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "positive",
        "stokes_number": "positive",
        "kolmogorov_length_scale": "positive",
        "reynolds_lambda": "positive",
        "normalized_accel_variance": "positive",
        "kolmogorov_velocity": "positive",
        "kolmogorov_time": "positive",
    }
)
def get_g12_radial_distribution_ao2008(
    particle_radius: NDArray[np.float64],
    stokes_number: NDArray[np.float64],
    kolmogorov_length_scale: float,
    reynolds_lambda: float,
    normalized_accel_variance: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> NDArray[np.float64]: ...
```


---
# phi_ao2008.md

# Phi Ao2008

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / Phi Ao2008

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py) module.

## PhiComputeTerms

[Show source in phi_ao2008.py:12](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L12)

Parameters for computing Φ function terms.

#### Signature

```python
class PhiComputeTerms(NamedTuple): ...
```



## _compute_phi_term1

[Show source in phi_ao2008.py:121](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L121)

Compute the first term of the Φ function.

 term_1 = {
    1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  1 / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )
}        ×  ( vₚ₁ - vₚ₂ ) / ( 2 φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) )² )

#### Signature

```python
def _compute_phi_term1(terms: PhiComputeTerms) -> NDArray[np.float64]: ...
```

#### See also

- [PhiComputeTerms](#phicomputeterms)



## _compute_phi_term2

[Show source in phi_ao2008.py:146](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L146)

Compute the second term of the Φ function.

term₂ =
{
  4 / [ (v₂ / φ)² − ( (1 / τ₂) + (1 / α) )² ]       − 1 / [ (v₂ / φ) + (1 / τ₂) + (1 / α) ]²       − 1 / [ (v₂ / φ) − (1 / τ₂) − (1 / α) ]²     }
× v₂ /       [ 2 φ ( (1 / τ₁) − (1 / α) + ( (1 / τ₂) + (1 / α) ) × (v₁ / v₂) ) ]

#### Signature

```python
def _compute_phi_term2(terms: PhiComputeTerms) -> NDArray[np.float64]: ...
```

#### See also

- [PhiComputeTerms](#phicomputeterms)



## _compute_phi_term3

[Show source in phi_ao2008.py:181](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L181)

Compute the third term of the Φ function.

term_3 =
{
    2φ / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )        -  2φ / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  vₚ₁ / ( ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )² )        +  vₚ₂ / ( ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )² )
}
    ×  1 / ( 2φ ( (vₚ₁ - vₚ₂/φ ) + (1 / τₚ₁) + (1 / τₚ₂) ) )

#### Signature

```python
def _compute_phi_term3(terms: PhiComputeTerms) -> NDArray[np.float64]: ...
```

#### See also

- [PhiComputeTerms](#phicomputeterms)



## get_phi_ao2008

[Show source in phi_ao2008.py:23](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/phi_ao2008.py#L23)

Compute the function Φ(α, φ) for the given particle properties using
Ayala et al. (2008).

This function calculates Φ(α, φ) when vₚ₁ > vₚ₂ by considering the
velocities (vₚ₁, vₚ₂) and inertia times (τₚ₁, τₚ₂). The equation is:

Φ(α, φ), for vₚ₁ > vₚ₂ =
    {  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  1 / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) ) }        ×  ( vₚ₁ - vₚ₂ ) / ( 2 φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) )² )

+ {  4 / ( (vₚ₂ / φ)² - ( (1 / τₚ₂) + (1 / α) )² )        -  1 / ( (vₚ₂ / φ) + (1 / τₚ₂) + (1 / α) )²        -  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )²  }        ×  ( vₚ₂ / ( 2 φ ( (1 / τₚ₁) - (1 / α)         + ( (1 / τₚ₂) + (1 / α) ) (vₚ₁ / vₚ₂) ) ) )

+ {  2φ / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )        -  2φ / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )        -  vₚ₁ / ( ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )² )        +  vₚ₂ / ( ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )² )  }        ×  1 / ( 2φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) ) )

- v₁ and v₂: Velocities of particles 1 and 2 in m/s.
- τ₁ and τ₂: Inertia timescales of particles 1 and 2 in s.
- α: Turbulent interaction parameter (dimensionless).
- φ: Characteristic velocity (m/s).

#### Arguments

- alpha : Turbulence/droplet interaction parameter (dimensionless).
- phi : Characteristic velocity parameter (m/s).
- particle_inertia_time : Inertia timescales τₚ₁ and τₚ₂ (s).
- particle_velocity : Velocities vₚ₁ and vₚ₂ (m/s).

#### Returns

- The computed Φ(α, φ) (dimensionless).

#### Examples

```py
import numpy as np
from particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008
    import get_phi_ao2008

alpha_val = 0.3
phi_val = 0.1
inertia_times = np.array([0.05, 0.06])
velocities = np.array([0.2, 0.18])
result = get_phi_ao2008(alpha_val, phi_val, inertia_times, velocities)
print(result)
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2.
  Theory and parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "alpha": "positive",
        "phi": "positive",
        "particle_inertia_time": "positive",
        "particle_velocity": "positive",
    }
)
def get_phi_ao2008(
    alpha: Union[float, NDArray[np.float64]],
    phi: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# psi_ao2008.md

# Psi Ao2008

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / Psi Ao2008

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.psi_ao2008](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/psi_ao2008.py) module.

## get_psi_ao2008

[Show source in psi_ao2008.py:12](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/psi_ao2008.py#L12)

Compute the function Ψ(α, φ) for the k-th droplet.

This function calculates Ψ(α, φ) for the droplet collision kernel in the
turbulent DNS model. The equation is:

- Ψ(α, φ) = 1 / ((1/τₚₖ) + (1/α) + (vₚₖ/φ))
            - (vₚₖ / (2φ ((1/τₚₖ) + (1/α) + (vₚₖ/φ))²))
    - τₚₖ is the inertia timescale of the droplet (s),
    - α is a parameter related to turbulence (dimensionless),
    - φ is a characteristic velocity/timescale parameter (m/s),
    - vₚₖ is the droplet velocity (m/s).

#### Arguments

- alpha : Parameter related to turbulence (dimensionless).
- phi : Characteristic velocity or timescale parameter (m/s).
- particle_inertia_time : Inertia timescale of the droplet τₚₖ (s).
- particle_velocity : Velocity of the droplet vₚₖ (m/s).

#### Returns

- The value of Ψ(α, φ) (dimensionless).

#### Examples

``` py
import numpy as np
import particula as par

alpha = 0.5
phi = 0.2
particle_inertia_time = 0.05
particle_velocity = 0.3

psi_value = par.dyanmics.get_psi_ao2008(
    alpha, phi, particle_inertia_time, particle_velocity
)
print(psi_value)
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2.
  Theory and parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "alpha": "positive",
        "phi": "positive",
        "particle_inertia_time": "positive",
        "particle_velocity": "positive",
    }
)
def get_psi_ao2008(
    alpha: Union[float, NDArray[np.float64]],
    phi: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# radial_velocity_module.md

# Radial Velocity Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / Radial Velocity Module

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/radial_velocity_module.py) module.

## get_radial_relative_velocity_ao2008

[Show source in radial_velocity_module.py:77](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/radial_velocity_module.py#L77)

Compute the radial relative velocity based on Ayala et al. (2008).

This function estimates the radial relative velocity between pairs of
particles considering both turbulent velocity dispersion and gravitational
acceleration. The conceptual form is:

- ⟨|wᵣ|⟩ = √(2/π) × √(σ² + (π/8) × (τₚ₁ - τₚ₂)² × g²)
    - wᵣ is the radial relative velocity in m/s,
    - σ is the turbulence velocity dispersion in m/s,
    - τₚ₁, τₚ₂ are the inertia timescales (s),
    - g is the gravitational acceleration (m/s²).

#### Arguments

- velocity_dispersion : Turbulence velocity dispersion (σ) in m/s.
- particle_inertia_time : Inertia timescale(s) of the particle(s) in seconds.

#### Returns

- The radial relative velocity ⟨|wᵣ|⟩ in m/s.

#### Examples

```py
import numpy as np
import particula as par

# Example usage (currently raises NotImplementedError)
try:
    rv = par.dynamics.get_radial_relative_velocity_ao2008(
        1.0, np.array([0.05, 0.1])
    )
except NotImplementedError as e:
    print(e)
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2. Theory
  and parameterization. New Journal of Physics, 10.

#### Signature

```python
@validate_inputs(
    {"velocity_dispersion": "positive", "particle_inertia_time": "positive"}
)
def get_radial_relative_velocity_ao2008(
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_radial_relative_velocity_dz2002

[Show source in radial_velocity_module.py:14](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/radial_velocity_module.py#L14)

Compute the radial relative velocity based on Dodin and Elperin (2002).

This function calculates the radial relative velocity between pairs of
particles under turbulent conditions, capturing the effects of different
inertia timescales. The equation is:

- ⟨|wᵣ|⟩ = √(2/π) × σ × f(b)
    - wᵣ is the radial relative velocity in m/s,
    - σ is the turbulence velocity dispersion in m/s,
    - b = (g × |τₚᵢ - τₚⱼ|) / (√2 × σ),
    - f(b) = ½√π (b + 0.5 / b) erf(b) + ½ exp(-b²).

#### Arguments

- velocity_dispersion : Turbulence velocity dispersion (σ) in m/s.
- particle_inertia_time : Inertia timescale(s) (τₚ) in seconds.

#### Returns

- The radial relative velocity ⟨|wᵣ|⟩ in m/s.

#### Examples

```py
import numpy as np
import particula as par

# Example with an array of inertia times
result = par.dynamics.get_radial_relative_velocity_dz2002(
    1.0, np.array([0.1, 0.2, 0.3])
)
print(result)
```

#### References

- Dodin, Z., & Elperin, T. (2002). Phys. Fluids, 14, 2921–2924.

#### Signature

```python
@validate_inputs(
    {"velocity_dispersion": "positive", "particle_inertia_time": "positive"}
)
def get_radial_relative_velocity_dz2002(
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# sigma_relative_velocity_ao2008.md

# Sigma Relative Velocity Ao2008

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / Sigma Relative Velocity Ao2008

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py) module.

## VelocityCorrelationTerms

[Show source in sigma_relative_velocity_ao2008.py:37](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L37)

Parameters from computing velocity correlation terms.

#### Signature

```python
class VelocityCorrelationTerms(NamedTuple): ...
```



## _compute_cross_correlation_velocity

[Show source in sigma_relative_velocity_ao2008.py:256](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L256)

Compute cross-correlation of fluctuating velocities for two droplets.

This function calculates the cross-correlation of the fluctuating
velocities of two droplets using the following equation:

Where the equation is

- ⟨v'¹ v'²⟩ = (u'² f₂(R) / (τ_p1 τ_p2)) *
    [b₁ d₁ Φ(c₁, e₁) - b₁ d₂ Φ(c₁, e₂) - b₂ d₁ Φ(c₂, e₁) + b₂ d₂ Φ(c₂, e₂)]
    - v'¹, v'² are the fluctuating velocities for droplets 1 and 2.
    - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
    - τ_p1, τ_p2 : Inertia timescales of droplets 1 and 2 [s].
    - f₂(R) : Longitudinal velocity correlation function.
    - Φ(c, e) : Function Φ computed using `get_phi_ao2008`.

#### Arguments

- fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
- collisional_radius : Distance between two colliding droplets [m].
- particle_inertia_time : Inertia timescale of droplet 1 [s].
- particle_velocity : Droplet velocity [m/s].
- taylor_microscale : Taylor microscale [m].
- eulerian_integral_length : Eulerian integral length scale [m].
- velocity_correlation_terms : Velocity correlation coefficients [-].

#### Returns

- Cross-correlation velocity [m²/s²].

#### Examples

```py
import numpy as np
ccv = _compute_cross_correlation_velocity(
    fluid_rms_velocity=0.3,
    collisional_radius=np.array([1e-4, 2e-4]),
    particle_inertia_time=np.array([1.0, 1.2]),
    particle_velocity=np.array([0.1, 0.2]),
    taylor_microscale=0.01,
    eulerian_integral_length=0.1,
    velocity_correlation_terms=VelocityCorrelationTerms(
        b1=0.1, b2=0.2, d1=0.3, d2=0.4, c1=0.5, c2=0.6, e1=0.7, e2=0.8
    )
)
# Output: array([...])
```

#### References

- Ayala, O. et al. (2008). Effects of turbulence on the geometric
  collision rate of sedimenting droplets. Part 2. Theory and
  parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "collisional_radius": "positive",
        "particle_inertia_time": "positive",
        "particle_velocity": "positive",
        "taylor_microscale": "positive",
        "eulerian_integral_length": "positive",
    }
)
def _compute_cross_correlation_velocity(
    fluid_rms_velocity: float,
    collisional_radius: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    taylor_microscale: float,
    eulerian_integral_length: float,
    velocity_correlation_terms: VelocityCorrelationTerms,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [VelocityCorrelationTerms](#velocitycorrelationterms)



## _compute_rms_fluctuation_velocity

[Show source in sigma_relative_velocity_ao2008.py:151](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L151)

Compute RMS fluctuation velocity for the k-th droplet.

This function calculates the square of the RMS fluctuation velocity for
the k-th droplet using the following equation:

Where the equation is:

- ⟨(v'ᵏ)²⟩ = (u'² / τ_pk) * [b₁ d₁ Ψ(c₁, e₁) - b₁ d₂ Ψ(c₁, e₂)
  - b₂ d₁ Ψ(c₂, e₁) + b₂ d₂ Ψ(c₂, e₂)]
    - v'ᵏ is the fluctuating velocity for droplet k.
    - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
    - τ_pk (particle_inertia_time) : Inertia timescale of the droplet k
        [s].
    - Ψ(c, e) : Function Ψ computed using `get_psi_ao2008`.

#### Arguments

- fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
- particle_inertia_time : Inertia timescale of the droplet k [s].
- particle_velocity : Droplet velocity [m/s].
- velocity_correlation_terms : Velocity correlation coefficients [-].

#### Returns

- RMS fluctuation velocity [m²/s²].

#### Examples

```py
import numpy as np
rms_fluct = _compute_rms_fluctuation_velocity(
    fluid_rms_velocity=0.3,
    particle_inertia_time=np.array([1.0, 1.2]),
    particle_velocity=np.array([0.1, 0.2]),
    velocity_correlation_terms=VelocityCorrelationTerms(
        b1=0.1, b2=0.2, d1=0.3, d2=0.4, c1=0.5, c2=0.6, e1=0.7, e2=0.8
    )
)
# Output: array([...])
```

#### References

- Ayala, O. et al. (2008). Effects of turbulence on the geometric
  collision rate of sedimenting droplets. Part 2. Theory and
  parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"fluid_rms_velocity": "positive", "particle_inertia_time": "positive"})
def _compute_rms_fluctuation_velocity(
    fluid_rms_velocity: float,
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    velocity_correlation_terms: VelocityCorrelationTerms,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [VelocityCorrelationTerms](#velocitycorrelationterms)



## get_relative_velocity_variance

[Show source in sigma_relative_velocity_ao2008.py:50](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/sigma_relative_velocity_ao2008.py#L50)

Compute the variance of particle relative-velocity fluctuations.

This function calculates the variance of particle relative-velocity
fluctuations using the following equation:

Where the equation is:

- σ² = ⟨(v'²)⟩₁ + ⟨(v'²)⟩₂ - 2⟨v'¹ v'²⟩
    - v'¹, v'² are the fluctuating velocities for droplets 1 and 2.

#### Arguments

- fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
- collisional_radius : Distance between two colliding droplets [m].
- particle_inertia_time : Inertia timescale of droplet 1 [s].
- particle_velocity : Droplet velocity [m/s].
- taylor_microscale : Taylor microscale [m].
- eulerian_integral_length : Eulerian integral length scale [m].
- lagrangian_integral_time : Lagrangian integral time scale [s].

#### Returns

- σ² : Variance of the particle relative-velocity fluctuation [m²/s²].

#### Examples

```py
import numpy as np
sigma_sq = get_relative_velocity_variance(
    fluid_rms_velocity=0.3,
    collisional_radius=np.array([1e-4, 2e-4]),
    particle_inertia_time=np.array([1.0, 1.2]),
    particle_velocity=np.array([0.1, 0.2]),
    taylor_microscale=0.01,
    eulerian_integral_length=0.1,
    lagrangian_integral_time=0.5,
    lagrangian_taylor_microscale_time=0.05
)
# Output: array([...])
```

#### References

- Ayala, O. et al. (2008). Effects of turbulence on the geometric
  collision rate of sedimenting droplets. Part 2. Theory and
  parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "collisional_radius": "positive",
        "particle_inertia_time": "positive",
        "particle_velocity": "positive",
    }
)
def get_relative_velocity_variance(
    fluid_rms_velocity: float,
    collisional_radius: NDArray[np.float64],
    particle_inertia_time: NDArray[np.float64],
    particle_velocity: NDArray[np.float64],
    taylor_microscale: float,
    eulerian_integral_length: float,
    lagrangian_integral_time: float,
    lagrangian_taylor_microscale_time: float,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# turbulent_dns_kernel_ao2008.md

# Turbulent Dns Kernel Ao2008

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / Turbulent Dns Kernel Ao2008

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/turbulent_dns_kernel_ao2008.py) module.

## get_turbulent_dns_kernel_ao2008

[Show source in turbulent_dns_kernel_ao2008.py:28](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/turbulent_dns_kernel_ao2008.py#L28)

Compute the geometric collision kernel Γ₁₂ from DNS simulations.

Where the equation is

- Γ₁₂ = 2π R² ⟨|wᵣ|⟩ g₁₂
    - Γ₁₂ is collision kernel [m³/s].
    - R is collision radius [m].
    - wᵣ is radial relative velocity [m/s].
    - g₁₂ is radial distribution function [dimensionless].

#### Arguments

- particle_radius : Particle radius [m].
- velocity_dispersion : Velocity dispersion [m/s].
- particle_inertia_time : Particle inertia time [s].
- stokes_number : Stokes number [-].
- kolmogorov_length_scale : Kolmogorov length scale [m].
- reynolds_lambda : Reynolds number [-].
- normalized_accel_variance : Normalized acceleration variance [-].
- kolmogorov_velocity : Kolmogorov velocity [m/s].
- kolmogorov_time : Kolmogorov time [s].

#### Returns

- Collision kernel Γ₁₂ [m³/s].

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "positive",
        "velocity_dispersion": "positive",
        "particle_inertia_time": "positive",
    }
)
def get_turbulent_dns_kernel_ao2008(
    particle_radius: Union[float, NDArray[np.float64]],
    velocity_dispersion: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    stokes_number: Union[float, NDArray[np.float64]],
    kolmogorov_length_scale: float,
    reynolds_lambda: float,
    normalized_accel_variance: float,
    kolmogorov_velocity: float,
    kolmogorov_time: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_turbulent_dns_kernel_ao2008_via_system_state

[Show source in turbulent_dns_kernel_ao2008.py:103](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/turbulent_dns_kernel_ao2008.py#L103)

Compute the geometric collision kernel Γ₁₂ using AO2008 for system.

This function orchestrates the calculation of the geometric collision
kernel by deriving necessary fluid, turbulence, and particle parameters
from the provided system state. The returned value (or array) represents
the collision kernel, Γ₁₂ [m³/s], which describes collision frequency
under turbulence.

#### Arguments

- particle_radius : Radius of the particles [m]. If an array is given,
    it is assumed to represent multiple particle sizes.
- particle_density : Density of the particles [kg/m³]. Must match the
    dimensionality of `particle_radius` if both are arrays.
- fluid_density : Density of the surrounding fluid [kg/m³].
- temperature : Temperature of the fluid [K].
- re_lambda : Turbulent Reynolds number based on the Taylor microscale.
- relative_velocity : Mean relative velocity between the particle and
    fluid [m/s]. Can be a single value or an array of the same
    dimensionality as `particle_radius`.
- turbulent_dissipation : Turbulent kinetic energy dissipation rate
    [m²/s³].

#### Returns

- Collision kernel Γ₁₂ [m³/s].

#### Examples

```py
kernel_via_state = get_turbulent_dns_kernel_ao2008_via_system_state(
    particle_radius=np.array([1e-6, 2e-6]),
    particle_density=1000.0,
    fluid_density=1.225,
    temperature=298.15,
    re_lambda=100.0,
    relative_velocity=0.1,
    turbulent_dissipation=0.01,
)
# Output: array([...])
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2.
  Theory and parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
def get_turbulent_dns_kernel_ao2008_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: float,
    temperature: float,
    re_lambda: float,
    relative_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: float,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# velocity_correlation_f2_ao2008.md

# Velocity Correlation F2 Ao2008

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / Velocity Correlation F2 Ao2008

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_f2_ao2008](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_f2_ao2008.py) module.

## get_f2_longitudinal_velocity_correlation

[Show source in velocity_correlation_f2_ao2008.py:16](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_f2_ao2008.py#L16)

Compute the longitudinal velocity correlation function f₂(R) from
Ayala et al. (2008).

This function describes the correlation of velocity fluctuations as a
function of collisional radius R between two colliding droplets.

Where the equation is:

- f₂(R) = 1 / (2√(1 - 2β²)) × {
    (1 + √(1 - 2β²)) exp[-2R / ((1 + √(1 - 2β²)) L_e)]
    - (1 - √(1 - 2β²)) exp[-2R / ((1 - √(1 - 2β²)) L_e)]
  }
    - f₂(R) is the longitudinal velocity correlation function [-].
    - R is the collisional radius [m].
    - β = (√2 * λ) / L_e
    - λ (taylor_microscale) : Taylor microscale [m].
    - L_e (eulerian_integral_length) : Eulerian integral length scale
      [m].

#### Arguments

- collisional_radius : Distance between two colliding droplets [m].
- taylor_microscale : Taylor microscale [m].
- eulerian_integral_length : Eulerian integral length scale [m].

#### Returns

- f₂(R) value [dimensionless].

#### Examples

```py
import numpy as np
example_f2 = get_f2_longitudinal_velocity_correlation(
    collisional_radius=np.array([1e-4, 2e-4]),
    taylor_microscale=1e-3,
    eulerian_integral_length=1e-2,
)
# Output: array([...])
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2. Theory
  and parameterization. New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "collisional_radius": "positive",
        "taylor_microscale": "positive",
        "eulerian_integral_length": "positive",
    }
)
def get_f2_longitudinal_velocity_correlation(
    collisional_radius: Union[float, NDArray[np.float64]],
    taylor_microscale: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# velocity_correlation_terms_ao2008.md

# Velocity Correlation Terms Ao2008

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Dynamics](../../index.md#dynamics) / [Coagulation](../index.md#coagulation) / [Turbulent Dns Kernel](./index.md#turbulent-dns-kernel) / Velocity Correlation Terms Ao2008

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py) module.

## compute_b1

[Show source in velocity_correlation_terms_ao2008.py:98](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L98)

Compute b₁, a dimensionless parameter in the Ayala 2008 model.

- b₁ = (1 + √(1 - 2z²)) / (2 √(1 - 2z²))
    - z is τ_T / T_L.

#### Arguments

- z : A dimensionless parameter related to turbulence [-].

#### Returns

- b₁ value [dimensionless].

#### Examples

```py
b1_val = compute_b1(0.5)
# Output: 0.866
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"z": "positive"})
def compute_b1(
    z: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_b2

[Show source in velocity_correlation_terms_ao2008.py:129](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L129)

Compute b₂, a dimensionless parameter in the Ayala 2008 model.

- b₂ = (1 - √(1 - 2z²)) / (2 √(1 - 2z²))
    - z is τ_T / T_L.

#### Arguments

- z : A dimensionless parameter related to turbulence [-].

#### Returns

- b₂ value [dimensionless].

#### Examples

```py
b2_val = compute_b2(0.5)
# Output: 0.134
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"z": "positive"})
def compute_b2(
    z: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_beta

[Show source in velocity_correlation_terms_ao2008.py:60](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L60)

Compute β, the ratio of microscale to integral length scale.

- β = (√2 × λ) / L_e
    - λ is Taylor microscale [m].
    - L_e is Eulerian integral length scale [m].

#### Arguments

- taylor_microscale : Taylor microscale [m].
- eulerian_integral_length : Eulerian integral length scale [m].

#### Returns

- β value [dimensionless].

#### Examples

```py
beta_val = compute_beta(0.001, 0.1)
# Output: 0.01414
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {"taylor_microscale": "positive", "eulerian_integral_length": "positive"}
)
def compute_beta(
    taylor_microscale: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_c1

[Show source in velocity_correlation_terms_ao2008.py:160](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L160)

Compute c₁, a dimensionless timescale factor in the Ayala 2008 model.

- c₁ = ((1 + √(1 - 2z²)) × T_L) / 2
    - z is τ_T / T_L.
    - T_L is the Lagrangian integral timescale [s].

#### Arguments

- z : A dimensionless parameter related to turbulence [-].
- lagrangian_integral_scale : Lagrangian integral timescale [s].

#### Returns

- c₁ value [dimensionless].

#### Examples

```py
c1_val = compute_c1(0.5, 1.0)
# Output: 0.933
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"z": "positive", "lagrangian_integral_scale": "positive"})
def compute_c1(
    z: Union[float, NDArray[np.float64]],
    lagrangian_integral_scale: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_c2

[Show source in velocity_correlation_terms_ao2008.py:193](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L193)

Compute c₂, a dimensionless timescale factor in the Ayala 2008 model.

- c₂ = ((1 - √(1 - 2z²)) × T_L) / 2
    - z is τ_T / T_L.
    - T_L is the Lagrangian integral timescale [s].

#### Arguments

- z : A dimensionless parameter related to turbulence [-].
- lagrangian_integral_scale : Lagrangian integral timescale [s].

#### Returns

- c₂ value [dimensionless].

#### Examples

```py
c2_val = compute_c2(0.5, 1.0)
# Output: 0.067
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"z": "positive", "lagrangian_integral_scale": "positive"})
def compute_c2(
    z: Union[float, NDArray[np.float64]],
    lagrangian_integral_scale: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_d1

[Show source in velocity_correlation_terms_ao2008.py:226](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L226)

Compute d₁, another dimensionless coefficient from Ayala 2008.

- d₁ = (1 + √(1 - 2β²)) / (2 √(1 - 2β²))
    - β is defined as β = (√2 × λ) / L_e.

#### Arguments

- beta : A dimensionless parameter related to turbulence [-].

#### Returns

- d₁ value [dimensionless].

#### Examples

```py
d1_val = compute_d1(0.5)
# Output: 0.866
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"beta": "positive"})
def compute_d1(
    beta: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_d2

[Show source in velocity_correlation_terms_ao2008.py:257](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L257)

Compute d₂, another dimensionless coefficient from Ayala 2008.

- d₂ = (1 - √(1 - 2β²)) / (2 √(1 - 2β²))
    - β is defined as β = (√2 × λ) / L_e.

#### Arguments

- beta : A dimensionless parameter related to turbulence [-].

#### Returns

- d₂ value [dimensionless].

#### Examples

```py
d2_val = compute_d2(0.5)
# Output: 0.134
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"beta": "positive"})
def compute_d2(
    beta: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_e1

[Show source in velocity_correlation_terms_ao2008.py:288](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L288)

Compute e₁, which scales the integral length by a factor in Ayala 2008.

- e₁ = ((1 + √(1 - 2β²)) × L_e) / 2
    - β is defined as β = (√2 × λ) / L_e.
    - L_e is the Eulerian integral length scale [m].

#### Arguments

- beta : A dimensionless parameter related to turbulence [-].
- eulerian_integral_length : Eulerian integral length scale [m].

#### Returns

- e₁ value [dimensionless].

#### Examples

```py
e1_val = compute_e1(0.5, 0.1)
# Output: 0.0866
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"beta": "positive", "eulerian_integral_length": "positive"})
def compute_e1(
    beta: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_e2

[Show source in velocity_correlation_terms_ao2008.py:321](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L321)

Compute e₂, which scales the integral length by a factor in Ayala 2008.

- e₂ = ((1 - √(1 - 2β²)) × L_e) / 2
    - β is defined as β = (√2 × λ) / L_e.
    - L_e is the Eulerian integral length scale [m].

#### Arguments

- beta : A dimensionless parameter related to turbulence [-].
- eulerian_integral_length : Eulerian integral length scale [m].

#### Returns

- e₂ value [dimensionless].

#### Examples

```py
e2_val = compute_e2(0.5, 0.1)
# Output: 0.0134
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"beta": "positive", "eulerian_integral_length": "positive"})
def compute_e2(
    beta: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## compute_z

[Show source in velocity_correlation_terms_ao2008.py:18](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_dns_kernel/velocity_correlation_terms_ao2008.py#L18)

Compute z, the ratio of Taylor microscale time to Lagrangian timescale.

Where the equation is
- z = τ_T / T_L
    - τ_T (lagrangian_taylor_microscale_time) is the Lagrangian Taylor
        microscale time [s].
    - T_L (lagrangian_integral_scale) is the Lagrangian integral
        timescale [s].

#### Arguments

- lagrangian_taylor_microscale_time : Lagrangian Taylor microscale
    time [s].
- lagrangian_integral_scale : Lagrangian integral timescale [s].

#### Returns

- z value [dimensionless].

#### Examples

```py
example_z = compute_z(0.5, 1.0)
# Output: 0.5
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "lagrangian_taylor_microscale_time": "positive",
        "lagrangian_integral_scale": "positive",
    }
)
def compute_z(
    lagrangian_taylor_microscale_time: Union[float, NDArray[np.float64]],
    lagrangian_integral_scale: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# turbulent_shear_kernel.md

# Turbulent Shear Kernel

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Turbulent Shear Kernel

> Auto-generated documentation for [particula.dynamics.coagulation.turbulent_shear_kernel](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_shear_kernel.py) module.

## get_turbulent_shear_kernel_st1956

[Show source in turbulent_shear_kernel.py:31](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_shear_kernel.py#L31)

Calculate the turbulent shear kernel (Equation 13A.2, Saffman & Turner,
1956).

This function implements the formula for collisions induced by turbulent
shear. The turbulent dissipation rate and kinematic viscosity determine
how rapidly eddies drive particle collisions.

Equation:
- K(D₁, D₂) = √(π × eₖ / (120 × ν)) × (D₁ + D₂)³
    - eₖ : Turbulent kinetic energy dissipation rate [m²/s³],
    - ν : Kinematic viscosity [m²/s],
    - D₁, D₂ : diameters of particles [m].

#### Arguments

- particle_radius : Array of particle radii [m].
- turbulent_dissipation : Turbulent energy dissipation rate [m²/s³].
- kinematic_viscosity : Kinematic viscosity [m²/s].

#### Returns

- Turbulent shear kernel matrix [m³/s], shape (n, n).

#### Examples

```py title="Example"
import numpy as np

r = np.array([1e-7, 2e-7])
k_matrix = get_turbulent_shear_kernel_st1956(
    particle_radius=r,
    turbulent_dissipation=1e-3,
    kinematic_viscosity=1.5e-5
)
print(k_matrix)
```

#### References

- Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
  in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.2.

#### Signature

```python
def get_turbulent_shear_kernel_st1956(
    particle_radius: NDArray[np.float64],
    turbulent_dissipation: float,
    kinematic_viscosity: float,
) -> NDArray[np.float64]: ...
```



## get_turbulent_shear_kernel_st1956_via_system_state

[Show source in turbulent_shear_kernel.py:86](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/turbulent_shear_kernel.py#L86)

Calculate the turbulent shear kernel using system state data.

This version derives the kinematic viscosity from the temperature and
fluid density, then uses get_turbulent_shear_kernel_st1956 for the
Saffman & Turner (1956) formula:

Equation:
    - K(D₁, D₂) = √(π × eₖ / (120 × ν)) × (D₁ + D₂)³
      - eₖ : Turbulent dissipation rate [m²/s³],
      - ν : Kinematic viscosity [m²/s],
      - D₁, D₂ : particle diameters [m].

#### Arguments

- particle_radius : Array of particle radii [m].
- turbulent_dissipation : Turbulent dissipation rate [m²/s³].
- temperature : Temperature [K].
- fluid_density : Fluid density [kg/m³].

#### Returns

- Turbulent shear kernel matrix [m³/s], shape (n, n).

#### Examples

```py title="Example"
import numpy as np

r = np.array([1e-7, 2e-7])
kernel_matrix = get_turbulent_shear_kernel_st1956_via_system_state(
    particle_radius=r,
    turbulent_dissipation=1e-3,
    temperature=300,
    fluid_density=1.2
)
print(kernel_matrix)
```

#### References

- Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
  in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
  physics (3rd ed.). John Wiley & Sons. Chapter 13, Equation 13A.2.

#### Signature

```python
def get_turbulent_shear_kernel_st1956_via_system_state(
    particle_radius: NDArray[np.float64],
    turbulent_dissipation: float,
    temperature: float,
    fluid_density: float,
) -> NDArray[np.float64]: ...
```


---
# condensation_strategies.md

# Condensation Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Condensation](./index.md#condensation) / Condensation Strategies

> Auto-generated documentation for [particula.dynamics.condensation.condensation_strategies](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py) module.

## CondensationIsothermal

[Show source in condensation_strategies.py:418](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L418)

Condensation strategy under isothermal conditions.

This class implements the isothermal condensation model, wherein
temperature remains constant during mass transfer. It calculates
condensation rates based on partial pressure differences, using
no latent heat terms.

#### Attributes

- Inherits attributes from the base CondensationStrategy:
  molar_mass, diffusion_coefficient, etc.

#### Methods

- mass_transfer_rate : Calculate the mass transfer rate under
  isothermal conditions.
- rate : Get the per-particle condensation rate, accounting for
  concentration.
- step : Advance the condensation state over a given time step.

#### Examples

```py title="Example Usage"
iso_cond = CondensationIsothermal(molar_mass=0.018)
rate_array = iso_cond.rate(particle, gas_species, 298.15, 101325)
# rate_array now contains the condensation rate per particle
```

#### References

- Aerosol Modeling, Chapter 2, Equation 2.40
- Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
    (D. Topping & M. Bane, Eds.). Wiley.
    [DOI](https://doi.org/10.1002/9781119625728)
- Seinfeld & Pandis, "Atmospheric Chemistry and Physics," 3rd Ed.,
  Wiley, 2016.

#### Signature

```python
class CondensationIsothermal(CondensationStrategy):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ): ...
```

#### See also

- [CondensationStrategy](#condensationstrategy)

### CondensationIsothermal().mass_transfer_rate

[Show source in condensation_strategies.py:468](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L468)

#### Signature

```python
def mass_transfer_rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationIsothermal().rate

[Show source in condensation_strategies.py:495](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L495)

#### Signature

```python
def rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationIsothermal().step

[Show source in condensation_strategies.py:524](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L524)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    time_step: float,
) -> Tuple[ParticleRepresentation, GasSpecies]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## CondensationStrategy

[Show source in condensation_strategies.py:63](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L63)

Abstract base class for condensation strategies.

This class defines the interface for various condensation models
used in atmospheric physics. Subclasses should implement specific
condensation algorithms based on different physical models and equations.

#### Attributes

- molar_mass : The molar mass of the species [kg/mol].
- diffusion_coefficient : The diffusion coefficient [m^2/s].
- accommodation_coefficient : The mass accommodation coefficient
  (unitless).
- update_gases : Whether to update gas concentrations after
  condensation.

#### Methods

- mean_free_path : Calculate the mean free path of the gas molecules.
- knudsen_number : Compute the Knudsen number for a given particle radius.
- first_order_mass_transport : Calculate first-order mass transport
    coefficient.
- calculate_pressure_delta : Compute the partial pressure difference.
- mass_transfer_rate : Abstract method for the mass transfer rate [kg/s].
- rate : Abstract method for condensation rate per particle/bin.
- step : Abstract method to perform one timestep of condensation.

#### Examples

```py title="Example Usage of CondensationStrategy"
import particula as par
strategy = par.dynamics.ConcreteCondensationStrategy(...)
# Use strategy.mass_transfer_rate(...) to get the transfer rate
```

#### References

- Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and
  - `Physics` - From Air Pollution to Climate Change (3rd ed.). Wiley.

#### Signature

```python
class CondensationStrategy(ABC):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ): ...
```

### CondensationStrategy()._fill_zero_radius

[Show source in condensation_strategies.py:234](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234)

Fill zero radius values with the maximum radius. The concentration
value of zero will ensure that the rate of condensation is zero. The
fill is necessary to avoid division by zero in the array operations.

#### Arguments

- radius : The radius of the particles.

#### Returns

- radius : The radius of the particles with zero values filled.

#### Raises

- Warning : If all radius values are zero.

#### Signature

```python
def _fill_zero_radius(self, radius: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

### CondensationStrategy().calculate_pressure_delta

[Show source in condensation_strategies.py:260](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L260)

Calculate the difference in partial pressure between the gas and
particle phases.

#### Arguments

- particle : The particle for which the partial pressure difference
    is to be calculated.
- gas_species : The gas species with which the particle is in
    contact.
- temperature : The temperature at which the partial pressure
    difference is to be calculated.
- radius : The radius of the particles.

#### Returns

- partial_pressure_delta : The difference in partial pressure
    between the gas and particle phases.

#### Signature

```python
def calculate_pressure_delta(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    radius: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationStrategy().first_order_mass_transport

[Show source in condensation_strategies.py:189](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L189)

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

#### Arguments

- radius : The radius of the particle [m].
- temperature : The temperature at which the first-order mass
    transport coefficient is to be calculated.
- pressure : The pressure of the gas phase.
- dynamic_viscosity : The dynamic viscosity of the gas [Pa*s]. If
    not provided, it will be calculated based on the temperature

#### Returns

The first-order mass transport coefficient per particle (m^3/s).

#### References

- Chapter 2, Equation 2.49 (excluding particle number)
- Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
    (D. Topping & M. Bane, Eds.). Wiley.
    [DOI](https://doi.org/10.1002/9781119625728)

#### Signature

```python
def first_order_mass_transport(
    self,
    particle_radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().knudsen_number

[Show source in condensation_strategies.py:154](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L154)

The Knudsen number for a particle.

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

#### Arguments

- radius : The radius of the particle [m].
- temperature : The temperature of the gas [K].
- pressure : The pressure of the gas [Pa].
- dynamic_viscosity : The dynamic viscosity of the gas [Pa*s]. If
    not provided, it will be calculated based on the temperature

#### Returns

The Knudsen number, which is the ratio of the mean free path to
    the particle radius.

#### References

- [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

#### Signature

```python
def knudsen_number(
    self,
    radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().mass_transfer_rate

[Show source in condensation_strategies.py:306](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L306)

Compute the isothermal mass transfer rate for a particle.

Implements dm/dt = 4π × r × Dᵢ × Mᵢ × f(Kn, α) × Δpᵢ / (R × T),
where:
- r is the particle radius,
- Dᵢ is diffusion coefficient,
- Mᵢ is molar mass,
- f(Kn, α) is the transition correction factor,
- Δpᵢ is the difference in partial pressure,
- R is the gas constant,
- T is temperature in Kelvin.

#### Arguments

- particle : The particle representation, providing radius,
  concentration, etc.
- gas_species : The gas species condensing onto the particles.
- temperature : System temperature [K].
- pressure : System pressure [Pa].
- dynamic_viscosity : Optional dynamic viscosity [Pa*s].

#### Returns

- Mass transfer rate [kg/s] for each particle.

#### Examples

```py title="Example Usage of mass_transfer_rate"
m_rate = iso_cond.mass_transfer_rate(
    particle, gas_species, 298.15, 101325
)
```

#### Signature

```python
@abstractmethod
def mass_transfer_rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationStrategy().mean_free_path

[Show source in condensation_strategies.py:124](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L124)

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

#### Arguments

- temperature : The temperature of the gas [K].
- pressure : The pressure of the gas [Pa].
- dynamic_viscosity : The dynamic viscosity of the gas [Pa*s]. If
    not provided, it will be calculated based on the temperature

#### Returns

The mean free path of the gas molecules in meters (m).

#### References

- Mean Free Path
    [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

#### Signature

```python
def mean_free_path(
    self, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().rate

[Show source in condensation_strategies.py:348](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L348)

Compute the net condensation rate per particle, scaled by
concentration.

Calculates the mass transfer rate and multiplies it by particle
concentration, yielding the total mass condensation rate per particle.

#### Arguments

- particle : ParticleRepresentation object with distribution and
  concentration.
- gas_species : GasSpecies object for the condensing gas.
- temperature : The absolute temperature in Kelvin.
- pressure : The pressure in Pascals.

#### Returns

- Condensation rate per particle or bin, in kg/s.

#### Examples

```py title="Example Usage of rate"
rates = iso_cond.rate(particle, gas_species, 298.15, 101325)
# returns array([...]) with condensation rates
```

#### Signature

```python
@abstractmethod
def rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationStrategy().step

[Show source in condensation_strategies.py:381](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L381)

Perform one timestep of isothermal condensation on the particle.

Calculates the mass transfer for the specified time_step and updates
both the particle mass and the gas concentration
(if update_gases=True).

#### Arguments

- particle : The particle representation to update.
- gas_species : The gas species whose concentration is reduced.
- temperature : System temperature [K].
- pressure : System pressure [Pa].
- time_step : The time interval for condensation [s].

#### Returns

- Updated ParticleRepresentation.
- Updated GasSpecies.

#### Examples

```py
updated_particle, updated_gas = iso_cond.step(
    particle, gas_species, 298.15, 101325, 1.0
)
```

#### Signature

```python
@abstractmethod
def step(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    time_step: float,
) -> Tuple[ParticleRepresentation, GasSpecies]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)


---
# mass_transfer.md

# Mass Transfer

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Condensation](./index.md#condensation) / Mass Transfer

> Auto-generated documentation for [particula.dynamics.condensation.mass_transfer](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py) module.

## get_first_order_mass_transport_k

[Show source in mass_transfer.py:35](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L35)

Calculate the first-order mass transport coefficient per particle.

This function computes the coefficient K that governs how fast mass is
transported to or from a particle in a vapor. The equation is:

- K = 4π × radius × D × X
    - K : Mass transport coefficient [m³/s].
    - radius : Particle radius [m].
    - D : Diffusion coefficient of the vapor [m²/s].
    - X : Vapor transition correction factor [unitless].

#### Arguments

- particle_radius : The radius of the particle [m].
- vapor_transition : The vapor transition correction factor [unitless].
- diffusion_coefficient : The diffusion coefficient of the vapor [m²/s].
  Defaults to 2e-5 (approx. air).

#### Returns

- The first-order mass transport coefficient per particle [m³/s].

#### Examples

```py title="Float input"
import particula as par
par.dynamics.get_first_order_mass_transport_k(
    particle_radius=1e-6,
    vapor_transition=0.6,
    diffusion_coefficient=2e-9
)
# Output: 1.5079644737231005e-14
```

```py title="Array input"
import particula as par
par.dynamics.get_first_order_mass_transport_k(
    particle_radius=np.array([1e-6, 2e-6]),
    vapor_transition=np.array([0.6, 0.6]),
    diffusion_coefficient=2e-9
)
# Output: array([1.50796447e-14, 6.03185789e-14])
```

#### References

- Aerosol Modeling: Chapter 2, Equation 2.49
- Wikipedia contributors, "Mass diffusivity,"
  https://en.wikipedia.org/wiki/Mass_diffusivity

#### Signature

```python
@validate_inputs({"particle_radius": "nonnegative"})
def get_first_order_mass_transport_k(
    particle_radius: Union[float, NDArray[np.float64]],
    vapor_transition: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_mass_transfer

[Show source in mass_transfer.py:236](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L236)

Route mass transfer calculation to single or multiple-species routines.

Depending on whether gas_mass represents one or multiple species, this
function calls either calculate_mass_transfer_single_species or
calculate_mass_transfer_multiple_species. The primary calculation
involves:

- mass_to_change = mass_rate × time_step × particle_concentration

#### Arguments

- mass_rate : The rate of mass transfer per particle [kg/s].
- time_step : The time step for the mass transfer calculation [s].
- gas_mass : The available mass of gas species [kg].
- particle_mass : The mass of each particle [kg].
- particle_concentration : The concentration of particles [#/m³].

#### Returns

- The mass transferred (array with the same shape as particle_mass).

#### Examples

```py title="Single species input"
import particula as par
par.dynamics.get_mass_transfer(
    mass_rate=np.array([0.1, 0.5]),
    time_step=10,
    gas_mass=np.array([0.5]),
    particle_mass=np.array([1.0, 50]),
    particle_concentration=np.array([1, 0.5])
)
```

```py title="Multiple species input"
import particula as par
par.dynamics.get_mass_transfer(
    mass_rate=np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]]),
    time_step=10,
    gas_mass=np.array([1.0, 0.8, 0.5]),
    particle_mass=np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]]),
    particle_concentration=np.array([5, 4])
)
```

#### Signature

```python
@validate_inputs(
    {
        "mass_rate": "finite",
        "time_step": "positive",
        "gas_mass": "nonnegative",
        "particle_mass": "nonnegative",
        "particle_concentration": "nonnegative",
    }
)
def get_mass_transfer(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_mass_transfer_of_multiple_species

[Show source in mass_transfer.py:388](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L388)

Calculate mass transfer for multiple gas species.

Here, gas_mass has multiple elements (each species). For each species,
this function calculates mass_to_change for all particle bins:

- mass_to_change = mass_rate × time_step × particle_concentration

Then it limits or scales that mass based on available gas mass and
particle mass in each species bin.

#### Arguments

- mass_rate : The mass transfer rate per particle for each gas
    species [kg/s].
- time_step : The time step [s].
- gas_mass : The available mass of each gas species [kg].
- particle_mass : The mass of each particle for each gas species [kg].
- particle_concentration : The concentration of particles [#/m³].

#### Returns

- The mass transferred for multiple gas species, matching the shape
  of (particle_mass).

#### Examples

```py title="Multiple species input"
import particula as par
par.dynamics.get_mass_transfer_of_multiple_species(
    mass_rate=np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]]),
    time_step=10,
    gas_mass=np.array([1.0, 0.8, 0.5]),
    particle_mass=np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]]),
    particle_concentration=np.array([5, 4])
)
# Output: array([...])
```

#### Signature

```python
@validate_inputs(
    {
        "mass_rate": "finite",
        "time_step": "positive",
        "gas_mass": "nonnegative",
        "particle_mass": "nonnegative",
        "particle_concentration": "nonnegative",
    }
)
def get_mass_transfer_of_multiple_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_mass_transfer_of_single_species

[Show source in mass_transfer.py:313](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L313)

Calculate mass transfer for a single gas species.

This function assumes gas_mass has a size of 1 (single species).
It first computes the total mass_to_change per particle:

- mass_to_change = mass_rate × time_step × particle_concentration

Then it scales or limits that mass based on available gas mass and
particle mass.

#### Arguments

- mass_rate : Mass transfer rate per particle [kg/s].
- time_step : The time step [s].
- gas_mass : Total available mass of the gas species [kg].
- particle_mass : The mass of each particle [kg].
- particle_concentration : Particle concentration [#/m³].

#### Returns

- The amount of mass transferred for the single gas species, shaped
  like particle_mass.

#### Examples

```py title="Single species input"
import particula as par
par.dynamics.get_mass_transfer_of_single_species(
    mass_rate=np.array([0.1, 0.5]),
    time_step=10,
    gas_mass=np.array([0.5]),
    particle_mass=np.array([1.0, 50]),
    particle_concentration=np.array([1, 0.5])
)
# Output: array([...])
```

#### Signature

```python
@validate_inputs(
    {
        "mass_rate": "finite",
        "time_step": "positive",
        "gas_mass": "nonnegative",
        "particle_mass": "nonnegative",
        "particle_concentration": "nonnegative",
    }
)
def get_mass_transfer_of_single_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_mass_transfer_rate

[Show source in mass_transfer.py:103](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L103)

Calculate the mass transfer rate for a particle.

This function calculates the mass transfer rate dm/dt, leveraging the
difference in partial pressure (pressure_delta) and the first-order
mass transport coefficient (K). The equation is:

- dm/dt = (K × Δp × M) / (R × T)
    - dm/dt : Mass transfer rate [kg/s].
    - K : First-order mass transport coefficient [m³/s].
    - Δp : Partial pressure difference [Pa].
    - M : Molar mass [kg/mol].
    - R : Universal gas constant [J/(mol·K)].
    - T : Temperature [K].

#### Arguments

- pressure_delta : The difference in partial pressure [Pa].
- first_order_mass_transport : The mass transport coefficient [m³/s].
- temperature : The temperature [K].
- molar_mass : The molar mass [kg/mol].

#### Returns

- The mass transfer rate [kg/s].

#### Examples

```py title="Single value input"
import particula as par
par.dynamics.mass_transfer_rate(
    pressure_delta=10.0,
    first_order_mass_transport=1e-17,
    temperature=300.0,
    molar_mass=0.02897
)
# Output: 1.16143004e-21
```

```py title="Array input"
import particula as par
par.dynamics.mass_transfer_rate(
    pressure_delta=np.array([10.0, 15.0]),
    first_order_mass_transport=np.array([1e-17, 2e-17]),
    temperature=300.0,
    molar_mass=0.02897
)
# Output: array([1.16143004e-21, 3.48429013e-21])
```

#### References

- Aerosol Modeling: Chapter 2, Equation 2.41
- Seinfeld and Pandis, "Atmospheric Chemistry and Physics,"
    Equation 13.3

#### Signature

```python
@validate_inputs(
    {
        "pressure_delta": "finite",
        "first_order_mass_transport": "finite",
        "temperature": "positive",
        "molar_mass": "positive",
    }
)
def get_mass_transfer_rate(
    pressure_delta: Union[float, NDArray[np.float64]],
    first_order_mass_transport: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_radius_transfer_rate

[Show source in mass_transfer.py:178](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L178)

Convert mass rate to radius growth/evaporation rate.

This function converts the mass transfer rate (dm/dt) into a radius
change rate (dr/dt). The equation is:

- dr/dt = (1 / 4πr²ρ) × dm/dt
    - dr/dt : Radius change rate [m/s].
    - r : Particle radius [m].
    - ρ : Particle density [kg/m³].
    - dm/dt : Mass change rate [kg/s].

#### Arguments

- mass_rate : The mass transfer rate [kg/s].
- particle_radius : The radius of the particle [m].
- density : The density of the particle [kg/m³].

#### Returns

- The radius growth (or evaporation) rate [m/s].

#### Examples

```py title="Single value input"
import particula as par
par.dynamics.radius_transfer_rate(
    mass_rate=1e-21,
    particle_radius=1e-6,
    density=1000
)
# Output: 7.95774715e-14
```

```py title="Array input"
import particula as par
par.dynamics.radius_transfer_rate(
    mass_rate=np.array([1e-21, 2e-21]),
    particle_radius=np.array([1e-6, 2e-6]),
    density=1000
)
# Output: array([7.95774715e-14, 1.98943679e-14])
```

#### Signature

```python
@validate_inputs(
    {"mass_rate": "finite", "particle_radius": "nonnegative", "density": "positive"}
)
def get_radius_transfer_rate(
    mass_rate: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# dilution.md

# Dilution

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Dynamics](./index.md#dynamics) / Dilution

> Auto-generated documentation for [particula.dynamics.dilution](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py) module.

## get_dilution_rate

[Show source in dilution.py:55](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L55)

Calculate the dilution rate of a substance in a system.

The dilution rate describes how quickly the concentration of a
substance decreases due to the volume dilution coefficient and
the current concentration. The calculation is:

- R = -(α × c)
    - R is the dilution rate [s⁻¹],
    - α is the volume dilution coefficient [s⁻¹],
    - c is the current concentration [#/m³].

#### Arguments

- coefficient : The volume dilution coefficient in inverse
    seconds (s⁻¹).
- concentration : The concentration of the substance in #/m³
    (or relevant units).

#### Returns

- The dilution rate in s⁻¹, returned as a negative value
  to indicate a decrease in concentration.

#### Examples

``` py title="Example (float input)"
get_dilution_rate(coefficient=0.01, concentration=100)
# Returns -1.0
```

``` py title="Example (array input)"
get_dilution_rate(
    coefficient=0.01,
    concentration=np.array([100, 200, 300]),
)
# Returns array([-1., -2., -3.])
```

#### References

- H. Fogler, "Elements of Chemical Reaction Engineering,"
  5th ed., Prentice Hall, 2016. [check]

#### Signature

```python
def get_dilution_rate(
    coefficient: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_volume_dilution_coefficient

[Show source in dilution.py:10](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L10)

Calculate the volume dilution coefficient.

This coefficient represents how quickly a substance is diluted within
a system of a given volume when a known input flow rate is supplied.
The equation is:

- α = Q / V
    - α is the volume dilution coefficient [s⁻¹],
    - Q is the input flow rate [m³/s],
    - V is the system volume [m³].

#### Arguments

- volume : The volume of the system in cubic meters (m³).
- input_flow_rate : The flow rate entering the system in
    cubic meters per second (m³/s).

#### Returns

- The volume dilution coefficient in inverse seconds (s⁻¹).

#### Examples

``` py title="Example (float input)"
get_volume_dilution_coefficient(volume=10, input_flow_rate=0.1)
# Returns 0.01
```

``` py title="Example (array input)"
get_volume_dilution_coefficient(
    volume=np.array([10, 20, 30]),
    input_flow_rate=np.array([0.1, 0.2, 0.3]),
)
# Returns array([0.01, 0.01, 0.01])
```

#### References

- O. Levenspiel, "Chemical Reaction Engineering," 3rd ed., Wiley, 1999.
[check]

#### Signature

```python
def get_volume_dilution_coefficient(
    volume: Union[float, NDArray[np.float64]],
    input_flow_rate: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# particle_process.md

# Particle Process

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Dynamics](./index.md#dynamics) / Particle Process

> Auto-generated documentation for [particula.dynamics.particle_process](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py) module.

## Coagulation

[Show source in particle_process.py:140](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L140)

Implements a coagulation process for aerosol particles.

This class applies a specified coagulation strategy to each particle
in an Aerosol, merging or aggregating particles as needed, based on
the chosen physical model.

#### Attributes

- coagulation_strategy : The coagulation strategy used for particle
  collision calculations.

#### Methods

- execute : Perform the coagulation step over a given time interval.
- rate : Calculate the coagulation rate for each particle.

#### Examples

```py title="Example Usage"
import particula as par
coagulation = par.dynamics.Coagulation(
    coagulation_strategy=my_strategy
)
updated_aerosol = coagulation.execute(aerosol, time_step=0.5)
# updated_aerosol now reflects coalesced or aggregated particles
```

#### References

- [Aerosol Wikipedia](https://en.wikipedia.org/wiki/Aerosol)
- Seinfeld, J. H. and Pandis, S. N., "Atmospheric Chemistry and
  - `Physics` - From Air Pollution to Climate Change," Wiley, 2016.

#### Signature

```python
class Coagulation(Runnable):
    def __init__(self, coagulation_strategy: CoagulationStrategyABC): ...
```

#### See also

- [CoagulationStrategyABC](coagulation/coagulation_strategy/coagulation_strategy_abc.md#coagulationstrategyabc)
- [Runnable](../runnable.md#runnable)

### Coagulation().execute

[Show source in particle_process.py:182](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L182)

Perform the coagulation process over a given time step.

#### Arguments

- aerosol : The Aerosol instance to modify.
- time_step : The total time interval for coagulation.
- sub_steps : Number of internal subdivisions for iterative
  calculation.

#### Returns

- Aerosol : The updated aerosol object after the coagulation step.

#### Examples

```py title="Example Coagulation Execution"
updated_aerosol = coagulation.execute(
    aerosol, time_step=0.5, sub_steps=2
)
# The aerosol now reflects changes from particle collisions
```

#### Signature

```python
def execute(self, aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)

### Coagulation().rate

[Show source in particle_process.py:216](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L216)

Compute the coagulation rate for each particle in the aerosol.

#### Arguments

- aerosol : The Aerosol instance containing particles.

#### Returns

- np.ndarray : An array of coagulation rates for each particle,
  in units related to particle collisions per unit time.

#### Examples

```py title="Coagulation Rate Calculation Example"
rates = coagulation.rate(aerosol)
# rates might look like array([0.1, 0.05, ...])
```

#### Signature

```python
def rate(self, aerosol: Aerosol) -> Any: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)



## MassCondensation

[Show source in particle_process.py:19](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L19)

Handles the mass condensation process for aerosols.

This class applies a specified condensation strategy to each particle
in an Aerosol, updating particle mass and reducing gas concentration
accordingly. It is designed to work with any CondensationStrategy
subclass.

#### Attributes

- condensation_strategy : The condensation strategy used for mass
  transfer calculations.

#### Methods

- execute : Perform the mass condensation over a specified time step.
- rate : Calculate the mass condensation rate for each particle.

#### Examples

```py title="Example Mass Condensation"
import particula as par
condensation = par.dyanmics.MassCondensation(
    condensation_strategy=my_strategy
)
updated_aerosol = condensation.execute(aerosol, time_step=1.0)
# updated_aerosol now reflects condensed mass
```

#### References

- [Aerosol Wikipedia](https://en.wikipedia.org/wiki/Aerosol)
- Seinfeld, J. H. and Pandis, S. N., "Atmospheric Chemistry and Physics:
  From Air Pollution to Climate Change," Wiley, 2016.

#### Signature

```python
class MassCondensation(Runnable):
    def __init__(self, condensation_strategy: CondensationStrategy): ...
```

#### See also

- [CondensationStrategy](condensation/condensation_strategies.md#condensationstrategy)
- [Runnable](../runnable.md#runnable)

### MassCondensation().execute

[Show source in particle_process.py:65](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L65)

Perform the mass condensation process over a given time step.

#### Arguments

- aerosol : The Aerosol instance to modify.
- time_step : The total time interval for condensation.
- sub_steps : Number of subdivisions for iterative calculation.

#### Returns

- The updated aerosol object after condensation.

#### Examples

```py title="Example Condensation Execution"
updated_aerosol = condensation.execute(
    aerosol, time_step=1.0, sub_steps=2
)
# The aerosol now has reduced/increased particle/gas mass
```

#### Signature

```python
def execute(self, aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)

### MassCondensation().rate

[Show source in particle_process.py:105](https://github.com/uncscode/particula/blob/main/particula/dynamics/particle_process.py#L105)

Compute mass condensation rates for each particle.

#### Arguments

- aerosol : The Aerosol instance containing particles and gases.

#### Returns

- An array of condensation rates for each particle,
  in units of mass per unit time.

#### Examples

```py title="Rate Calculation Example"
rates = condensation.rate(aerosol)
# rates may look like array([1.2e-12, 4.5e-12, ...])
```

#### Signature

```python
def rate(self, aerosol: Aerosol) -> Any: ...
```

#### See also

- [Aerosol](../aerosol.md#aerosol)


---
# wall_loss_coefficient.md

# Wall Loss Coefficient

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Dynamics](../index.md#dynamics) / [Properties](./index.md#properties) / Wall Loss Coefficient

> Auto-generated documentation for [particula.dynamics.properties.wall_loss_coefficient](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py) module.

## get_rectangle_wall_loss_coefficient

[Show source in wall_loss_coefficient.py:101](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L101)

Calculate the particle wall loss coefficient in a rectangular chamber.

This function computes the wall loss coefficient (β₀) for a rectangular
chamber of length (L), width (W), and height (H). It uses the wall eddy
diffusivity, particle diffusion coefficient, particle settling velocity,
and chamber dimensions:

- β₀ ~ (some function of wall_eddy_diffusivity, diffusion_coefficient,
settling_velocity, and L×W×H)

#### Arguments

- wall_eddy_diffusivity : Wall eddy diffusivity in s⁻¹.
- diffusion_coefficient : Particle diffusion coefficient in m²/s.
- settling_velocity : Particle settling velocity in m/s.
- chamber_dimensions : A tuple (length, width, height) in m.

#### Returns

- The wall loss coefficient β₀ (float or NDArray[np.float64]), in s⁻¹.

#### Examples

```py title="Example (float inputs)"
from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient
)

beta_0 = get_rectangle_wall_loss_coefficient(
    wall_eddy_diffusivity=1e-3,
    diffusion_coefficient=1e-5,
    settling_velocity=2e-4,
    chamber_dimensions=(1.0, 0.5, 0.5)
)
print(beta_0)
# Example output: 0.0009
```

#### References

- Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
  GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
  SHAPE. J Aerosol Sci, 12(5).
  https://doi.org/10.1016/0021-8502(81)90036-7

#### Signature

```python
def get_rectangle_wall_loss_coefficient(
    wall_eddy_diffusivity: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]],
    settling_velocity: Union[float, NDArray[np.float64]],
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_rectangle_wall_loss_coefficient_via_system_state

[Show source in wall_loss_coefficient.py:251](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L251)

Calculate the wall loss coefficient for a rectangular chamber based on
the system state.

This function computes the wall loss coefficient for a rectangular chamber
using the system's physical state, including the wall eddy diffusivity,
particle properties (radius, density), and environmental conditions
(temperature, pressure). The chamber dimensions (length, width, height)
are also considered.

#### Arguments

- `wall_eddy_diffusivity` - The rate of wall eddy diffusivity in inverse
    seconds (s⁻¹).
- `particle_radius` - The radius of the particle in meters (m).
- `particle_density` - The density of the particle in kilograms per cubic
    meter (kg/m³).
- `temperature` - The temperature of the system in Kelvin (K).
- `pressure` - The pressure of the system in Pascals (Pa).
- `chamber_dimensions` - A tuple containing the length, width, and height
    of the rectangular chamber in meters (m).

#### Returns

The calculated wall loss coefficient for the rectangular chamber.

#### References

- Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
    GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
    SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
    https://doi.org/10.1016/0021-8502(81)90036-7

#### Signature

```python
def get_rectangle_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_spherical_wall_loss_coefficient

[Show source in wall_loss_coefficient.py:35](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L35)

Calculate the particle wall loss coefficient in a spherical chamber.

This function computes the wall loss coefficient based on a spherical
chamber approximation. It uses the wall eddy diffusivity, particle
diffusion coefficient, particle settling velocity, and chamber radius.
The calculation is:

- k = 6 × √(Dₑ × D) / (π × R) × f + vₛ × (3 / (4 × R))
    - k is the wall loss coefficient [s⁻¹],
    - Dₑ is the wall eddy diffusivity [m²/s or effective rate],
    - D is the particle diffusion coefficient [m²/s],
    - f is the Debye function evaluation (unitless),
    - vₛ is the settling velocity [m/s],
    - R is the chamber radius [m].

#### Arguments

- wall_eddy_diffusivity : The wall eddy diffusivity (or rate) in s⁻¹.
- diffusion_coefficient : The diffusion coefficient of the particle
    in m²/s.
- settling_velocity : The particle settling velocity in m/s.
- chamber_radius : The spherical chamber radius in m.

#### Returns

- The wall loss coefficient k, in inverse seconds
   (float or NDArray[np.float64]).

#### Examples

```py title="Example (float inputs)"
from particula.dynamics.properties.wall_loss_coefficient import (
    get_spherical_wall_loss_coefficient
)

k_value = get_spherical_wall_loss_coefficient(
    wall_eddy_diffusivity=1e-2,
    diffusion_coefficient=5e-6,
    settling_velocity=1e-4,
    chamber_radius=0.5
)
print(k_value)
# Example output: 0.0012
```

#### References

- Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
  loss rates in vessels. Aerosol Science and Technology, 2(3), 303-309.
  https://doi.org/10.1080/02786828308958636

#### Signature

```python
def get_spherical_wall_loss_coefficient(
    wall_eddy_diffusivity: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]],
    settling_velocity: Union[float, NDArray[np.float64]],
    chamber_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_spherical_wall_loss_coefficient_via_system_state

[Show source in wall_loss_coefficient.py:169](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L169)

Calculate the spherical chamber wall loss coefficient via system state.

This version uses the system's physical conditions (particle radius, density,
temperature, pressure) to compute the needed diffusion and settling velocity
before calculating the spherical wall loss coefficient:

- k = f(
    wall_eddy_diffusivity,
    diffusion_coefficient_via_system_state,
    settling_velocity_via_system_state,
    chamber_radius
)

#### Arguments

- wall_eddy_diffusivity : Wall eddy diffusivity in s⁻¹.
- particle_radius : Particle radius in m.
- particle_density : Particle density in kg/m³.
- temperature : System temperature in K.
- pressure : System pressure in Pa.
- chamber_radius : Chamber radius in m.

#### Returns

- The computed wall loss coefficient k (float or NDArray[np.float64])
    in s⁻¹.

#### Examples

```py title="Example"
from particula.dynamics.properties.wall_loss_coefficient import (
    get_spherical_wall_loss_coefficient_via_system_state
)

k_value = get_spherical_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity=1e-2,
    particle_radius=1e-7,
    particle_density=1000,
    temperature=298,
    pressure=101325,
    chamber_radius=0.5
)
print(k_value)
# Example output: 0.0018
```

#### References

- Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
  loss rates in vessels. Aerosol Science and Technology, 2(3), 303-309.
  https://doi.org/10.1080/02786828308958636

#### Signature

```python
def get_spherical_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# wall_loss.md

# Wall Loss

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Dynamics](./index.md#dynamics) / Wall Loss

> Auto-generated documentation for [particula.dynamics.wall_loss](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss.py) module.

## get_rectangle_wall_loss_rate

[Show source in wall_loss.py:87](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss.py#L87)

Calculate the wall loss rate of particles in a rectangular chamber.

This function calculates the rate of particle deposition onto the walls
of a rectangular chamber, given the wall eddy diffusivity, particle
properties (radius, density, concentration), and environmental conditions
(temperature, pressure). The final loss rate is computed via:

- L = -(k × c)
    - L is the wall loss rate [#/m³·s],
    - k is the wall loss coefficient [1/s],
    - c is the particle concentration [#/m³].

#### Arguments

- wall_eddy_diffusivity : Wall eddy diffusivity in s⁻¹.
- particle_radius : Particle radius in m.
- particle_density : Particle density in kg/m³.
- particle_concentration : Particle concentration in #/m³.
- temperature : System temperature in K.
- pressure : System pressure in Pa.
- chamber_dimensions : (length, width, height) of the
    rectangular chamber in m.

#### Returns

- The wall loss rate (float or NDArray[np.float64]) in #/m³·s.

#### Examples

```py title="Example"
import particula as par
loss_rate = par.dynamics.wall_loss.get_rectangle_wall_loss_rate(
    wall_eddy_diffusivity=1e-4,
    particle_radius=5e-8,
    particle_density=1200,
    particle_concentration=2e10,
    temperature=300,
    pressure=101325,
    chamber_dimensions=(1.0, 0.5, 0.5)
)
print(loss_rate)
# Example output: -4.6e7
```

#### References

- J. Hinds, "Aerosol Technology," 2nd ed., John Wiley & Sons, 1999.
[check]

#### Signature

```python
def get_rectangle_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_spherical_wall_loss_rate

[Show source in wall_loss.py:16](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss.py#L16)

Calculate the wall loss rate of particles in a spherical chamber.

This function calculates the rate at which particles deposit onto the
walls of a spherical chamber. The calculation is based on the wall eddy
diffusivity and key particle properties (radius, density, concentration),
together with environmental conditions (temperature, pressure).
The loss rate is determined via:

- L = -(k × c)
    - L is the wall loss rate [#/m³·s],
    - k is the wall loss coefficient [1/s] from the system state,
    - c is the particle concentration [#/m³].

#### Arguments

- wall_eddy_diffusivity : Wall eddy diffusivity in s⁻¹.
- particle_radius : Particle radius in m.
- particle_density : Particle density in kg/m³.
- particle_concentration : Particle concentration in #/m³.
- temperature : System temperature in K.
- pressure : System pressure in Pa.
- chamber_radius : Radius of the spherical chamber in m.

#### Returns

- The wall loss rate (float or NDArray[np.float64]) in #/m³·s.

#### Examples

```py title="Example"
import particula as par
rate = par.dynamics.wall_loss.get_spherical_wall_loss_rate(
    wall_eddy_diffusivity=1e-3,
    particle_radius=1e-7,
    particle_density=1000,
    particle_concentration=1e11,
    temperature=298,
    pressure=101325,
    chamber_radius=0.5
)
print(rate)
# Example output: -1.2e8
```

#### References

- Wikipedia contributors, "Aerosol dynamics," Wikipedia,
  https://en.wikipedia.org/wiki/Aerosol.

#### Signature

```python
def get_spherical_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_radius: float,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# partitioning.md

# Partitioning

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Equilibria](./index.md#equilibria) / Partitioning

> Auto-generated documentation for [particula.equilibria.partitioning](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py) module.

## get_properties_for_liquid_vapor_partitioning

[Show source in partitioning.py:212](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L212)

Get properties for liquid-vapor partitioning.

#### Signature

```python
def get_properties_for_liquid_vapor_partitioning(
    water_activity_desired, molar_mass, oxygen2carbon, density
): ...
```



## liquid_vapor_obj_function

[Show source in partitioning.py:9](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L9)

Objective function for liquid-vapor partitioning.

#### Signature

```python
def liquid_vapor_obj_function(
    e_j_partition_guess,
    c_star_j_dry,
    concentration_organic_matter,
    gamma_organic_ab,
    mass_fraction_water_ab,
    q_ab,
    molar_mass,
    error_only=True,
): ...
```



## liquid_vapor_partitioning

[Show source in partitioning.py:158](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L158)

Thermodynamic equilibrium between liquid and vapor phase.
with activity coefficients,

#### Signature

```python
def liquid_vapor_partitioning(
    c_star_j_dry,
    concentration_organic_matter,
    molar_mass,
    gamma_organic_ab,
    mass_fraction_water_ab,
    q_ab,
    partition_coefficient_guess=None,
): ...
```


---
# atmosphere.md

# Atmosphere

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Atmosphere

> Auto-generated documentation for [particula.gas.atmosphere](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py) module.

## Atmosphere

[Show source in atmosphere.py:8](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py#L8)

Represents a mixture of gas species under specific conditions.

This class represents the atmospheric environment by detailing properties
such as temperature and pressure, alongside a dynamic list of gas species
present.

#### Attributes

- temperature : Temperature of the gas mixture in Kelvin.
- total_pressure : Total atmospheric pressure of the mixture in
  Pascals.
- species : List of GasSpecies objects representing the
    various species within the gas mixture.

#### Methods

- add_species : Adds a GasSpecies object to the mixture.
- remove_species : Removes a GasSpecies object from the mixture by index.

#### Signature

```python
class Atmosphere: ...
```

### Atmosphere().__getitem__

[Show source in atmosphere.py:65](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py#L65)

Retrieve a gas species by index.

#### Arguments

- index : The index of the gas species to retrieve.

#### Returns

- The gas species at the specified index.

#### Signature

```python
def __getitem__(self, index: int) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### Atmosphere().__iter__

[Show source in atmosphere.py:56](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py#L56)

Allow iteration over the species in the gas mixture.

#### Returns

- Iterator[GasSpecies] : An iterator over the gas species objects.

#### Signature

```python
def __iter__(self): ...
```

### Atmosphere().__len__

[Show source in atmosphere.py:77](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py#L77)

Return the number of species in the gas mixture.

#### Returns

- The number of gas species in the mixture.

#### Signature

```python
def __len__(self) -> int: ...
```

### Atmosphere().__str__

[Show source in atmosphere.py:86](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py#L86)

Provide a string representation of the Atmosphere object.

#### Returns

- Includes the temperature, pressure, and a list of species.

#### Signature

```python
def __str__(self) -> str: ...
```

### Atmosphere().add_species

[Show source in atmosphere.py:32](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py#L32)

Add a GasSpecies object to the mixture.

#### Arguments

- gas_species : The gas species to be added.

#### Signature

```python
def add_species(self, gas_species: GasSpecies) -> None: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### Atmosphere().remove_species

[Show source in atmosphere.py:41](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere.py#L41)

Remove a gas species from the mixture by its index.

#### Arguments

- index : Index of the gas species to remove. Must be in range.

#### Raises

- IndexError : If the provided index is out of bounds.

#### Signature

```python
def remove_species(self, index: int) -> None: ...
```


---
# atmosphere_builders.md

# Atmosphere Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Atmosphere Builders

> Auto-generated documentation for [particula.gas.atmosphere_builders](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere_builders.py) module.

## AtmosphereBuilder

[Show source in atmosphere_builders.py:18](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere_builders.py#L18)

Builder class for creating Atmosphere objects using a fluent interface.

This class provides methods to configure and build an Atmosphere object,
allowing for step-by-step setting of atmospheric properties and
species composition.

#### Attributes

- temperature : Temperature of the gas mixture in Kelvin.
- total_pressure : Total pressure of the gas mixture in Pascals.
- species : List of GasSpecies objects in the mixture (starts empty).

#### Methods

- set_temperature : Set the temperature (with optional unit handling).
- set_pressure : Set the total pressure (with optional unit handling).
- add_species : Add a GasSpecies object to the gas mixture.
- set_parameters : Set multiple parameters from a dictionary.
- build : Validate parameters and return an Atmosphere object.

#### Examples

```py title="Create an atmosphere using the builder"
import particula as par
builder = par.gas.AtmosphereBuilder()
atmosphere = (
    builder.set_temperature(300, "K")
    .set_pressure(101325, "Pa")
    .add_species(par.gas.GasSpecies(name="O2", molar_mass=0.032))
    .build()
)
```

#### Signature

```python
class AtmosphereBuilder(BuilderABC, BuilderTemperatureMixin, BuilderPressureMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderPressureMixin](../builder_mixin.md#builderpressuremixin)
- [BuilderTemperatureMixin](../builder_mixin.md#buildertemperaturemixin)

### AtmosphereBuilder().add_species

[Show source in atmosphere_builders.py:62](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere_builders.py#L62)

Add a GasSpecies object to the gas mixture.

#### Arguments

- species : The GasSpecies object to be added.

#### Returns

- AtmosphereBuilder : This builder, for method chaining.

#### Signature

```python
def add_species(self, species: GasSpecies) -> "AtmosphereBuilder": ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### AtmosphereBuilder().build

[Show source in atmosphere_builders.py:75](https://github.com/uncscode/particula/blob/main/particula/gas/atmosphere_builders.py#L75)

Validate the configuration and construct the Atmosphere object.

This method checks that all necessary conditions are met for a valid
Atmosphere instance (e.g., at least one species must be present) and
then initializes the Atmosphere.

#### Returns

- Atmosphere : The newly created Atmosphere object.

#### Raises

- ValueError : If no species have been added to the mixture.

#### Signature

```python
def build(self) -> Atmosphere: ...
```

#### See also

- [Atmosphere](./atmosphere.md#atmosphere)


---
# concentration_function.md

# Concentration Function

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Concentration Function

> Auto-generated documentation for [particula.gas.properties.concentration_function](https://github.com/uncscode/particula/blob/main/particula/gas/properties/concentration_function.py) module.

## get_concentration_from_pressure

[Show source in concentration_function.py:10](https://github.com/uncscode/particula/blob/main/particula/gas/properties/concentration_function.py#L10)

Calculate the concentration of a gas using the ideal gas law.

The concentration is determined from the partial pressure, molar mass,
and temperature using the ideal gas equation:

- C = (P × M) / (R × T)
    - C is the concentration in kg/m³,
    - P is the partial pressure in Pascals (Pa),
    - M is the molar mass in kg/mol,
    - R is the universal gas constant (J/(mol·K)),
    - T is the temperature in Kelvin.

#### Arguments

partial_pressure : Partial pressure of the gas in Pascals (Pa).
molar_mass : Molar mass of the gas in kg/mol.
temperature : Temperature in Kelvin.

#### Examples

```py title="Floating-point Example Usage"
import particula as par
par.gas.get_concentration_from_pressure(101325, 0.02897, 298.15)
# Output: 1.184587604735883
```

#### Returns

Concentration of the gas in kg/m³.

#### Signature

```python
def get_concentration_from_pressure(
    partial_pressure: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# dynamic_viscosity.md

# Dynamic Viscosity

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Dynamic Viscosity

> Auto-generated documentation for [particula.gas.properties.dynamic_viscosity](https://github.com/uncscode/particula/blob/main/particula/gas/properties/dynamic_viscosity.py) module.

## get_dynamic_viscosity

[Show source in dynamic_viscosity.py:27](https://github.com/uncscode/particula/blob/main/particula/gas/properties/dynamic_viscosity.py#L27)

Calculate the dynamic viscosity of air using Sutherland's formula.

- μ(T) = μ₀ × (T / T₀)^(3/2) × (T₀ + S) / (T + S)
    - μ(T) is the dynamic viscosity at temperature T (Pa·s).
    - μ₀ is the reference viscosity (Pa·s).
    - T is the temperature in Kelvin.
    - T₀ is the reference temperature in Kelvin.
    - S is the Sutherland constant in Kelvin.

#### Arguments

- temperature : Desired air temperature in Kelvin. Must be > 0.
- reference_viscosity : Gas viscosity at the reference temperature
    (default is STP).
- reference_temperature : Gas temperature in Kelvin for the reference
    viscosity (default is STP).

#### Returns

- Dynamic viscosity of air at the given temperature in Pa·s.

#### Examples

``` py title="Example Float Usage"
import particula as par
par.gas.get_dynamic_viscosity(300.0)
# Output (approx.): 1.846e-05
```

#### References

- Wolfram Formula Repository, "Sutherland's Formula,"
  https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula

#### Signature

```python
@validate_inputs({"temperature": "positive"})
def get_dynamic_viscosity(
    temperature: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP,
    reference_temperature: float = REF_TEMPERATURE_STP,
) -> float: ...
```

#### See also

- [REF_TEMPERATURE_STP](../../util/constants.md#ref_temperature_stp)
- [REF_VISCOSITY_AIR_STP](../../util/constants.md#ref_viscosity_air_stp)


---
# fluid_rms_velocity.md

# Fluid Rms Velocity

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Fluid Rms Velocity

> Auto-generated documentation for [particula.gas.properties.fluid_rms_velocity](https://github.com/uncscode/particula/blob/main/particula/gas/properties/fluid_rms_velocity.py) module.

## get_fluid_rms_velocity

[Show source in fluid_rms_velocity.py:13](https://github.com/uncscode/particula/blob/main/particula/gas/properties/fluid_rms_velocity.py#L13)

Calculate the fluid RMS fluctuation velocity.

The fluid root-mean-square (RMS) velocity fluctuation quantifies
turbulence intensity in a fluid flow. It is calculated as:

- u' = (R_λ^(1/2) v_K) / 15^(1/4)
    - u' is Fluid RMS fluctuation velocity [m/s]
    - R_λ (re_lambda) is Taylor-microscale Reynolds number [-]
    - v_K is Kolmogorov velocity scale, computed as v_K = ( ε)^(1/4) [m/s]
    - v (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s]
    - ε (turbulent_dissipation) is Turbulent energy dissipation rate [m²/s³]

#### Arguments

- re_lambda : Taylor-microscale Reynolds number [-]
- kinematic_viscosity : Kinematic viscosity of the fluid [m²/s]
- turbulent_dissipation : Rate of dissipation of turbulent kinetic
    energy [m²/s³]

#### Returns

- Fluid RMS fluctuation velocity [m/s]

#### Examples

``` py title="Example Usage"
velocity = get_fluid_rms_velocity(500, 1.5e-5, 0.1)
# Output (example): 0.35
```

``` py title="Example Usage with Array Input"
velocity = get_fluid_rms_velocity(
    np.array([500, 600]),
    np.array([1.5e-5, 1.7e-5]),
    np.array([0.1, 0.12])
)
# Output (example): array([0.35, 0.41])
```

#### References

- H. Tennekes and J. L. Lumley, "A First Course in Turbulence,"
  MIT Press, 1972. [check this]

#### Signature

```python
@validate_inputs(
    {
        "re_lambda": "positive",
        "kinematic_viscosity": "positive",
        "turbulent_dissipation": "positive",
    }
)
def get_fluid_rms_velocity(
    re_lambda: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# integral_scale_module.md

# Integral Scale Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Integral Scale Module

> Auto-generated documentation for [particula.gas.properties.integral_scale_module](https://github.com/uncscode/particula/blob/main/particula/gas/properties/integral_scale_module.py) module.

## get_eulerian_integral_length

[Show source in integral_scale_module.py:54](https://github.com/uncscode/particula/blob/main/particula/gas/properties/integral_scale_module.py#L54)

Calculate the Eulerian integral length scale.

The Eulerian integral length scale is a measure of the size of the largest
turbulent eddies in a fluid flow.

- L_e = 0.5 × (u'³) / ε
    - L_e is Eulerian integral length scale [m].
    - fluid_rms_velocity (u') is Fluid RMS fluctuation velocity [m/s].
    - turbulent_dissipation (ε) is Turbulent energy dissipation rate
        [m²/s³].

#### Arguments

- fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
- turbulent_dissipation : Turbulent kinetic energy dissipation rate
    [m²/s³].

#### Returns

- Eulerian integral length scale [m].

#### Examples

``` py title="Example"
import particula as par
par.gas.get_eulerian_integral_length(0.3, 1e-4)
# Output: 1350.0
```

#### References

- Hinze, J. O., "Turbulence," McGraw-Hill, 1975. [Check this reference]
- Wikipedia contributors, "Turbulence," Wikipedia.

#### Signature

```python
@validate_inputs({"fluid_rms_velocity": "positive", "turbulent_dissipation": "positive"})
def get_eulerian_integral_length(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_lagrangian_integral_time

[Show source in integral_scale_module.py:12](https://github.com/uncscode/particula/blob/main/particula/gas/properties/integral_scale_module.py#L12)

Calculate the Lagrangian integral timescale.

The Lagrangian integral timescale is a measure of the time it takes for
a fluid particle to travel a distance equal to the integral length scale.

- T_L = (u'²) / ε
    - T_L is Lagrangian integral timescale [s].
    - fluid_rms_velocity (u') is Fluid RMS fluctuation velocity [m/s].
    - turbulent_dissipation (ε) is Turbulent energy dissipation rate
        [m²/s³].

#### Arguments

- fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
- turbulent_dissipation : Turbulent kinetic energy dissipation rate
    [m²/s³].

#### Returns

- Lagrangian integral timescale [s].

#### Examples

``` py title="Example Usage"
import particula as par
par.gas.get_lagrangian_integral_time(0.3, 1e-4)
# Output: 900.0
```

#### References

- Townsend, A. A., "The Structure of Turbulent Shear Flow," 2nd ed.,
  Cambridge University Press, 1976. [Check this reference]
- Wikipedia contributors, "Turbulence," Wikipedia.

#### Signature

```python
@validate_inputs({"fluid_rms_velocity": "positive", "turbulent_dissipation": "positive"})
def get_lagrangian_integral_time(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# kinematic_viscosity.md

# Kinematic Viscosity

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Kinematic Viscosity

> Auto-generated documentation for [particula.gas.properties.kinematic_viscosity](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py) module.

## get_kinematic_viscosity

[Show source in kinematic_viscosity.py:31](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py#L31)

Calculate the kinematic viscosity of a fluid.

The function calculates ν by dividing the dynamic viscosity (μ)
by the fluid density (ρ).

- ν = μ / ρ
    - ν is Kinematic viscosity [m²/s].
    - μ is Dynamic viscosity [Pa·s].
    - ρ is Fluid density [kg/m³].

#### Arguments

- dynamic_viscosity : Dynamic viscosity of the fluid [Pa·s].
- fluid_density : Density of the fluid [kg/m³].

#### Returns

- The kinematic viscosity [m²/s].

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_kinematic_viscosity(1.8e-5, 1.2)
# Output: ~1.5e-5
```

#### References

- "Viscosity Conversion Formula," Wolfram Formula Repository.
  https://resources.wolframcloud.com/FormulaRepository/resources/Viscosity-Conversion-Formula

#### Signature

```python
@validate_inputs({"dynamic_viscosity": "positive", "fluid_density": "positive"})
def get_kinematic_viscosity(dynamic_viscosity: float, fluid_density: float) -> float: ...
```



## get_kinematic_viscosity_via_system_state

[Show source in kinematic_viscosity.py:70](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py#L70)

Calculate the kinematic viscosity of air by first computing its dynamic
viscosity.

This function uses get_dynamic_viscosity(...) and divides by the given
fluid_density to get the kinematic viscosity.

- ν = μ / ρ
    - ν is Kinematic viscosity [m²/s].
    - μ is Dynamic viscosity [Pa·s].
    - ρ is Fluid density [kg/m³].

Where:
    - ν is Kinematic viscosity [m²/s].
    - μ is Dynamic viscosity [Pa·s].
    - ρ is Fluid density [kg/m³].

#### Arguments

- temperature : Desired air temperature [K]. Must be > 0.
- fluid_density : Density of the fluid [kg/m³].
- reference_viscosity : Reference dynamic viscosity [Pa·s].
- reference_temperature : Reference temperature [K].

#### Returns

- The kinematic viscosity of air [m²/s].

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_kinematic_viscosity_via_system_state(300, 1.2)
# Output: ~1.5e-5
```

#### References

- "Sutherland's Formula," Wolfram Formula Repository,
  https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula

#### Signature

```python
@validate_inputs({"temperature": "positive", "fluid_density": "positive"})
def get_kinematic_viscosity_via_system_state(
    temperature: float,
    fluid_density: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP,
    reference_temperature: float = REF_TEMPERATURE_STP,
) -> float: ...
```

#### See also

- [REF_TEMPERATURE_STP](../../util/constants.md#ref_temperature_stp)
- [REF_VISCOSITY_AIR_STP](../../util/constants.md#ref_viscosity_air_stp)


---
# kolmogorov_module.md

# Kolmogorov Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Kolmogorov Module

> Auto-generated documentation for [particula.gas.properties.kolmogorov_module](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kolmogorov_module.py) module.

## get_kolmogorov_length

[Show source in kolmogorov_module.py:59](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kolmogorov_module.py#L59)

Calculate the Kolmogorov length scale.

The Kolmogorov length scale represents the smallest eddies in a turbulent
flow where viscosity dominates. It is defined as:

- η = (ν³ / ε)^(1/4)
    - η is the Kolmogorov length scale [m]
    - ν is the kinematic viscosity of the fluid [m^2/s]
    - ε is the rate of dissipation of turbulent kinetic energy [m^2/s^3]

Where:
    - η Kolmogorov length scale [m]
    - ν Kinematic viscosity of the fluid [m^2/s]
    - ε Rate of dissipation of turbulent kinetic energy [m^2/s^3]

#### Arguments

- kinematic_viscosity : Kinematic viscosity of the fluid [m^2/s]
- turbulent_dissipation : Rate of dissipation of turbulent kinetic
    energy [m^2/s^3]

#### Returns

- Kolmogorov length scale [m]

#### Examples

```py title="Kolmogorov length scale of a fluid"
import particula as par
par.gas.get_kolmogorov_length(1.5e-5, 0.1)
# Output: 0.0029154759474226504
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
the geometric collision rate of sedimenting droplets. Part 2. Theory
and parameterization. New Journal of Physics, 10.
https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {"kinematic_viscosity": "positive", "turbulent_dissipation": "positive"}
)
def get_kolmogorov_length(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_kolmogorov_time

[Show source in kolmogorov_module.py:17](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kolmogorov_module.py#L17)

Calculate the Kolmogorov time of a fluid.

The Kolmogorov time scale represents the smallest timescale in turbulence
where viscous forces dominate over inertial effects. This timescale
characterizes the turnover time of the smallest turbulent
eddies. It is given by:

- τ_K = (v / ε)^(1/2)
    - τ_K is the Kolmogorov time [s]
    - v is the kinematic viscosity of the fluid [m^2/s]

#### Arguments

- kinematic_viscosity : Kinematic viscosity of the fluid [m^2/s]
- turbulent_dissipation : Rate of dissipation of turbulent kinetic
    energy [m^2/s^3]

#### Returns

- Kolmogorov time [s]

#### Examples

```py title="Kolmogorov time of a fluid"
import particula as par
par.gas.get_kolmogorov_time(1.5e-5, 0.1)
# Output: 0.3872983346207417
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
the geometric collision rate of sedimenting droplets. Part 2. Theory
and parameterization. New Journal of Physics, 10.
https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {"kinematic_viscosity": "positive", "turbulent_dissipation": "positive"}
)
def get_kolmogorov_time(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_kolmogorov_velocity

[Show source in kolmogorov_module.py:105](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kolmogorov_module.py#L105)

Calculate the Kolmogorov velocity scale.

The Kolmogorov velocity scale characterizes the smallest turbulent velocity
fluctuations and is given by:

- v_k = (v ε)^(1/4)
    - v_k is the Kolmogorov velocity scale [m/s]
    - v is the kinematic viscosity of the fluid [m^2/s]
    - ε is the rate of dissipation of turbulent kinetic energy [m^2/s^3]

#### Arguments

- kinematic_viscosity : Kinematic viscosity of the fluid [m^2/s]
- turbulent_dissipation : Rate of dissipation of turbulent kinetic
    energy [m^2/s^3]

#### Returns

- Kolmogorov velocity scale [m/s]

#### Examples

```py title="Kolmogorov velocity scale of a fluid"
import particula as par
par.gas.get_kolmogorov_velocity(1.5e-5, 0.1)
# Output: 0.3872983346207417
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
the geometric collision rate of sedimenting droplets. Part 2. Theory
and parameterization. New Journal of Physics, 10.
https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {"kinematic_viscosity": "positive", "turbulent_dissipation": "positive"}
)
def get_kolmogorov_velocity(
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# mean_free_path.md

# Mean Free Path

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Mean Free Path

> Auto-generated documentation for [particula.gas.properties.mean_free_path](https://github.com/uncscode/particula/blob/main/particula/gas/properties/mean_free_path.py) module.

## get_molecule_mean_free_path

[Show source in mean_free_path.py:30](https://github.com/uncscode/particula/blob/main/particula/gas/properties/mean_free_path.py#L30)

Calculate the mean free path of a gas molecule in air.

This function calculates λ based on the input conditions. If
dynamic_viscosity is not provided, it is computed via
get_dynamic_viscosity(temperature).

- λ = (2 × μ / P) / √(8 × M / (π × R × T))
    - λ is Mean free path [m].
    - μ is Dynamic viscosity [Pa·s].
    - P is Gas pressure [Pa].
    - M is Molar mass [kg/mol].
    - R is Universal gas constant [J/(mol·K)].
    - T is Gas temperature [K].

#### Arguments

- molar_mass : The molar mass of the gas molecule [kg/mol].
- temperature : The temperature of the gas [K].
- pressure : The pressure of the gas [Pa].
- dynamic_viscosity : The dynamic viscosity of the gas [Pa·s].
    If None, it will be calculated based on the temperature.

#### Returns

- Mean free path of the gas molecule in meters (m).

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_molecule_mean_free_path()
# Returns mean free path at ~298K and 101325Pa, ~6.5e-8 m
```

#### References

- "Mean Free Path," Wikipedia, The Free Encyclopedia.
  https://en.wikipedia.org/wiki/Mean_free_path

#### Signature

```python
def get_molecule_mean_free_path(
    molar_mass: ignore = MOLECULAR_WEIGHT_AIR,
    temperature: float = 298.15,
    pressure: float = 101325,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [MOLECULAR_WEIGHT_AIR](../../util/constants.md#molecular_weight_air)


---
# normalize_accel_variance.md

# Normalize Accel Variance

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Normalize Accel Variance

> Auto-generated documentation for [particula.gas.properties.normalize_accel_variance](https://github.com/uncscode/particula/blob/main/particula/gas/properties/normalize_accel_variance.py) module.

## get_normalized_accel_variance_ao2008

[Show source in normalize_accel_variance.py:12](https://github.com/uncscode/particula/blob/main/particula/gas/properties/normalize_accel_variance.py#L12)

Calculate the normalized acceleration variance in isotropic turbulence.

This coefficient describes the statistical behavior of acceleration
fluctuations in turbulent flows.

- a_o = (11 + 7 R_λ) / (205 + R_λ)
    - a_o is Normalized acceleration variance in isotropic turbulence [-].
    - R_λ is Taylor-microscale Reynolds number [-].

Where:
    - a_o (accel_variance) is Normalized acceleration variance in isotropic
      turbulence [-].
    - R_λ (re_lambda) is Taylor-microscale Reynolds number [-].
    - ε (numerical_stability_epsilon) is Small number added to R_λ
      for numerical stability.

#### Arguments

- re_lambda : Taylor-microscale Reynolds number [-]

#### Returns

- accel_variance : Normalized acceleration variance [-]

#### Examples

```py title="Example Usage"
import particula as par
par.gas.get_normalized_accel_variance_ao2008(500.0)
# Output: ~0.05
```

#### References

- The equivalent numerically stable version used is this.
    (7 + 11 / (R_λ + ε)) / (1 + 205 / (R_λ + ε))
- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2. Theory
    and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs({"re_lambda": "positive"})
def get_normalized_accel_variance_ao2008(
    re_lambda: Union[float, NDArray[np.float64]],
    numerical_stability_epsilon: float = 1e-14,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# pressure_function.md

# Pressure Function

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Pressure Function

> Auto-generated documentation for [particula.gas.properties.pressure_function](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py) module.

## get_partial_pressure

[Show source in pressure_function.py:11](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py#L11)

Calculate the partial pressure of a gas from its concentration, molar mass,
and temperature.

- p = (c × R × T) / M
    - p is Partial pressure [Pa].
    - c is Gas concentration [kg/m³].
    - R is Universal gas constant [J/(mol·K)].
    - T is Temperature [K].
    - M is Molar mass [kg/mol].

#### Arguments

- concentration : Concentration of the gas [kg/m³].
- molar_mass : Molar mass of the gas [kg/mol].
- temperature : Temperature [K].

#### Returns

- Partial pressure of the gas [Pa].

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_partial_pressure(1.2, 0.02897, 298)
# Output: ~986.4 Pa
```

#### References

- Wikipedia contributors, "Ideal gas law," Wikipedia,
  https://en.wikipedia.org/wiki/Ideal_gas_law

#### Signature

```python
@validate_inputs(
    {"concentration": "nonnegative", "molar_mass": "positive", "temperature": "positive"}
)
def get_partial_pressure(
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_saturation_ratio_from_pressure

[Show source in pressure_function.py:57](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py#L57)

Calculate the saturation ratio of the gas at a given partial pressure and
pure vapor pressure.

The saturation ratio is defined as the ratio of partial pressure to the
pure vapor pressure.

- S = p / p_vap
    - S is Saturation ratio (dimensionless).
    - p is Partial pressure [Pa].
    - p_vap is Pure vapor pressure [Pa].

#### Arguments

- partial_pressure : Partial pressure [Pa].
- pure_vapor_pressure : Pure vapor pressure [Pa].

#### Returns

- Saturation ratio of the gas (dimensionless).

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_saturation_ratio_from_pressure(800.0, 1000.0)
# Output: 0.8
```

#### References

- Wikipedia contributors, "Relative humidity," Wikipedia,
  https://en.wikipedia.org/wiki/Relative_humidity

#### Signature

```python
@validate_inputs({"partial_pressure": "positive", "pure_vapor_pressure": "positive"})
def get_saturation_ratio_from_pressure(
    partial_pressure: Union[float, NDArray[np.float64]],
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# taylor_microscale_module.md

# Taylor Microscale Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Taylor Microscale Module

> Auto-generated documentation for [particula.gas.properties.taylor_microscale_module](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py) module.

## get_lagrangian_taylor_microscale_time

[Show source in taylor_microscale_module.py:13](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L13)

Calculate the Lagrangian Taylor microscale time.

The Lagrangian Taylor microscale time (τ_T) represents the characteristic
time for the decay of turbulent velocity correlations. It provides insight
into the memory of turbulent fluid elements. It is given by:

- τ_T = τ_k * (2 R_λ / (15^(1/2) a_o))^(1/2)
    - τ_T is Lagrangian Taylor microscale time [s]
    - τ_k (kolmogorov_time) is Kolmogorov time scale [s]
    - R_λ (re_lambda) is Taylor-microscale Reynolds number [-]
    - a_o (accel_variance) is Normalized acceleration variance in isotropic
        turbulence [-]

#### Arguments

- kolmogorov_time : Kolmogorov time scale [s]
- re_lambda : Taylor-microscale Reynolds number [-]
- accel_variance : Normalized acceleration variance in isotropic
    turbulence [-]

#### Examples

``` py title="Example Usage"
import particula as par
par.gas.get_lagrangian_taylor_microscale_time(0.387, 500, 0.05)
# Output: 0.3872983346207417
```

#### Returns

- Lagrangian Taylor microscale time [s]

#### Signature

```python
@validate_inputs(
    {
        "kolmogorov_time": "positive",
        "re_lambda": "positive",
        "accel_variance": "positive",
    }
)
def get_lagrangian_taylor_microscale_time(
    kolmogorov_time: Union[float, NDArray[np.float64]],
    re_lambda: Union[float, NDArray[np.float64]],
    accel_variance: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_taylor_microscale

[Show source in taylor_microscale_module.py:59](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L59)

Calculate the Taylor microscale.

The Taylor microscale (λ) represents an intermediate length scale in
turbulence, linking the dissipative and energy-containing ranges of
turbulence. It characterizes the smoothness of velocity fluctuations
in turbulent flows. It is given by:

- λ = u' * (15 ν² / ε)^(1/2)
    - λ is Taylor microscale [m]
    - u' (rms_velocity) is Fluid RMS fluctuation velocity [m/s]
    - v (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s]
    - ε (turbulent_dissipation) is Turbulent kinetic energy dissipation
        rate [m²/s³]

#### Arguments

- fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s]
- kinematic_viscosity : Kinematic viscosity of the fluid [m²/s]
- turbulent_dissipation : Turbulent kinetic energy dissipation rate
    [m²/s³]

#### Returns

- Taylor microscale [m]

#### Examples

``` py title="Example Usage"
import particula as par
par.gas.get_taylor_microscale(0.35, 1.5e-5, 0.1)
# Output: 0.00021081851067789195
```

#### References

- https://en.wikipedia.org/wiki/Taylor_microscale

#### Signature

```python
@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "kinematic_viscosity": "positive",
        "turbulent_dissipation": "positive",
    }
)
def get_taylor_microscale(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    turbulent_dissipation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_taylor_microscale_reynolds_number

[Show source in taylor_microscale_module.py:110](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L110)

Compute the Taylor-microscale Reynolds number (Re_λ).

The Taylor-scale micro Reynolds number is a dimensionless quantity used in
turbulence studies to characterize the relative importance of inertial and
viscous forces at the Taylor microscale.

- Re_λ = (u' λ) / ν
    - u' (fluid_rms_velocity) is Fluid (RMS) velocity fluctuation [m/s].
    - λ (taylor_microscale) is Taylor microscale [m].
    - ν (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s].

#### Arguments

- fluid_rms_velocity : Fluid RMS velocity fluctuation [m/s].
- taylor_microscale : Taylor microscale [m].
- kinematic_viscosity : Kinematic viscosity of the fluid [m²/s].

#### Returns

- Taylor-microscale Reynolds number [dimensionless].

#### Examples

``` py title="Example Usage"
import particula as par
par.gas.get_taylor_microscale_reynolds_number(0.35, 0.00021, 1.5e-5)
# Output: 500.0
```

#### References

- https://en.wikipedia.org/wiki/Taylor_microscale

#### Signature

```python
@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "taylor_microscale": "positive",
        "kinematic_viscosity": "positive",
    }
)
def get_taylor_microscale_reynolds_number(
    fluid_rms_velocity: Union[float, NDArray[np.float64]],
    taylor_microscale: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# thermal_conductivity.md

# Thermal Conductivity

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Thermal Conductivity

> Auto-generated documentation for [particula.gas.properties.thermal_conductivity](https://github.com/uncscode/particula/blob/main/particula/gas/properties/thermal_conductivity.py) module.

## get_thermal_conductivity

[Show source in thermal_conductivity.py:13](https://github.com/uncscode/particula/blob/main/particula/gas/properties/thermal_conductivity.py#L13)

Thermal conductivity of air as a function of temperature.

Calculate the thermal conductivity of air as a function of temperature.
Based on a simplified linear relation from atmospheric science literature.
Only valid for temperatures within the range typically found on
Earth's surface.

- k(T) = 1e-3 × (4.39 + 0.071 × T)
    - k(T) is Thermal conductivity [W/(m·K)].
    - T is Temperature [K].

#### Arguments

- temperature : The temperature in Kelvin (K).

#### Returns

- The thermal conductivity [W/(m·K)] or [J/(m·s·K)].

#### Examples

``` py title="Example Usage"
import particula as par
par.gas.get_thermal_conductivity(300)
# ~0.449 W/(m·K)
```

#### References

- Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.

#### Signature

```python
@validate_inputs({"temperature": "nonnegative"})
def get_thermal_conductivity(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# vapor_pressure_module.md

# Vapor Pressure Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Vapor Pressure Module

> Auto-generated documentation for [particula.gas.properties.vapor_pressure_module](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py) module.

## get_antoine_vapor_pressure

[Show source in vapor_pressure_module.py:13](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py#L13)

Calculate vapor pressure using the Antoine equation.

The Antoine equation relates the logarithm of vapor pressure to
temperature for a pure substance.

- P = 10^(a - b / (T - c)) × 133.322
    - P is Vapor pressure [Pa].
    - a, b, c is Antoine equation parameters (dimensionless).
    - T is Temperature [K].

#### Arguments

- a : Antoine parameter a (dimensionless).
- b : Antoine parameter b (dimensionless).
- c : Antoine parameter c (dimensionless).
- temperature : Temperature in Kelvin [K].

#### Returns

- Vapor pressure in Pascals [Pa].

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_antoine_vapor_pressure(
    8.07131, 1730.63, 233.426, 373.15
)
# Output: ~101325 Pa (roughly 1 atm)
```

#### References

- https://en.wikipedia.org/wiki/Antoine_equation
- Kelvin conversion details:
  https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

#### Signature

```python
@validate_inputs(
    {"a": "finite", "b": "finite", "c": "finite", "temperature": "positive"}
)
def get_antoine_vapor_pressure(
    a: Union[float, NDArray[np.float64]],
    b: Union[float, NDArray[np.float64]],
    c: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_buck_vapor_pressure

[Show source in vapor_pressure_module.py:124](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py#L124)

Calculate vapor pressure using the Buck equation for water vapor.

Uses separate empirical formulas below 0 °C and above 0 °C to compute
water vapor pressure.

- For T < 0 °C, as
    p = 6.1115 × exp( (23.036 - T/333.7) × T / (279.82 + T ) ) × 100
- For T ≥ 0 °C, as
    p = 6.1121 × exp( (18.678 - T/234.5) × T / (257.14 + T ) ) × 100
    - p is Vapor pressure [Pa].
    - T is Temperature in Celsius [°C] (converted internally from Kelvin).

#### Arguments

- temperature : Temperature in Kelvin [K].

#### Returns

- Vapor pressure in Pascals [Pa].

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_buck_vapor_pressure(273.15)
# Output: ~611 Pa (around ice point)
```

#### References

- Buck, A. L., (1981)
- https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
@validate_inputs({"temperature": "positive"})
def get_buck_vapor_pressure(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_clausius_clapeyron_vapor_pressure

[Show source in vapor_pressure_module.py:66](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py#L66)

Calculate vapor pressure using Clausius-Clapeyron equation.

This function calculates the final vapor pressure based on an initial
temperature/pressure pair and the latent heat of vaporization,
assuming ideal gas behavior.

- P_final = P_initial × exp( (L / R) × (1 / T_initial - 1 / T_final) )
    - P_final is Final vapor pressure [Pa].
    - P_initial is Initial vapor pressure [Pa].
    - L is Latent heat of vaporization [J/mol].
    - R is Universal gas constant [J/(mol·K)].
    - T_initial is Initial temperature [K].
    - T_final is Final temperature [K].

#### Arguments

- latent_heat : Latent heat of vaporization [J/mol].
- temperature_initial : Initial temperature [K].
- pressure_initial : Initial vapor pressure [Pa].
- temperature : Final temperature [K].
- gas_constant : Gas constant (default 8.314 J/(mol·K)).

#### Returns

- Pure vapor pressure [Pa].

#### Examples

```py title="Example usage"
import particula as par
par.gas.get_clausius_clapeyron_vapor_pressure(
    40660, 373.15, 101325, 300
)
# Output: ~35307 Pa
```

#### References

- https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

#### Signature

```python
@validate_inputs(
    {
        "latent_heat": "positive",
        "temperature_initial": "positive",
        "pressure_initial": "nonnegative",
        "temperature": "positive",
    }
)
def get_clausius_clapeyron_vapor_pressure(
    latent_heat: Union[float, NDArray[np.float64]],
    temperature_initial: Union[float, NDArray[np.float64]],
    pressure_initial: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    gas_constant: float = GAS_CONSTANT,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [GAS_CONSTANT](../../util/constants.md#gas_constant)


---
# species.md

# Species

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Species

> Auto-generated documentation for [particula.gas.species](https://github.com/uncscode/particula/blob/main/particula/gas/species.py) module.

## GasSpecies

[Show source in species.py:23](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L23)

Represents an individual or array of gas species with properties like
name, molar mass, vapor pressure, and condensability.

#### Attributes

- name : The name of the gas species.
- molar_mass : The molar mass of the gas species in kg/mol.
- pure_vapor_pressure_strategy : The strategy (or list of strategies)
  for calculating the pure vapor pressure of the gas species.
- condensable : Indicates whether the gas species is condensable.
- concentration : The concentration of the gas species in kg/m^3.

#### Methods

- get_name : Return the name of the gas species.
- get_molar_mass : Return the molar mass in kg/mol.
- get_condensable : Return whether the species is condensable.
- get_concentration : Return the concentration in kg/m^3.
- get_pure_vapor_pressure : Calculate pure vapor pressure at a given Temp.
- get_partial_pressure : Calculate partial pressure at a given Temp.
- get_saturation_ratio : Calculate saturation ratio at a given Temp.
- get_saturation_concentration : Calculate saturation concentration at a
  given Temperature.
- add_concentration : Add concentration to the species.
- set_concentration : Overwrite concentration value.

#### Examples

```py title="GasSpecies usage example"
import particula as par
constant_vapor_pressure = par.gas.ConstantVaporPressureStrategy(2330)
species = par.gas.GasSpecies(
    name="Water",
    molar_mass=0.018,
    vapor_pressure_strategy=constant_vapor_pressure,
    condensable=True,
    concentration=1e-3,  # kg/m^3
)
print(species.get_name(), species.get_concentration())
```

#### Signature

```python
class GasSpecies:
    @validate_inputs({"molar_mass": "positive"})
    def __init__(
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float64]],
        vapor_pressure_strategy: Union[
            VaporPressureStrategy, list[VaporPressureStrategy]
        ] = ConstantVaporPressureStrategy(0.0),
        condensable: Union[bool, NDArray[np.bool_]] = True,
        concentration: Union[float, NDArray[np.float64]] = 0.0,
    ) -> None: ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)

### GasSpecies().__len__

[Show source in species.py:108](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L108)

Return the number of gas species (1 if scalar; array length if
ndarray).

#### Returns

- float or int : Number of species (array length or 1).

#### Examples

```py title="Example of len()"
len(gas_object)
```

#### Signature

```python
def __len__(self): ...
```

### GasSpecies().__str__

[Show source in species.py:99](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L99)

Return a string representation of the GasSpecies object.

#### Returns

- str : The string name of the gas species.

#### Signature

```python
def __str__(self): ...
```

### GasSpecies()._check_if_negative_concentration

[Show source in species.py:375](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L375)

Ensure concentration is not negative. Log a warning if it is and set
to 0.

#### Arguments

- values : Concentration values to check.

#### Returns

- Corrected concentration (≥ 0).

#### Signature

```python
def _check_if_negative_concentration(
    self, values: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies()._check_non_positive_value

[Show source in species.py:396](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L396)

Raise an error if any value is non-positive.

#### Arguments

- value : The numeric value(s) to check.
- name : Name of the parameter for the error message.

#### Raises

- ValueError : If any value <= 0 is detected.

#### Signature

```python
def _check_non_positive_value(
    self, value: Union[float, NDArray[np.float64]], name: str
) -> None: ...
```

### GasSpecies().add_concentration

[Show source in species.py:340](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L340)

Add concentration (kg/m^3) to the gas species.

#### Arguments

- added_concentration : The amount to add in kg/m^3.

#### Examples

```py title="Example of add_concentration()"
gas_object.add_concentration(1e-10)
```

#### Signature

```python
def add_concentration(
    self, added_concentration: Union[float, NDArray[np.float64]]
) -> None: ...
```

### GasSpecies().get_concentration

[Show source in species.py:169](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L169)

Return the concentration of the gas species in kg/m^3.

#### Returns

- Species concentration.

#### Signature

```python
def get_concentration(self) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_condensable

[Show source in species.py:155](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L155)

Check if the gas species is condensable.

#### Returns

- True if condensable, else False.

#### Examples

``` py title="Example of get_condensable()"
gas_object.get_condensable()
```

#### Signature

```python
def get_condensable(self) -> Union[bool, NDArray[np.bool_]]: ...
```

### GasSpecies().get_molar_mass

[Show source in species.py:141](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L141)

Return the molar mass of the gas species in kg/mol.

#### Returns

- Molar mass in kg/mol.

#### Examples

```py title="Example of get_molar_mass()"
gas_object.get_molar_mass()
```

#### Signature

```python
def get_molar_mass(self) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_name

[Show source in species.py:127](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L127)

Return the name of the gas species.

#### Returns

- Name of the gas species.

#### Examples

```py title="Example of get_name()"
gas_object.get_name()
```

#### Signature

```python
def get_name(self) -> Union[str, NDArray[np.str_]]: ...
```

### GasSpecies().get_partial_pressure

[Show source in species.py:214](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L214)

Calculate the partial pressure of the gas at a given temperature (K).

#### Arguments

- temperature : The temperature in Kelvin.

#### Returns

- Partial pressure in Pa.

#### Raises

- ValueError : If the vapor pressure strategy is not set.

#### Examples

```py title="Example of get_partial_pressure()"
gas_object.get_partial_pressure(temperature=298)
```

#### Signature

```python
def get_partial_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_pure_vapor_pressure

[Show source in species.py:178](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L178)

Calculate the pure vapor pressure at a given temperature (K).

#### Arguments

- temperature : The temperature in Kelvin.

#### Returns

- Pure vapor pressure in Pa.

#### Raises

- ValueError : If no vapor pressure strategy is set.

#### Examples

```py title="Example"
gas_object.get_pure_vapor_pressure(temperature=298)
```

#### Signature

```python
def get_pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_saturation_concentration

[Show source in species.py:300](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L300)

Calculate the saturation concentration at a given temperature (K).

#### Arguments

- temperature : The temperature in Kelvin.

#### Returns

- The saturation concentration in kg/m^3.

#### Raises

- ValueError : If the vapor pressure strategy is not set.

#### Examples

```py title="Example of get_saturation_concentration()"
gas_object.get_saturation_concentration(temperature=298)
```

#### Signature

```python
def get_saturation_concentration(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().get_saturation_ratio

[Show source in species.py:257](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L257)

Calculate the saturation ratio of the gas at a given temperature (K).

#### Arguments

- temperature : The temperature in Kelvin.

#### Returns

- The saturation ratio.

#### Raises

- ValueError : If the vapor pressure strategy is not set.

#### Examples

```py title="Example of get_saturation_ratio()"
gas_object.get_saturation_ratio(temperature=298)
```

#### Signature

```python
def get_saturation_ratio(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### GasSpecies().set_concentration

[Show source in species.py:356](https://github.com/uncscode/particula/blob/main/particula/gas/species.py#L356)

Overwrite the concentration of the gas species in kg/m^3.

#### Arguments

- new_concentration : The new concentration value in kg/m^3.

#### Examples

```py title="Example of set_concentration()"
gas_object.set_concentration(1e-10)
```

#### Signature

```python
def set_concentration(
    self, new_concentration: Union[float, NDArray[np.float64]]
) -> None: ...
```


---
# species_builders.md

# Species Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Species Builders

> Auto-generated documentation for [particula.gas.species_builders](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py) module.

## GasSpeciesBuilder

[Show source in species_builders.py:28](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py#L28)

Builder class for GasSpecies objects with preset default parameters.

This subclass of GasSpeciesBuilder initializes certain parameters
(e.g., name, molar_mass, vapor_pressure_strategy, etc.) to predefined
values. Suitable for quick testing or examples.

#### Methods

- build : Validate parameters and return a GasSpecies object.

- set_name : Set the name of the gas species.
- set_vapor_pressure_strategy : Set the vapor pressure strategy.
- set_condensable : Set whether the species is condensable.
- set_molar_mass : From BuilderMolarMassMixin.
- set_concentration : From BuilderConcentrationMixin.
- build : Validate parameters and return a GasSpecies object.

#### Attributes

- name : The name of the gas species.
- molar_mass : The molar mass of the gas species in kg/mol.
- vapor_pressure_strategy : The vapor pressure strategy for the
    gas species.
- condensable : Whether the gas species is condensable.
- concentration : The concentration of the gas species in the
    mixture, in kg/m^3.

#### Examples

``` py title="Create a gas species using the builder"
import particula as par
builder = par.gas.GasSpeciesBuilder()
gas_object = (
    builder.set_name("Oxygen")
    .set_molar_mass(0.032, "kg/mol")
    .set_vapor_pressure_strategy(
        par.gas.ConstantVaporPressureStrategy(vapor_pressure=101325)
    )
    .set_condensable(False)
    .set_concentration(1.2, "kg/m^3")
    .build()
)
# gas_object is now a GasSpecies instance with the specified
# parameters.
```

#### Signature

```python
class GasSpeciesBuilder(BuilderABC, BuilderMolarMassMixin, BuilderConcentrationMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderMolarMassMixin](../builder_mixin.md#buildermolarmassmixin)

### GasSpeciesBuilder().build

[Show source in species_builders.py:139](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py#L139)

Validate parameters and return a GasSpecies object.

#### Returns

- The constructed GasSpecies instance.

#### Raises

- ValueError : If any required parameters are missing or invalid.

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### GasSpeciesBuilder().set_condensable

[Show source in species_builders.py:123](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py#L123)

Set whether the gas species is condensable.

#### Arguments

- condensable : Boolean or array indicating condensability.

#### Returns

- This builder instance.

#### Signature

```python
def set_condensable(
    self, condensable: Union[bool, NDArray[np.bool_]]
) -> "GasSpeciesBuilder": ...
```

### GasSpeciesBuilder().set_name

[Show source in species_builders.py:92](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py#L92)

Set the name of the gas species.

#### Arguments

- name : The name of the gas species.

#### Returns

- This builder instance.

#### Signature

```python
def set_name(self, name: Union[str, NDArray[np.str_]]) -> "GasSpeciesBuilder": ...
```

### GasSpeciesBuilder().set_vapor_pressure_strategy

[Show source in species_builders.py:107](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py#L107)

Set the vapor pressure strategy for the gas species.

#### Arguments

- strategy : The vapor pressure strategy (or list of strategies).

#### Returns

- This builder instance.

#### Signature

```python
def set_vapor_pressure_strategy(
    self, strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]]
) -> "GasSpeciesBuilder": ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)



## PresetGasSpeciesBuilder

[Show source in species_builders.py:158](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py#L158)

Builder class for GasSpecies objects, allowing for a more fluent and
readable creation of GasSpecies instances with optional parameters.

#### Examples

``` py title="Create a gas species using the preset builder"
import particula as par
gas_object = par.gas.PresetGasSpeciesBuilder().build()
# gas_object is now a GasSpecies instance with the preset
# parameters.

#### Signature

```python
class PresetGasSpeciesBuilder(GasSpeciesBuilder):
    def __init__(self): ...
```

#### See also

- [GasSpeciesBuilder](#gasspeciesbuilder)

### PresetGasSpeciesBuilder().build

[Show source in species_builders.py:182](https://github.com/uncscode/particula/blob/main/particula/gas/species_builders.py#L182)

Validate parameters and return a GasSpecies object with preset
defaults.

#### Returns

- GasSpecies : The constructed GasSpecies instance.

#### Signature

```python
def build(self) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)


---
# species_factories.md

# Species Factories

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Species Factories

> Auto-generated documentation for [particula.gas.species_factories](https://github.com/uncscode/particula/blob/main/particula/gas/species_factories.py) module.

## GasSpeciesFactory

[Show source in species_factories.py:12](https://github.com/uncscode/particula/blob/main/particula/gas/species_factories.py#L12)

Factory for creating species builders that produce GasSpecies objects.

This class provides methods to retrieve a builder (e.g., 'gas_species'
or 'preset_gas_species') and instantiate a GasSpecies object from it
using user-specified parameters.

#### Methods

- get_builders : Return a dictionary of builder objects.
- get_strategy : Construct and return a GasSpecies object with the
  chosen builder.

#### Returns

- GasSpecies : A gas species instance from the specified builder.

#### Raises

- ValueError : If an unknown strategy type is provided.

#### Examples

```py title="Create a preset gas species using the factory"
import particula as par
factory = par.gas.GasSpeciesFactory()
gas_object = factory.get_strategy("preset_gas_species", parameters)
```

```py title="Create a gas species using the factory"
import particula as par
factory = par.gas.GasSpeciesFactory()
parameters = {
    "name": "Oxygen",
    "molar_mass": 0.032,
    "vapor_pressure_strategy": ConstantVaporPressureStrategy(
        vapor_pressure=101325
    ),
    "condensable": False,
    "concentration": 1.2,
}
gas_object = factory.get_strategy("gas_species", parameters)
```

#### Signature

```python
class GasSpeciesFactory(
    StrategyFactoryABC[Union[GasSpeciesBuilder, PresetGasSpeciesBuilder], GasSpecies]
): ...
```

#### See also

- [GasSpeciesBuilder](./species_builders.md#gasspeciesbuilder)
- [GasSpecies](./species.md#gasspecies)
- [PresetGasSpeciesBuilder](./species_builders.md#presetgasspeciesbuilder)

### GasSpeciesFactory().get_builders

[Show source in species_factories.py:63](https://github.com/uncscode/particula/blob/main/particula/gas/species_factories.py#L63)

Return a mapping of strategy types to builder instances.

#### Returns

- dict[str, Union[GasSpeciesBuilder, PresetGasSpeciesBuilder]] :
  A dictionary where:
    * "gas_species" -> GasSpeciesBuilder
    * "preset_gas_species" -> PresetGasSpeciesBuilder

#### Examples

```py title="get_builders Example"
import particula as par
factory = par.gas.GasSpeciesFactory()
builder_map = factory.get_builders()
# builder_map["gas_species"] -> GasSpeciesBuilder()
```

#### Signature

```python
def get_builders(self): ...
```


---
# vapor_pressure_builders.md

# Vapor Pressure Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Vapor Pressure Builders

> Auto-generated documentation for [particula.gas.vapor_pressure_builders](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py) module.

## AntoineBuilder

[Show source in vapor_pressure_builders.py:37](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L37)

Builder class for AntoineVaporPressureStrategy. It allows setting
the coefficients 'a', 'b', and 'c' separately and then building the
strategy object. Follows the general form of the Antoine equation in
Unicode:

log₁₀(P) = a − b / (T − c)

#### Attributes

- a : Coefficient "a" of the Antoine equation (dimensionless).
- b : Coefficient "b" (in Kelvin).
- c : Coefficient "c" (in Kelvin).

#### Methods

- set_a : Set the coefficient "a" of the Antoine equation.
- set_b : Set the coefficient "b".
- set_c : Set the coefficient "c".
- build : Validate parameters and return an AntoineVaporPressureStrategy.

#### Examples

``` py title="AntoineBuilder"
strategy = (
    AntoineBuilder()
    .set_a(8.07131)
    .set_b(1730.63)
    .set_c(233.426)
    .build()
)
```

``` py title="AntoineBuilder with units"
strategy = (
    AntoineBuilder()
    .set_a(8.07131)
    .set_b(1730.63, "K")
    .set_c(233.426, "K")
    .build()
)
```

#### References

- `-` *Equation* - log10(P_mmHG) = a - b / (Temperature_K - c)
  - `(Reference` - https://en.wikipedia.org/wiki/Antoine_equation)
- "Vapor Pressure,"
  [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure)
- "Atmospheric Pressure Unit Conversions,"
  [Wikipedia](https://en.wikipedia.org/wiki/Pascal_(unit))

#### Signature

```python
class AntoineBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### AntoineBuilder().build

[Show source in vapor_pressure_builders.py:123](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L123)

Validate and return an AntoineVaporPressureStrategy using the set
coefficients.

#### Returns

- Configured with coefficients a, b, and c.

#### Signature

```python
def build(self) -> AntoineVaporPressureStrategy: ...
```

#### See also

- [AntoineVaporPressureStrategy](./vapor_pressure_strategies.md#antoinevaporpressurestrategy)

### AntoineBuilder().set_a

[Show source in vapor_pressure_builders.py:95](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L95)

Set the coefficient 'a' of the Antoine equation.

#### Signature

```python
def set_a(self, a: float, a_units: Optional[str] = None) -> "AntoineBuilder": ...
```

### AntoineBuilder().set_b

[Show source in vapor_pressure_builders.py:107](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L107)

Set the coefficient 'b' of the Antoine equation.

#### Signature

```python
@validate_inputs({"b": "positive"})
def set_b(self, b: float, b_units: str = "K") -> "AntoineBuilder": ...
```

### AntoineBuilder().set_c

[Show source in vapor_pressure_builders.py:115](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L115)

Set the coefficient 'c' of the Antoine equation.

#### Signature

```python
@validate_inputs({"c": "positive"})
def set_c(self, c: float, c_units: str = "K") -> "AntoineBuilder": ...
```



## ClausiusClapeyronBuilder

[Show source in vapor_pressure_builders.py:135](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L135)

Builder class for ClausiusClapeyronStrategy. This class facilitates
setting the latent heat of vaporization, initial temperature, and initial
pressure with unit handling and then builds the strategy object.

The Clausius–Clapeyron relation can be approximated as:

- dP / dT = (L / (R × T²))

#### Methods

- set_latent_heat : Set latent heat in J/kg (or convertible units).
- set_temperature_initial : Set initial temperature in K
    (or convertible units).
- set_pressure_initial : Set initial pressure in Pa
    (or convertible units).
- build : Validate parameters and return a ClausiusClapeyronStrategy.

#### Examples

``` py title="ClausiusClapeyronBuilder"
strategy = (
    ClausiusClapeyronBuilder()
    .set_latent_heat(2260)
    .set_temperature_initial(373.15)
    .set_pressure_initial(101325)
    .build()
)
```

``` py title="ClausiusClapeyronBuilder with units"
strategy = (
    ClausiusClapeyronBuilder()
    .set_latent_heat(2260, "J/kg")
    .set_temperature_initial(373.15, "K")
    .set_pressure_initial(101325, "Pa")
    .build()
)
```

#### References

- `-` *Equation* - dP/dT = L / (R * T^2)
  https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

#### Signature

```python
class ClausiusClapeyronBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### ClausiusClapeyronBuilder().build

[Show source in vapor_pressure_builders.py:230](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L230)

Validate parameters and return a ClausiusClapeyronStrategy object.

#### Returns

- Configured with latent heat, initial Temperature, and Pressure.

#### Signature

```python
def build(self) -> ClausiusClapeyronStrategy: ...
```

#### See also

- [ClausiusClapeyronStrategy](./vapor_pressure_strategies.md#clausiusclapeyronstrategy)

### ClausiusClapeyronBuilder().set_latent_heat

[Show source in vapor_pressure_builders.py:191](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L191)

Set the latent heat of vaporization: Default units J/kg.

#### Signature

```python
@validate_inputs({"latent_heat": "positive"})
def set_latent_heat(
    self, latent_heat: float, latent_heat_units: str
) -> "ClausiusClapeyronBuilder": ...
```

### ClausiusClapeyronBuilder().set_pressure_initial

[Show source in vapor_pressure_builders.py:217](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L217)

Set the initial pressure. Default units: Pa.

#### Signature

```python
@validate_inputs({"pressure_initial": "positive"})
def set_pressure_initial(
    self, pressure_initial: float, pressure_initial_units: str
) -> "ClausiusClapeyronBuilder": ...
```

### ClausiusClapeyronBuilder().set_temperature_initial

[Show source in vapor_pressure_builders.py:204](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L204)

Set the initial temperature. Default units: K.

#### Signature

```python
@validate_inputs({"temperature_initial": "positive"})
def set_temperature_initial(
    self, temperature_initial: float, temperature_initial_units: str
) -> "ClausiusClapeyronBuilder": ...
```



## ConstantBuilder

[Show source in vapor_pressure_builders.py:245](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L245)

Builder class for ConstantVaporPressureStrategy. This class facilitates
setting the constant vapor pressure and then building the strategy object.

#### Attributes

- `-` *vapor_pressure* - The vapor pressure in Pa (scalar/float).

#### Methods

- `-` *set_vapor_pressure* - Set the constant vapor pressure in Pa
  (or convertible units).
- build : Validate parameters and return a ConstantVaporPressureStrategy.

#### Examples

``` py title="ConstantBuilder"
strategy = (
    ConstantBuilder()
    .set_vapor_pressure(101325)
    .build()
)
```

``` py title="ConstantBuilder with units"
strategy = (
    ConstantBuilder()
    .set_vapor_pressure(1, "atm")
    .build()
)
```

#### References

- `-` *Equation* - P = vapor_pressure
  https://en.wikipedia.org/wiki/Vapor_pressure

#### Signature

```python
class ConstantBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### ConstantBuilder().build

[Show source in vapor_pressure_builders.py:297](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L297)

Validate parameters and return a ConstantVaporPressureStrategy object.

#### Returns

- Configured with vapor_pressure in Pa.

#### Signature

```python
def build(self) -> ConstantVaporPressureStrategy: ...
```

#### See also

- [ConstantVaporPressureStrategy](./vapor_pressure_strategies.md#constantvaporpressurestrategy)

### ConstantBuilder().set_vapor_pressure

[Show source in vapor_pressure_builders.py:284](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L284)

Set the constant vapor pressure.

#### Signature

```python
@validate_inputs({"vapor_pressure": "positive"})
def set_vapor_pressure(
    self, vapor_pressure: float, vapor_pressure_units: str
) -> "ConstantBuilder": ...
```



## WaterBuckBuilder

[Show source in vapor_pressure_builders.py:308](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L308)

Builder class for WaterBuckStrategy.

This class facilitates the building of the WaterBuckStrategy object.
Which as of now has no additional parameters to set but could be
extended in the future (e.g., ice-only calculations).

#### Examples

```py title="WaterBuckBuilder"
import particula as par
strategy = par.gas.WaterBuckBuilder().build()
```

#### Signature

```python
class WaterBuckBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### WaterBuckBuilder().build

[Show source in vapor_pressure_builders.py:325](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L325)

Build and return a WaterBuckStrategy object.

#### Returns

- Configured for water-specific Buck vapor pressure.

#### Signature

```python
def build(self) -> WaterBuckStrategy: ...
```

#### See also

- [WaterBuckStrategy](./vapor_pressure_strategies.md#waterbuckstrategy)


---
# vapor_pressure_factories.md

# Vapor Pressure Factories

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Vapor Pressure Factories

> Auto-generated documentation for [particula.gas.vapor_pressure_factories](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py) module.

## VaporPressureFactory

[Show source in vapor_pressure_factories.py:22](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py#L22)

Factory class to create vapor pressure strategy
builders.

This class provides a way to generate multiple vapor pressure calculation
strategies (e.g., constant, Antoine, Clausius-Clapeyron, or Water Buck) by
commissioning the appropriate builder. It is useful for scenarios requiring
a flexible way to switch or extend vapor pressure calculation methods.

#### Attributes

- None

#### Methods

- get_builders : Returns the mapping of strategy types to builder
  instances.
- get_strategy : Returns the selected vapor pressure strategy,
  given a strategy type and parameters.

#### Examples

```py title="Example VaporPressureFactory usage"
import particula as par

factory = par.gas.VaporPressureFactory()
# Create a constant vapor pressure strategy:
strategy = factory.get_strategy(
    "constant", {"constant_vapor_pressure": 101325.0}
)
# strategy is an instance of ConstantVaporPressureStrategy
```

#### References

- "Vapor Pressure,"
[Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).

#### Signature

```python
class VaporPressureFactory(
    StrategyFactoryABC[
        Union[
            ConstantBuilder, AntoineBuilder, ClausiusClapeyronBuilder, WaterBuckBuilder
        ],
        Union[
            ConstantVaporPressureStrategy,
            AntoineVaporPressureStrategy,
            ClausiusClapeyronStrategy,
            WaterBuckStrategy,
        ],
    ]
): ...
```

#### See also

- [AntoineBuilder](./vapor_pressure_builders.md#antoinebuilder)
- [AntoineVaporPressureStrategy](./vapor_pressure_strategies.md#antoinevaporpressurestrategy)
- [ClausiusClapeyronBuilder](./vapor_pressure_builders.md#clausiusclapeyronbuilder)
- [ClausiusClapeyronStrategy](./vapor_pressure_strategies.md#clausiusclapeyronstrategy)
- [ConstantBuilder](./vapor_pressure_builders.md#constantbuilder)
- [ConstantVaporPressureStrategy](./vapor_pressure_strategies.md#constantvaporpressurestrategy)
- [WaterBuckBuilder](./vapor_pressure_builders.md#waterbuckbuilder)
- [WaterBuckStrategy](./vapor_pressure_strategies.md#waterbuckstrategy)

### VaporPressureFactory().get_builders

[Show source in vapor_pressure_factories.py:73](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py#L73)

Return a dictionary mapping strategy types to builder instances.

#### Returns

dict:
    - `-` *"constant"* - ConstantBuilder
    - `-` *"antoine"* - AntoineBuilder
    - `-` *"clausius_clapeyron"* - ClausiusClapeyronBuilder
    - `-` *"water_buck"* - WaterBuckBuilder

#### Examples

```py
import particula as par
builders_dict = par.gas.VaporPressureFactory().get_builders()
builder = builders_dict["constant"]
# builder is an instance of ConstantBuilder
```

#### Signature

```python
def get_builders(self): ...
```


---
# vapor_pressure_strategies.md

# Vapor Pressure Strategies

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Gas](./index.md#gas) / Vapor Pressure Strategies

> Auto-generated documentation for [particula.gas.vapor_pressure_strategies](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py) module.

## AntoineVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:249](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L249)

Vapor pressure strategy using the Antoine equation.

This class calculates vapor pressure by applying the Antoine equation,
which relates temperature in Kelvin to the logarithm of vapor pressure.

#### Attributes

- a : Coefficient "a" in the Antoine equation.
- b : Coefficient "b" in the Antoine equation.
- c : Coefficient "c" in the Antoine equation.

#### Methods

- `-` *partial_pressure* - Compute partial pressure from concentration.
- `-` *concentration* - Compute concentration from partial pressure.
- `-` *saturation_ratio* - Compute ratio of partial pressure to saturation
  pressure.
- `-` *saturation_concentration* - Compute concentration at saturation pressure.
- `-` *pure_vapor_pressure* - Computes vapor pressure from the Antoine equation.

#### Examples

```py title="Antoine Vapor Pressure Example"
import particula as par
strategy = par.gas.AntoineVaporPressureStrategy(
    a=8.07131, b=1730.63, c=233.426
)
vp = strategy.pure_vapor_pressure(temperature=373.15)
# Returns the vapor pressure in Pascals
```

#### References

- "Antoine Equation,"
  [Wikipedia](https://en.wikipedia.org/wiki/Antoine_equation).
- Kelvin-based adaptation:
  https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

#### Signature

```python
class AntoineVaporPressureStrategy(VaporPressureStrategy):
    def __init__(
        self,
        a: Union[float, NDArray[np.float64]] = 0.0,
        b: Union[float, NDArray[np.float64]] = 0.0,
        c: Union[float, NDArray[np.float64]] = 0.0,
    ): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### AntoineVaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:298](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L298)

Calculate vapor pressure using the Antoine equation.

#### Arguments

- temperature : Temperature in Kelvin.

#### Returns

Vapor pressure in Pascals.

#### Examples

``` py title="Antoine Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

#### References

- `-` *Equation* - log10(P) = a - b / (T - c)
- https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
- Kelvin form:
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ClausiusClapeyronStrategy

[Show source in vapor_pressure_strategies.py:328](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L328)

Vapor pressure strategy using the Clausius-Clapeyron equation.

This class calculates vapor pressure by applying the Clausius-Clapeyron
relation, which relates how the vapor pressure of a substance changes
with temperature, given latent heat data and a reference point.

#### Attributes

- latent_heat : Latent heat of vaporization (J/mol).
- temperature_initial : Reference temperature (K).
- pressure_initial : Reference pressure (Pa).

#### Methods

- `-` *partial_pressure* - Compute partial pressure from concentration.
- `-` *concentration* - Compute concentration from partial pressure.
- `-` *saturation_ratio* - Compute ratio of partial pressure to saturation
  pressure.
- `-` *saturation_concentration* - Compute concentration at saturation pressure.
- `-` *pure_vapor_pressure* - Computes vapor pressure via Clausius-Clapeyron
  relation.

#### Examples

```py title="Clausius-Clapeyron Example"
strategy = ClausiusClapeyronStrategy(
    latent_heat=4.07e4,
    temperature_initial=298.15,
    pressure_initial=3167.0
)
vp = strategy.pure_vapor_pressure(temperature=310)
```

#### References

- "Clausius–Clapeyron relation,"
  [Wikipedia](https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation).

#### Signature

```python
class ClausiusClapeyronStrategy(VaporPressureStrategy):
    def __init__(
        self,
        latent_heat: Union[float, NDArray[np.float64]],
        temperature_initial: Union[float, NDArray[np.float64]],
        pressure_initial: Union[float, NDArray[np.float64]],
    ): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### ClausiusClapeyronStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:384](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L384)

Calculate vapor pressure using Clausius-Clapeyron equation.

#### Arguments

- temperature : Final temperature in Kelvin.

#### Returns

Pure vapor pressure in Pascals.

#### Examples

``` py title="Clausius-Clapeyron Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

#### References

- https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ConstantVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:192](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L192)

Vapor pressure strategy with a constant value.

This class returns a single, unchanging vapor pressure value regardless of
the temperature. It is useful for scenarios that require a simplified
model.

#### Attributes

- vapor_pressure : The constant vapor pressure in Pascals.

#### Methods

- `-` *partial_pressure* - Compute partial pressure from concentration.
- `-` *concentration* - Compute concentration from partial pressure.
- `-` *saturation_ratio* - Compute ratio of partial pressure to saturation
  pressure.
- `-` *saturation_concentration* - Compute concentration at saturation pressure.
- `-` *pure_vapor_pressure* - Returns the constant vapor pressure.

#### Examples

```py title="Constant Vapor Pressure Example"
import particula as par
strategy = par.gas.ConstantVaporPressureStrategy(101325.0)
vp = strategy.pure_vapor_pressure(temperature=300)
# vp is 101325.0
```

#### References

- None

#### Signature

```python
class ConstantVaporPressureStrategy(VaporPressureStrategy):
    def __init__(self, vapor_pressure: Union[float, NDArray[np.float64]]): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### ConstantVaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:226](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L226)

Return the constant vapor pressure value.

#### Arguments

- temperature : Not used.

#### Returns

The constant vapor pressure value in Pascals.

#### Examples

``` py title="Constant Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## VaporPressureStrategy

[Show source in vapor_pressure_strategies.py:28](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L28)

Base class for vapor pressure calculations.

This abstract class defines standard methods for partial pressure,
concentration, saturation ratio, saturation concentration, and pure vapor
pressure. Subclasses must implement the pure_vapor_pressure method
with specific formulae or empirical correlations for vapor pressure.

#### Attributes

- None

#### Methods

- `-` *partial_pressure* - Compute partial pressure from concentration.
- `-` *concentration* - Compute concentration from partial pressure.
- `-` *saturation_ratio* - Compute ratio of partial pressure to saturation
  pressure.
- `-` *saturation_concentration* - Compute concentration at saturation pressure.
- `-` *pure_vapor_pressure* - Abstract method to compute pure (saturation) vapor
  pressure.

#### Examples

```py title="General Usage"
# Cannot instantiate directly:
#    strategy = VaporPressureStrategy()  # Error (abstract)
# Use a derived strategy class instead.
```

#### References

- "Vapor Pressure,"
[Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).

#### Signature

```python
class VaporPressureStrategy(ABC): ...
```

### VaporPressureStrategy().concentration

[Show source in vapor_pressure_strategies.py:90](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L90)

Calculate the concentration of the gas at a given pressure and
temperature.

#### Arguments

- `-` *partial_pressure* - Pressure in Pascals.
- `-` *molar_mass* - Molar mass of the gas in kg/mol.
- `-` *temperature* - Temperature in Kelvin.

#### Returns

The concentration of the gas in kg/m^3.

#### Examples

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

#### Signature

```python
def concentration(
    self,
    partial_pressure: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().partial_pressure

[Show source in vapor_pressure_strategies.py:61](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L61)

Calculate the partial pressure of the gas from its concentration, molar
mass, and temperature.

#### Arguments

- concentration : Concentration of the gas in kg/m^3.
- molar_mass : Molar mass of the gas in kg/mol.
- temperature : Temperature in Kelvin.

#### Returns

Partial pressure of the gas in Pascals.

#### Examples

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

#### Signature

```python
def partial_pressure(
    self,
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:180](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L180)

Calculate the pure (saturation) vapor pressure at a given
temperature. Units are in Pascals Pa=kg/(m·s²).

#### Arguments

- temperature : Temperature in Kelvin.

#### Signature

```python
@abstractmethod
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().saturation_concentration

[Show source in vapor_pressure_strategies.py:151](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L151)

Calculate the saturation concentration of the gas at a given
temperature.

#### Arguments

- molar_mass : Molar mass of the gas in kg/mol.
- temperature : Temperature in Kelvin.

#### Returns

The saturation concentration of the gas in kg/m^3.

#### Examples

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

#### Signature

```python
def saturation_concentration(
    self,
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```

### VaporPressureStrategy().saturation_ratio

[Show source in vapor_pressure_strategies.py:121](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L121)

Calculate the saturation ratio of the gas at a given pressure and
temperature.

#### Arguments

- pressure : Pressure in Pascals.
- temperature : Temperature in Kelvin.

#### Returns

The saturation ratio of the gas.

#### Examples

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

#### Signature

```python
def saturation_ratio(
    self,
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## WaterBuckStrategy

[Show source in vapor_pressure_strategies.py:414](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L414)

Vapor pressure strategy using the Buck equation for water.

This class computes water vapor pressure using the Buck equation, an
empirically derived correlation often applied in meteorology to determine
the saturation vapor pressure of water.

#### Methods

- `-` *partial_pressure* - Compute partial pressure from concentration.
- `-` *concentration* - Compute concentration from partial pressure.
- `-` *saturation_ratio* - Compute ratio of partial pressure to saturation
  pressure.
- `-` *saturation_concentration* - Compute concentration at saturation pressure.
- `-` *pure_vapor_pressure* - Computes water vapor pressure from the Buck
  equation.

#### Examples

```py title="Water Buck Vapor Pressure Example"
strategy = WaterBuckStrategy()
vp = strategy.pure_vapor_pressure(temperature=298.15)
# Returns water vapor pressure in Pascals
```

#### References

- A. L. Buck, "New Equations for Computing Vapor Pressure...",
  J. Appl. Meteor. Climatol. 20(12), 1527–1532 (1981).
- https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
class WaterBuckStrategy(VaporPressureStrategy): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### WaterBuckStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:444](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L444)

Calculate vapor pressure using the Buck equation for water vapor.

#### Arguments

- `-` *temperature* - Temperature in Kelvin.

#### Returns

Vapor pressure in Pascals.

#### Examples

``` py title="Water Buck Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

#### References

- Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
  Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
  https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.
- https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```


---
# logger_setup.md

# Logger Setup

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Logger Setup

> Auto-generated documentation for [particula.logger_setup](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py) module.

#### Attributes

- `current_dir` - get path of the current directory: os.path.dirname(os.path.abspath(__file__))

- `log_dir` - add the logging directory to the path: os.path.join(current_dir, 'logging')


## setup

[Show source in logger_setup.py:58](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py#L58)

Setup for logging in the particula package.

#### Signature

```python
def setup(): ...
```


---
# activity_builders.md

# Activity Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Activity Builders

> Auto-generated documentation for [particula.particles.activity_builders](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py) module.

## ActivityIdealMassBuilder

[Show source in activity_builders.py:27](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L27)

Builds an ActivityIdealMass object for calculating activity based on
ideal mass fractions.

A concise builder for ActivityIdealMass. This class requires no extra
parameters beyond the defaults. It ensures the returned strategy follows
Raoult's Law for mass-based activities.

#### Methods

- `-` *build* - Validates any required parameters and returns the strategy.

#### Examples

```py title="Example Usage"
import particula as par
builder = par.particles.ActivityIdealMassBuilder()
strategy = builder.build()
result = strategy.activity([1.0, 2.0, 3.0])
# result -> ...
```

#### References

- "Raoult's Law,"
    [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

#### Signature

```python
class ActivityIdealMassBuilder(BuilderABC):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### ActivityIdealMassBuilder().build

[Show source in activity_builders.py:57](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L57)

Validate and return an ActivityIdealMass strategy instance.

#### Returns

- ActivityIdealMass : The validated strategy for
  ideal mass-based activity calculations.

#### Examples

```py title="Build Method Example"
builder = par.particles.ActivityIdealMassBuilder()
mass_activity_strategy = builder.build()
# Use mass_activity_strategy.activity(...)
```

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)



## ActivityIdealMolarBuilder

[Show source in activity_builders.py:75](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L75)

Builds an ActivityIdealMolar object for calculating activity from
ideal mole fractions.

This builder sets up any required parameters (e.g., molar mass) and
creates an ActivityIdealMolar strategy. Uses Raoult's Law in terms
of mole fraction.

#### Attributes

- molar_mass : Molar mass for each species, in kilograms per mole.

#### Methods

- `-` *set_molar_mass* - Assigns the molar masses (with unit
    handling).
- `-` *set_parameters* - Batch-assign parameters from a dictionary.
- `-` *build* - Finalizes the builder and returns the strategy.

#### Examples

```py title="Example Usage"
import particula as par
builder = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(0.01815, "kg/mol")
)
strategy = builder.build()
result = strategy.activity([1.0, 2.0, 3.0])
# result -> ...
```

#### References

- "Raoult's Law,"
[Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

#### Signature

```python
class ActivityIdealMolarBuilder(BuilderABC, BuilderMolarMassMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderMolarMassMixin](../builder_mixin.md#buildermolarmassmixin)

### ActivityIdealMolarBuilder().build

[Show source in activity_builders.py:115](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L115)

Validate parameters and create an ActivityIdealMolar strategy.

Ensures molar_mass is properly configured before building.

#### Returns

- ActivityIdealMolar : An ideal strategy based on mole fractions.

#### Examples

```py title="Build Method Example"
builder = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(0.028, "kg/mol")
)
molar_activity_strategy = builder.build()
# molar_activity_strategy.activity(...)
```

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)



## ActivityKappaParameterBuilder

[Show source in activity_builders.py:138](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L138)

Builds an ActivityKappaParameter object for non-ideal activity
calculations.

This builder requires kappa, density, molar_mass, and water_index.
Kappa is the hygroscopicity parameter, used to capture non-ideal
behavior. The optional water_index identifies which species is water.

#### Attributes

- kappa : NDArray of kappa parameters for each species.
- density : NDArray of densities, in kilograms per cubic meter.
- molar_mass : NDArray of molar masses, in kilograms per mole.
- water_index : Integer index of the water species.

#### Methods

- `-` *set_kappa* - Assigns kappa values (must be nonnegative).
- `-` *set_water_index* - Sets the index of the water species.
- `-` *set_density* - Assigns density values (with unit handling).
- `-` *set_molar_mass* - Assigns molar mass values (with unit
    handling).
- `-` *set_parameters* - Batch-assign parameters from a dictionary.
- `-` *build* - Finalizes checks and returns the strategy.

#### Examples

```py title="Example Usage"
import particula as par
import numpy as np

builder = (
    par.particles.ActivityKappaParameterBuilder()
    .set_kappa(np.array([0.1, 0.0]))
    .set_density(np.array([1000, 1200]), "kg/m^3"))
    .set_molar_mass(np.array([0.018, 0.058]), "kg/mol")
    .set_water_index(0)
)
strategy = builder.build()
result = strategy.activity(np.array([1.0, 2.0]))
# result -> ...
```

#### References

- Petters, M. D., and Kreidenweis, S. M. (2007).
  "A single parameter representation of hygroscopic growth and
   cloud condensation nucleus activity," Atmospheric Chemistry
   and Physics, 7(8), 1961–1971.
   [DOI](https://doi.org/10.5194/acp-7-1961-2007)

#### Signature

```python
class ActivityKappaParameterBuilder(
    BuilderABC, BuilderDensityMixin, BuilderMolarMassMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderMolarMassMixin](../builder_mixin.md#buildermolarmassmixin)

### ActivityKappaParameterBuilder().build

[Show source in activity_builders.py:235](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L235)

Validate parameters and instantiate an ActivityKappaParameter strategy.

#### Returns

- ActivityKappaParameter : The non-ideal activity strategy
  utilizing the kappa hygroscopic parameter.

#### Examples

```py title="Build Method Example"
kappa_activity_strategy = (
    par.particles.ActivityKappaParameterBuilder()
    .set_kappa([0.1, 0.2])
    .set_density([1000, 1200], "kg/m^3")
    .set_molar_mass([0.018, 0.046], "kg/mol")
    .set_water_index(0)
    .build()
)
# kappa_activity_strategy ...
```

#### Signature

```python
def build(self) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)

### ActivityKappaParameterBuilder().set_kappa

[Show source in activity_builders.py:197](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L197)

Set the kappa parameter for the activity calculation.

#### Arguments

- `kappa` - The kappa parameter for the activity calculation.
- `kappa_units` - Not used. (for interface consistency)

#### Signature

```python
def set_kappa(
    self, kappa: Union[float, NDArray[np.float64]], kappa_units: Optional[str] = None
): ...
```

### ActivityKappaParameterBuilder().set_water_index

[Show source in activity_builders.py:217](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L217)

Set the array index of the species.

#### Arguments

- `water_index` - The array index of the species.
- `water_index_units` - Not used. (for interface consistency)

#### Signature

```python
def set_water_index(self, water_index: int, water_index_units: Optional[str] = None): ...
```


---
# activity_factories.md

# Activity Factories

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Activity Factories

> Auto-generated documentation for [particula.particles.activity_factories](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py) module.

## ActivityFactory

[Show source in activity_factories.py:20](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L20)

Factory for creating activity strategy builders for liquid mixtures.

This class supports various strategies (e.g., mass-ideal, molar-ideal,
kappa-parameter) to compute activity and partial pressures of species
based on Raoult's Law or kappa hygroscopic parameter.

#### Methods

get_builders:
    Provides a mapping from strategy type to its corresponding builder.
get_strategy(strategy_type, parameters):
    Validates inputs and returns a strategy instance for the specified
    strategy type (e.g., 'mass_ideal', 'molar_ideal', or
    'kappa_parameter').

#### Returns

- `-` *ActivityStrategy* - Instance configured for the chosen activity
    approach.

#### Raises

- `-` *ValueError* - If the strategy type is unknown or if required parameters
  are missing or invalid.

#### Examples

```py title="Factory Usage Example"
import particula as par
factory = par.particles.ActivityFactory()
strategy = factory.get_strategy("mass_ideal")
result = strategy.activity([1.0, 2.0, 3.0])
# result -> ...
```

#### References

- "Raoult's Law,"
    [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

#### Signature

```python
class ActivityFactory(
    StrategyFactoryABC[
        Union[
            ActivityIdealMassBuilder,
            ActivityIdealMolarBuilder,
            ActivityKappaParameterBuilder,
        ],
        Union[ActivityIdealMass, ActivityIdealMolar, ActivityKappaParameter],
    ]
): ...
```

#### See also

- [ActivityIdealMassBuilder](./activity_builders.md#activityidealmassbuilder)
- [ActivityIdealMass](./activity_strategies.md#activityidealmass)
- [ActivityIdealMolarBuilder](./activity_builders.md#activityidealmolarbuilder)
- [ActivityIdealMolar](./activity_strategies.md#activityidealmolar)
- [ActivityKappaParameterBuilder](./activity_builders.md#activitykappaparameterbuilder)
- [ActivityKappaParameter](./activity_strategies.md#activitykappaparameter)

### ActivityFactory().get_builders

[Show source in activity_factories.py:67](https://github.com/uncscode/particula/blob/main/particula/particles/activity_factories.py#L67)

Return a mapping of strategy types to their corresponding builders.

#### Returns

- `dict[str,` *Any]* - A dictionary mapping the activity strategy type
(e.g., 'mass_ideal', 'molar_ideal', 'kappa_parameter') to a builder
instance.

#### Examples

```py title="Builders Retrieval Example"
factory = ActivityFactory()
builder_map = factory.get_builders()
mass_ideal_builder = builder_map["mass_ideal"]
# mass_ideal_builder -> ActivityIdealMassBuilder()

#### Signature

```python
def get_builders(self): ...
```


---
# activity_strategies.md

# Activity Strategies

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Activity Strategies

> Auto-generated documentation for [particula.particles.activity_strategies](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py) module.

## ActivityIdealMass

[Show source in activity_strategies.py:145](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L145)

Calculate ideal activity based on mass fractions (Raoult's Law).

#### Attributes

- None

#### Methods

- activity : Computes activity from mass concentration,
    treating mass fractions as ideal.

#### Examples

```py title="Example Usage"
import particula as par
strategy = par.particles.ActivityIdealMass()
a = strategy.activity([0.5, 1.0, 1.5])
# a -> ...
```

#### References

- "Raoult's Law,"
    [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

#### Signature

```python
class ActivityIdealMass(ActivityStrategy): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityIdealMass().activity

[Show source in activity_strategies.py:169](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L169)

Calculate the activity of a species based on mass concentration.

#### Arguments

- mass_concentration : Concentration of the species in kg/m^3.

#### Returns

- Activity of the species, unitless.

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityIdealMolar

[Show source in activity_strategies.py:94](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L94)

Calculate ideal activity based on mole fractions (Raoult's Law).

#### Attributes

- molar_mass : Molar mass of the species in kg/mol.

#### Methods

- activity : Computes ideal activity from mass concentration
  and molar mass.

#### Examples

```py title="Example Usage"
import particula as par
strategy = par.particles.ActivityIdealMolar(molar_mass=0.018)
# mass_concentration in kg/m^3
a = strategy.activity(np.array([1.2, 2.5, 3.0]))
# a -> ...
```

#### References

- "Raoult's Law,"
[Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

#### Signature

```python
class ActivityIdealMolar(ActivityStrategy):
    def __init__(self, molar_mass: Union[float, NDArray[np.float64]] = 0.0): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityIdealMolar().activity

[Show source in activity_strategies.py:128](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L128)

Calculate the activity of a species based on mass concentration.

#### Arguments

- mass_concentration : Concentration of the species in kg/m^3.

#### Returns

- Activity of the species, unitless.

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityIdealVolume

[Show source in activity_strategies.py:184](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L184)

Calculate ideal activity based on volume fractions (Raoult's Law).

#### Attributes

- density : The density of the species in kg/m^3, used to
  derive volume fractions from mass concentrations.

#### Methods

- activity : Computes activity from mass concentration and density.

#### Examples

```py title="Example Usage"
strategy = ActivityIdealVolume(density=1000.0)
a = strategy.activity(2.5)
# a -> ...
```

#### References

- "Raoult's Law,"
    [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

#### Signature

```python
class ActivityIdealVolume(ActivityStrategy):
    def __init__(self, density: Union[float, NDArray[np.float64]] = 0.0): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityIdealVolume().activity

[Show source in activity_strategies.py:216](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L216)

Calculate the activity of a species based on mass concentration.

#### Arguments

- mass_concentration : Concentration of the species in kg/m^3.
- density : Density of the species in kg/m^3.

#### Returns

- Activity of the species, unitless.

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityKappaParameter

[Show source in activity_strategies.py:235](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L235)

Non-ideal activity strategy using the kappa hygroscopic parameter.

#### Attributes

- kappa : Kappa hygroscopic parameters (array or scalar).
- density : Densities (array or scalar) in kg/m^3.
- molar_mass : Molar masses (array or scalar) in kg/mol.
- water_index : Index identifying the water species in arrays.

#### Methods

- activity : Computes non-ideal activity using kappa
  hygroscopicity approach.

#### Examples

```py title="Example Usage"
import particula as par
import numpy as np
strategy = par.particles.ActivityKappaParameter(
    kappa=np.array([0.1, 0.0]),
    density=np.array([1000.0, 1200.0]),
    molar_mass=np.array([0.018, 0.058]),
    water_index=0,
)
a = strategy.activity(np.array([1.0, 2.0]))
# a -> ...
```

#### References

- Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
  representation of hygroscopic growth and cloud condensation
  nucleus activity. Atmospheric Chemistry and Physics, 7(8),
  1961-1971. [DOI](https://doi.org/10.5194/acp-7-1961-2007).

#### Signature

```python
class ActivityKappaParameter(ActivityStrategy):
    def __init__(
        self,
        kappa: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        density: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        molar_mass: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        water_index: int = 0,
    ): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityKappaParameter().activity

[Show source in activity_strategies.py:291](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L291)

Calculate the activity of a species based on mass concentration.

#### Arguments

- mass_concentration : Concentration of the species in kg/m^3.

#### Returns

- Activity of the species, unitless.

#### References

- Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
  representation of hygroscopic growth and cloud condensation
  nucleus activity. Atmospheric Chemistry and Physics, 7(8),
  1961-1971. [DOI](https://doi.org/10.5194/acp-7-1961-2007).

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityStrategy

[Show source in activity_strategies.py:22](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L22)

Abstract base class for vapor pressure and activity calculations.

This interface is used by subclasses for computing particle activity
and partial pressures. General methods include activity() and
partial_pressure().

#### Attributes

- None

#### Methods

- get_name : Return the type of the activity strategy.
- activity : Calculate the activity of a species. (abstract method)
- partial_pressure : Calculate the partial pressure of a species
    using its pure vapor pressure and computed activity.

#### Examples

```py title="Example Subclass"
class CustomActivity(ActivityStrategy):
    def activity(self, mass_concentration):
        return 1.0

my_activity = CustomActivity()
pvap = my_activity.partial_pressure(101325.0, 1.0)
# pvap -> 101325.0
```

#### References

- "Vapor Pressure,"
    [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).

#### Signature

```python
class ActivityStrategy(ABC): ...
```

### ActivityStrategy().activity

[Show source in activity_strategies.py:55](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L55)

Calculate the activity of a species based on its mass concentration.

#### Arguments

- mass_concentration : Concentration of the species in kg/m^3.

#### Returns

- Activity of the species, unitless.

#### Signature

```python
@abstractmethod
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### ActivityStrategy().get_name

[Show source in activity_strategies.py:69](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L69)

Return the type of the activity strategy.

#### Signature

```python
def get_name(self) -> str: ...
```

### ActivityStrategy().partial_pressure

[Show source in activity_strategies.py:73](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L73)

Calculate the vapor pressure of species in the particle phase.

#### Arguments

- pure_vapor_pressure : Pure vapor pressure of the species in Pa.
- mass_concentration : Concentration of the species in kg/m^3.

#### Returns

- Vapor pressure of the particle in Pa.

#### Signature

```python
def partial_pressure(
    self,
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
    mass_concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# change_particle_representation.md

# Change Particle Representation

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Change Particle Representation

> Auto-generated documentation for [particula.particles.change_particle_representation](https://github.com/uncscode/particula/blob/main/particula/particles/change_particle_representation.py) module.

## get_particle_resolved_binned_radius

[Show source in change_particle_representation.py:18](https://github.com/uncscode/particula/blob/main/particula/particles/change_particle_representation.py#L18)

Determine binned radii for kernel calculations.

If bin_radius is provided, those edges are used directly. Otherwise,
a log-spaced array is generated based on the particle's minimum and
maximum radii and either a total number of bins or bins per radius
decade.

#### Arguments

- particle : The ParticleRepresentation instance for radius binning.
- bin_radius : Optional array of radius bin edges in meters.
- total_bins : Exact number of bins to generate, if set.
- bins_per_radius_decade : Number of bins per decade of radius,
  used only if total_bins is None.

#### Returns

- NDArray[np.float64] : The bin edges (radii) in meters.

#### Raises

- ValueError : If finite radii cannot be determined for binning.

#### Signature

```python
def get_particle_resolved_binned_radius(
    particle: ParticleRepresentation,
    bin_radius: Optional[NDArray[np.float64]] = None,
    total_bins: Optional[int] = None,
    bins_per_radius_decade: int = 10,
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## get_speciated_mass_representation_from_particle_resolved

[Show source in change_particle_representation.py:80](https://github.com/uncscode/particula/blob/main/particula/particles/change_particle_representation.py#L80)

Convert a ParticleResolvedSpeciatedMass to a SpeciatedMassMovingBin.

This function bins the mass and charge distributions for each species
according to the provided bin_radius array, using median or mean
values in each bin. The distribution_strategy is switched to
SpeciatedMassMovingBin.

#### Arguments

- particle : The ParticleRepresentation to convert.
- bin_radius : Array of radius bin edges in meters.

#### Returns

- ParticleRepresentation : A new representation with binned
  mass and concentration for each species.

#### Signature

```python
def get_speciated_mass_representation_from_particle_resolved(
    particle: ParticleRepresentation, bin_radius: NDArray[np.float64]
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)


---
# distribution_builders.md

# Distribution Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Distribution Builders

> Auto-generated documentation for [particula.particles.distribution_builders](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py) module.

## MassBasedMovingBinBuilder

[Show source in distribution_builders.py:16](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L16)

Builds and configures a MassBasedMovingBin instance for mass-based
distributions.

This builder requires no parameters, but is kept for consistency with
other builder patterns. Ensures a uniform interface for creating
MassBasedMovingBin objects.

#### Methods

- build : Return a MassBasedMovingBin instance.

#### Examples

```py title="Example"
import particula as par

builder = par.particles.MassBasedMovingBinBuilder()
strategy = builder.build()
# strategy -> MassBasedMovingBin()
```

#### Signature

```python
class MassBasedMovingBinBuilder(BuilderABC):
    def __init__(self) -> None: ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### MassBasedMovingBinBuilder().build

[Show source in distribution_builders.py:42](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L42)

Build and return a MassBasedMovingBin instance.

#### Returns

- MassBasedMovingBin : A strategy for mass-based particle
    distributions.

#### Examples

```py title="Build Example"
import particula as par
builder = par.particles.MassBasedMovingBinBuilder()
strategy = builder.build()
```

#### Signature

```python
def build(self) -> MassBasedMovingBin: ...
```

#### See also

- [MassBasedMovingBin](./distribution_strategies.md#massbasedmovingbin)



## ParticleResolvedSpeciatedMassBuilder

[Show source in distribution_builders.py:145](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L145)

Builds and configures a ParticleResolvedSpeciatedMass instance.

This builder requires no parameters, but follows the same pattern
to ensure uniform usage. ParticleResolvedSpeciatedMass is useful for
specific calculations when each particle's species composition must
be resolved individually.

#### Methods

- build : Return a ParticleResolvedSpeciatedMass instance.

#### Examples

```py title="Example"
import particula as par
builder = par.particles.ParticleResolvedSpeciatedMassBuilder()
strategy = builder.build()
# strategy -> ParticleResolvedSpeciatedMass()
```

#### Signature

```python
class ParticleResolvedSpeciatedMassBuilder(BuilderABC):
    def __init__(self) -> None: ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### ParticleResolvedSpeciatedMassBuilder().build

[Show source in distribution_builders.py:170](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L170)

Build and return a ParticleResolvedSpeciatedMass instance.

#### Returns

- ParticleResolvedSpeciatedMass : A strategy that resolves
  each particle's species composition independently.

#### Examples

```py title="Build Example"
import particula as par
builder = par.particles.ParticleResolvedSpeciatedMassBuilder()
strategy = builder.build()
```

#### Signature

```python
def build(self) -> ParticleResolvedSpeciatedMass: ...
```

#### See also

- [ParticleResolvedSpeciatedMass](./distribution_strategies.md#particleresolvedspeciatedmass)



## RadiiBasedMovingBinBuilder

[Show source in distribution_builders.py:60](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L60)

Builds and configures a RadiiBasedMovingBin instance for radius-based
distributions.

This builder requires no parameters, but is provided for consistency
with other builder patterns. Ensures a uniform interface for creating
RadiiBasedMovingBin objects.

#### Methods

- build : Return a RadiiBasedMovingBin instance.

#### Examples

```py title="Example"
import particula as par
builder = par.particles.RadiiBasedMovingBinBuilder()
strategy = builder.build()
# strategy -> RadiiBasedMovingBin()
```

#### Signature

```python
class RadiiBasedMovingBinBuilder(BuilderABC):
    def __init__(self) -> None: ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### RadiiBasedMovingBinBuilder().build

[Show source in distribution_builders.py:85](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L85)

Build and return a RadiiBasedMovingBin instance.

#### Returns

- RadiiBasedMovingBin : A strategy for radius-based particle
    distributions.

#### Examples

```py title="Build Example"
import particula as par
builder = par.particles.RadiiBasedMovingBinBuilder()
strategy = builder.build()
```

#### Signature

```python
def build(self) -> RadiiBasedMovingBin: ...
```

#### See also

- [RadiiBasedMovingBin](./distribution_strategies.md#radiibasedmovingbin)



## SpeciatedMassMovingBinBuilder

[Show source in distribution_builders.py:103](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L103)

Builds and configures a SpeciatedMassMovingBin instance for speciated
mass distributions.

This builder requires no parameters, but provides consistency with
other builder patterns and ensures a uniform interface for creating
SpeciatedMassMovingBin objects.

#### Methods

- build : Return a SpeciatedMassMovingBin instance.

#### Examples

```py title="Example"
import particula as par
builder = par.particles.SpeciatedMassMovingBinBuilder()
strategy = builder.build()
# strategy -> SpeciatedMassMovingBin()
```

#### Signature

```python
class SpeciatedMassMovingBinBuilder(BuilderABC):
    def __init__(self) -> None: ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)

### SpeciatedMassMovingBinBuilder().build

[Show source in distribution_builders.py:128](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L128)

Build and return a SpeciatedMassMovingBin instance.

#### Returns

- SpeciatedMassMovingBin : A strategy for speciated mass
    distributions.

#### Examples

```py title="Build Example"
builder = SpeciatedMassMovingBinBuilder()
strategy = builder.build()
```

#### Signature

```python
def build(self) -> SpeciatedMassMovingBin: ...
```

#### See also

- [SpeciatedMassMovingBin](./distribution_strategies.md#speciatedmassmovingbin)


---
# distribution_factories.md

# Distribution Factories

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Distribution Factories

> Auto-generated documentation for [particula.particles.distribution_factories](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_factories.py) module.

## DistributionFactory

[Show source in distribution_factories.py:19](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_factories.py#L19)

Factory class to create distribution strategies from builders.

This factory is used to obtain particle distribution strategies
based on the specified representation type (mass-based, radius-based,
speciated, or particle-resolved).

#### Methods

- get_builders : Return a mapping of strategy types to builder
    instances.
- get_strategy : Return a strategy instance for a given strategy type.

#### Returns

- DistributionStrategy : An instance configured for the chosen
  distribution representation.

#### Raises

- ValueError : If an unknown strategy type is provided or if
  required parameters are missing or invalid.

#### Examples

```py title="DistributionFactory Example"
import particula as par
factory = par.particles.DistributionFactory()
strategy = factory.get_strategy("mass_based_moving_bin")
# strategy -> MassBasedMovingBin()
```

#### Signature

```python
class DistributionFactory(
    StrategyFactoryABC[
        Union[
            MassBasedMovingBinBuilder,
            RadiiBasedMovingBinBuilder,
            SpeciatedMassMovingBinBuilder,
            ParticleResolvedSpeciatedMassBuilder,
        ],
        Union[
            MassBasedMovingBin,
            RadiiBasedMovingBin,
            SpeciatedMassMovingBin,
            ParticleResolvedSpeciatedMass,
        ],
    ]
): ...
```

#### See also

- [MassBasedMovingBinBuilder](./distribution_builders.md#massbasedmovingbinbuilder)
- [MassBasedMovingBin](./distribution_strategies.md#massbasedmovingbin)
- [ParticleResolvedSpeciatedMassBuilder](./distribution_builders.md#particleresolvedspeciatedmassbuilder)
- [ParticleResolvedSpeciatedMass](./distribution_strategies.md#particleresolvedspeciatedmass)
- [RadiiBasedMovingBinBuilder](./distribution_builders.md#radiibasedmovingbinbuilder)
- [RadiiBasedMovingBin](./distribution_strategies.md#radiibasedmovingbin)
- [SpeciatedMassMovingBinBuilder](./distribution_builders.md#speciatedmassmovingbinbuilder)
- [SpeciatedMassMovingBin](./distribution_strategies.md#speciatedmassmovingbin)

### DistributionFactory().get_builders

[Show source in distribution_factories.py:64](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_factories.py#L64)

Return a mapping of strategy types to builder instances.

#### Returns

- A dictionary where each key is a string identifying the strategy
    type, and each value is the corresponding builder object.

#### Examples

```py title="get_builders Example"
import particula as par
factory = par.particles.DistributionFactory()
builder_map = factory.get_builders()
# builder_map["mass_based_moving_bin"] -> MassBasedMovingBinBuilder
```

#### Signature

```python
def get_builders(self): ...
```


---
# distribution_strategies.md

# Distribution Strategies

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Distribution Strategies

> Auto-generated documentation for [particula.particles.distribution_strategies](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py) module.

## DistributionStrategy

[Show source in distribution_strategies.py:17](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L17)

Abstract base class defining common interfaces for
mass, radius, and total mass calculations across
different particle distribution representations.

#### Methods

- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.
- collide_pairs : Perform collision logic on specified particle pairs.

#### Signature

```python
class DistributionStrategy(ABC): ...
```

### DistributionStrategy().add_concentration

[Show source in distribution_strategies.py:132](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L132)

Add concentration to the distribution of particles.

#### Arguments

- distribution : The distribution of particle sizes or masses.
- concentration : The concentration of each particle
  size or mass.
- added_distribution : The distribution to be added.
- added_concentration : The concentration to be added.

#### Returns

- The updated distribution array
- The updated concentration array.

#### Signature

```python
@abstractmethod
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().add_mass

[Show source in distribution_strategies.py:109](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L109)

Add mass to the distribution of particles.

#### Arguments

- distribution : The distribution of particle sizes or masses.
- concentration : The concentration of each particle
  size or mass.
- density : The density of the particles.
- added_mass : The mass to be added per distribution bin.

#### Returns

- The updated distribution array.
- The updated concentration array.

#### Signature

```python
@abstractmethod
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().collide_pairs

[Show source in distribution_strategies.py:155](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L155)

Collide index pairs in the distribution.

#### Arguments

- distribution : The distribution of particle sizes or masses.
- concentration : The concentration of each particle size or mass.
- density : The density of the particles.
- indices : The indices of the particles to collide.

#### Returns

- The updated distribution array
- The updated concentration array.

#### Signature

```python
@abstractmethod
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().get_mass

[Show source in distribution_strategies.py:53](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L53)

Calculate the mass of the particles or bin.

#### Arguments

- distribution : The distribution of particle sizes or masses.
- density : The density of the particles.

#### Returns

- The mass of the particles.

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_name

[Show source in distribution_strategies.py:34](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L34)

Return the type of the distribution strategy.

#### Signature

```python
def get_name(self) -> str: ...
```

### DistributionStrategy().get_radius

[Show source in distribution_strategies.py:94](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L94)

Calculate the radius of the particles.

#### Arguments

- distribution : The distribution of particle sizes or masses.
- density : The density of the particles.

#### Returns

- The radius of the particles in meters.

#### Signature

```python
@abstractmethod
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_species_mass

[Show source in distribution_strategies.py:38](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L38)

Return the mass per species in the distribution.

#### Arguments

- distribution : The distribution of particle sizes or masses.
- density : The density of the particles.

#### Returns

- The mass of the particles (per species).

#### Signature

```python
@abstractmethod
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_total_mass

[Show source in distribution_strategies.py:72](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L72)

Calculate the total mass of all particles (or bin).

#### Arguments

- distribution : The distribution of particle sizes or masses.
- concentration : The concentration of each particle
  size or mass in the distribution.
- density : The density of the particles.

#### Returns

- The total mass of the particles.

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## MassBasedMovingBin

[Show source in distribution_strategies.py:178](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L178)

Strategy for particles represented by their mass distribution.

Calculates particle mass, radius, and total mass based on the
particle mass, number concentration, and density. This moving-bin
approach adjusts mass bins on mass addition events.

#### Methods

- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.

#### Signature

```python
class MassBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### MassBasedMovingBin().add_concentration

[Show source in distribution_strategies.py:220](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L220)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().add_mass

[Show source in distribution_strategies.py:210](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L210)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().collide_pairs

[Show source in distribution_strategies.py:256](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L256)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().get_radius

[Show source in distribution_strategies.py:202](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L202)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### MassBasedMovingBin().get_species_mass

[Show source in distribution_strategies.py:196](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L196)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## ParticleResolvedSpeciatedMass

[Show source in distribution_strategies.py:473](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L473)

Strategy for particle-resolved masses with multiple species.

Allows each particle to have separate masses for each species, with
individualized densities. This strategy provides a more detailed
approach when each particle's composition must be modeled explicitly.

#### Methods

- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.
- collide_pairs : Perform collision logic on specified particle pairs.

#### Signature

```python
class ParticleResolvedSpeciatedMass(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### ParticleResolvedSpeciatedMass().add_concentration

[Show source in distribution_strategies.py:535](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L535)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().add_mass

[Show source in distribution_strategies.py:507](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L507)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().collide_pairs

[Show source in distribution_strategies.py:593](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L593)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().get_radius

[Show source in distribution_strategies.py:497](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L497)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### ParticleResolvedSpeciatedMass().get_species_mass

[Show source in distribution_strategies.py:492](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L492)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## RadiiBasedMovingBin

[Show source in distribution_strategies.py:271](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L271)

Strategy for particles represented by their radius distribution.

Calculates particle mass, radius, and total mass based on particle
radius, number concentration, and density. This moving-bin approach
recalculates radii when mass is added.

#### Methods

- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.

#### Signature

```python
class RadiiBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### RadiiBasedMovingBin().add_concentration

[Show source in distribution_strategies.py:321](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L321)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().add_mass

[Show source in distribution_strategies.py:303](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L303)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().collide_pairs

[Show source in distribution_strategies.py:356](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L356)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().get_radius

[Show source in distribution_strategies.py:296](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L296)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### RadiiBasedMovingBin().get_species_mass

[Show source in distribution_strategies.py:289](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L289)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## SpeciatedMassMovingBin

[Show source in distribution_strategies.py:371](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L371)

Strategy for particles with speciated mass distribution.

Each particle may contain multiple species, each with a unique
density. This strategy calculates mass, radius, and total mass from
the species-level masses and overall particle concentrations.

#### Methods

- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.

#### Signature

```python
class SpeciatedMassMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### SpeciatedMassMovingBin().add_concentration

[Show source in distribution_strategies.py:423](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L423)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().add_mass

[Show source in distribution_strategies.py:401](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L401)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().collide_pairs

[Show source in distribution_strategies.py:458](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L458)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().get_radius

[Show source in distribution_strategies.py:394](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L394)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### SpeciatedMassMovingBin().get_species_mass

[Show source in distribution_strategies.py:389](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies.py#L389)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```


---
# activity_module.md

# Activity Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Activity Module

> Auto-generated documentation for [particula.particles.properties.activity_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py) module.

## get_ideal_activity_mass

[Show source in activity_module.py:111](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L111)

Compute the ideal activity based on mass fractions.

This function calculates the activity of a species using its mass fraction.
In an ideal mixture, the activity (aᵢ) can be expressed as:

- aᵢ = wᵢ
    - wᵢ is the mass fraction of species i.

#### Arguments

- mass_concentration : Mass concentration of the species in kg/m³.

#### Returns

- Ideal activity of the species as a dimensionless value.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_ideal_activity_mass(np.array([1.0, 2.0]))
# Output: array([...])
```

#### References

- Raoult's Law, "Raoult's law," Wikipedia,
  https://en.wikipedia.org/wiki/Raoult%27s_law.

#### Signature

```python
@validate_inputs({"mass_concentration": "nonnegative"})
def get_ideal_activity_mass(
    mass_concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_ideal_activity_molar

[Show source in activity_module.py:15](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L15)

Compute the ideal activity based on mole fractions.

This function calculates the activity of a species using its mole fraction,
which follows Raoult's Law. The ideal activity (aᵢ) is determined using:

- aᵢ = Xᵢ
    - Xᵢ is the mole fraction of species i.

#### Arguments

- mass_concentration : Mass concentration of the species in kg/m³.
- molar_mass : Molar mass of the species in kg/mol.

#### Returns

- Ideal activity of the species as a dimensionless value.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_ideal_activity_molar(
    mass_concentration=np.array([1.0, 2.0]),
    molar_mass=np.array([18.015, 28.97])
)
# Output: array([...])
```

#### References

- Raoult's Law, "Raoult's law," Wikipedia,
  https://en.wikipedia.org/wiki/Raoult%27s_law.

#### Signature

```python
@validate_inputs({"mass_concentration": "nonnegative", "molar_mass": "positive"})
def get_ideal_activity_molar(
    mass_concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_ideal_activity_volume

[Show source in activity_module.py:63](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L63)

Compute the ideal activity based on volume fractions.

This function calculates the activity of a species using its volume
fraction. In an ideal mixture, the activity (aᵢ) can be expressed as:

- aᵢ = φᵢ
    - φᵢ is the volume fraction of species i.

#### Arguments

- mass_concentration : Mass concentration of the species in kg/m³.
- density : Density of the species in kg/m³.

#### Returns

- Ideal activity of the species as a dimensionless value.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_ideal_activity_volume(
    mass_concentration=np.array([1.0, 2.0]),
    density=np.array([1000.0, 1200.0])
)
# Output: array([...])
```

#### References

- Raoult's Law, "Raoult's law," Wikipedia,
  https://en.wikipedia.org/wiki/Raoult%27s_law.

#### Signature

```python
@validate_inputs({"mass_concentration": "nonnegative", "density": "positive"})
def get_ideal_activity_volume(
    mass_concentration: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_kappa_activity

[Show source in activity_module.py:152](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L152)

Compute species activity using the κ (kappa) hygroscopic growth parameter.

This function calculates the activity of a mixture by combining
volume-fraction weighted κ-values. The water activity (aₘ) is
determined by:

- aₘ = 1 / (1 + κₑ ( Vₛ / Vₐ ))
    - κₑ is the volume-fraction weighted hygroscopic parameter.
    - Vₛ is the total solute volume fraction (all species except water).
    - Vₐ is the water volume fraction.

#### Arguments

- mass_concentration : Array of mass concentrations in kg/m³.
- kappa : Array of κ (kappa) hygroscopic parameters, dimensionless.
- density : Array of densities in kg/m³ for each species.
- molar_mass : Array of molar masses in kg/mol for each species.
- water_index : Index of the water component in the arrays.

#### Returns

- Array of species activities, dimensionless.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_kappa_activity(
    mass_concentration=np.array([[1.0, 2.0], [3.0, 4.0]]),
    kappa=np.array([0.0, 0.2]),
    density=np.array([1000.0, 1200.0]),
    molar_mass=np.array([18.015, 28.97]),
    water_index=0
)
# Output: array([...])
```

#### References

- Petters, M. D., & Kreidenweis, S. M. (2007). "A single parameter
  representation of hygroscopic growth and cloud condensation nucleus
  activity," Atmospheric Chemistry and Physics, 7(8), 1961-1971.
  - `DOI` - https://doi.org/10.5194/acp-7-1961-2007.

#### Signature

```python
@validate_inputs(
    {
        "mass_concentration": "nonnegative",
        "kappa": "nonnegative",
        "density": "positive",
        "molar_mass": "positive",
    }
)
def get_kappa_activity(
    mass_concentration: NDArray[np.float64],
    kappa: NDArray[np.float64],
    density: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    water_index: int,
) -> NDArray[np.float64]: ...
```



## get_surface_partial_pressure

[Show source in activity_module.py:265](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L265)

Compute the partial pressure from activity and pure vapor pressure.

This function calculates the partial pressure (pᵢ) of a species, given its
activity (aᵢ) and pure vapor pressure (pᵢ*). It follows:

- pᵢ = aᵢ × pᵢ*

#### Arguments

- pure_vapor_pressure : Pure vapor pressure of the species in Pa.
- activity : Activity of the species, dimensionless.

#### Returns

- Partial pressure of the species in Pa.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_surface_partial_pressure(1000.0, 0.95)
# Output: 950.0
```

#### Signature

```python
@validate_inputs({"pure_vapor_pressure": "positive", "activity": "nonnegative"})
def get_surface_partial_pressure(
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
    activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# aerodynamic_mobility_module.md

# Aerodynamic Mobility Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Aerodynamic Mobility Module

> Auto-generated documentation for [particula.particles.properties.aerodynamic_mobility_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_mobility_module.py) module.

## get_aerodynamic_mobility

[Show source in aerodynamic_mobility_module.py:13](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_mobility_module.py#L13)

Calculate the aerodynamic mobility of a particle using classical fluid
mechanics.

The aerodynamic mobility (B) can be determined by:

- B = C / (6πμr)
    - B is the aerodynamic mobility (m²/s).
    - C is the slip correction factor (dimensionless).
    - μ is the dynamic viscosity of the fluid (Pa·s).
    - r is the radius of the particle (m).

#### Arguments

- particle_radius : The radius of the particle in meters.
- slip_correction_factor : Slip correction factor (dimensionless).
- dynamic_viscosity : Dynamic viscosity of the fluid in Pa·s.

#### Returns

- The particle aerodynamic mobility in m²/s.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_particle_aerodynamic_mobility(
    particle_radius=0.00005,
    slip_correction_factor=1.1,
    dynamic_viscosity=0.0000181
)

References:
- Wikipedia contributors, "Stokes' Law," Wikipedia,
https://en.wikipedia.org/wiki/Stokes%27_law.

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "positive",
        "slip_correction_factor": "nonnegative",
        "dynamic_viscosity": "positive",
    }
)
def get_aerodynamic_mobility(
    particle_radius: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# aerodynamic_size.md

# Aerodynamic Size

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Aerodynamic Size

> Auto-generated documentation for [particula.particles.properties.aerodynamic_size](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_size.py) module.

## get_aerodynamic_length

[Show source in aerodynamic_size.py:24](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_size.py#L24)

Calculate the aerodynamic length scale of a particle for a given shape.

The aerodynamic length (d_a) is determined by:

- d_a = d_p × √( (C_p / C_a) × (ρ / (ρ₀ × χ)) )
    - d_a is the aerodynamic size (m).
    - d_p is the physical size (m).
    - C_p is the slip correction factor for the physical size.
    - C_a is the slip correction factor for the aerodynamic size.
    - ρ is the particle density (kg/m³).
    - ρ₀ is the reference density (kg/m³).
    - χ is the shape factor (dimensionless).

#### Arguments

- physical_length : Physical length scale of the particle (m).
- physical_slip_correction_factor : Slip correction factor for the
    particle's physical size (dimensionless).
- aerodynamic_slip_correction_factor : Slip correction factor for the
    particle's aerodynamic size (dimensionless).
- density : Density of the particle in kg/m³.
- reference_density : Reference density in kg/m³, typically water
    (1000 by default).
- aerodynamic_shape_factor : Shape factor
    (dimensionless, 1.0 for spheres).

#### Returns

- Aerodynamic length scale (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_aerodynamic_length(
    physical_length=0.00005,
    physical_slip_correction_factor=1.1,
    aerodynamic_slip_correction_factor=1.0,
    density=1000,
    reference_density=1000,
    aerodynamic_shape_factor=1.0,
)
# Output: ...
```

#### References

- `-` *"Aerosol* - Aerodynamic diameter," Wikipedia,
  https://en.wikipedia.org/wiki/Aerosol#Aerodynamic_diameter
- Hinds, W.C. (1998). Aerosol Technology: Properties, behavior, and
  measurement of airborne particles (2nd ed.). Wiley-Interscience.
  (pp. 51–53, Section 3.6).

#### Signature

```python
@validate_inputs(
    {
        "physical_length": "nonnegative",
        "physical_slip_correction_factor": "nonnegative",
        "aerodynamic_slip_correction_factor": "nonnegative",
        "density": "positive",
    }
)
def get_aerodynamic_length(
    physical_length: Union[float, NDArray[np.float64]],
    physical_slip_correction_factor: Union[float, NDArray[np.float64]],
    aerodynamic_slip_correction_factor: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    reference_density: float = 1000,
    aerodynamic_shape_factor: float = 1.0,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_aerodynamic_shape_factor

[Show source in aerodynamic_size.py:97](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_size.py#L97)

Retrieve the aerodynamic shape factor for a given particle shape.

The shape factor (χ) accounts for non-sphericity in aerodynamic
calculations. For spheres, χ=1.0. Larger values indicate more deviation
from spherical shape.

#### Arguments

- shape_key : String representing the particle's shape
    (e.g. "sphere", "sand").

#### Returns

- The shape factor (dimensionless).

#### Examples

``` py title="Example"
shape_factor = get_aerodynamic_shape_factor("sand")
# shape_factor = 1.57
```

#### Raises

- ValueError : If the shape key is not found in the predefined
    dictionary.

#### References

- Hinds, W.C. (1998). Aerosol Technology: Properties, behavior, and
  measurement of airborne particles (2nd ed.). Wiley-Interscience.

#### Signature

```python
def get_aerodynamic_shape_factor(shape_key: str) -> float: ...
```


---
# collision_radius_module.md

# Collision Radius Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Collision Radius Module

> Auto-generated documentation for [particula.particles.properties.collision_radius_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py) module.

## get_collision_radius_mg1988

[Show source in collision_radius_module.py:17](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L17)

Calculate the collision radius using the mg1988 model.

The collision radius (R_c) is set equal to the radius of gyration (R_g):

- R_c = R_g

#### Arguments

- gyration_radius : Radius of gyration of the particle (m).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_mg1988(1.5)
# 1.5
```

#### References

- Mulholland, G. W., Mountain, R. D., Samson, R. J., & Ernst, M. H.
(1988). "Cluster Size Distribution for Free Molecular Agglomeration."
  Energy and Fuels, 2(4). https://doi.org/10.1021/ef00010a014

#### Signature

```python
@validate_inputs({"gyration_radius": "nonnegative"})
def get_collision_radius_mg1988(
    gyration_radius: Union[NDArray[np.float64], float],
) -> Union[NDArray[np.float64], float]: ...
```



## get_collision_radius_mzg2002

[Show source in collision_radius_module.py:98](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L98)

Calculate the collision radius using the mzg2002 model.

The collision radius (R_c) is given by the empirical relation:

- R_c = 1.037 × (k₀^0.077) × R_g
    - R_c is the collision radius (m).
    - k₀ is the fractal prefactor (dimensionless).
    - R_g is the radius of gyration (m).

#### Arguments

- gyration_radius : Radius of gyration of the particle (m).
- fractal_prefactor : Fractal prefactor of particle (dimensionless).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_mzg2002(1.5, 1.2)
# 1.577...
```

#### References

- Zurita-Gotor, M., & Rosner, D. E. (2002). "Effective diameters for
  collisions of fractal-like aggregates: Recommendations for improved
  aerosol coagulation frequency predictions." Journal of Colloid and
  Interface Science, 255(1).
  https://doi.org/10.1006/jcis.2002.8634

#### Signature

```python
@validate_inputs({"gyration_radius": "positive", "fractal_prefactor": "positive"})
def get_collision_radius_mzg2002(
    gyration_radius: Union[NDArray[np.float64], float],
    fractal_prefactor: Union[NDArray[np.float64], float],
) -> Union[NDArray[np.float64], float]: ...
```



## get_collision_radius_sr1992

[Show source in collision_radius_module.py:53](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L53)

Calculate the collision radius using the sr1992 model.

This model includes the fractal dimension (d_f). The collision radius
(R_c) is:

- R_c = √((d_f + 2) / 3) × R_g
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - R_g is the radius of gyration (m).

#### Arguments

- gyration_radius : Radius of gyration of the particle (m).
- fractal_dimension : Fractal dimension of the particle
    (dimensionless).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_sr1992(1.5, 1.2)
# 1.8371173...
```

#### References

- Rogak, S. N., & Flagan, R. C. (1992). "Coagulation of aerosol
  agglomerates in the transition regime." Journal of Colloid and
  Interface Science, 151(1), 203-224.
  https://doi.org/10.1016/0021-9797(92)90252-H

#### Signature

```python
@validate_inputs({"gyration_radius": "positive", "fractal_dimension": "positive"})
def get_collision_radius_sr1992(
    gyration_radius: Union[NDArray[np.float64], float],
    fractal_dimension: Union[NDArray[np.float64], float],
) -> Union[NDArray[np.float64], float]: ...
```



## get_collision_radius_tt2012

[Show source in collision_radius_module.py:142](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L142)

Calculate the collision radius using the tt2012 model.

This function uses fitting parameters α₁, α₂ based on the fractal
dimension (d_f) and number of monomers (N). The collision radius
(R_c) is derived in multiple steps, ultimately returning:

- R_c = (radius_s_ii) / 2

#### Arguments

- fractal_dimension : Fractal dimension of the particle (dimensionless).
- number_of_particles : Number of monomers in the aggregate.
- gyration_radius : Radius of gyration of the particle (m).
- radius_monomer : Radius of the monomer (m).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_tt2012(2.5, 100, 1.5, 0.1)
# 2.034...
```

#### References

- Thajudeen, T., Gopalakrishnan, R., & Hogan, C. J. (2012). "The
  collision rate of nonspherical particles and aggregates for all
  diffusive knudsen numbers." Aerosol Science and Technology, 46(11).
  https://doi.org/10.1080/02786826.2012.701353

#### Signature

```python
@validate_inputs(
    {
        "fractal_dimension": "positive",
        "number_of_particles": "positive",
        "gyration_radius": "positive",
        "radius_monomer": "positive",
    }
)
def get_collision_radius_tt2012(
    fractal_dimension: float,
    number_of_particles: float,
    gyration_radius: Union[NDArray[np.float64], float],
    radius_monomer: float,
) -> Union[NDArray[np.float64], float]: ...
```



## get_collision_radius_wq2022_rg

[Show source in collision_radius_module.py:199](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L199)

Calculate the collision radius using the wq2022_rg model.

This function uses a fitted model based on the ratio (R_g / rₘ).
The collision radius (R_c) is:

- R_c = (A × (R_g / rₘ) + B) × rₘ
    - R_c is the collision radius (m).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B are empirical coefficients from Qian et al. (2022).

#### Arguments

- gyration_radius : Radius of gyration of the particle (m).
- radius_monomer : Monomer radius (m).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg(1.5, 0.1)
# 1.50...
```

#### References

- Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
  "Effects of agglomerate characteristics on their collision kernels in
  the free molecular regime." Journal of Aerosol Science, 159.
  https://doi.org/10.1016/j.jaerosci.2021.105868

#### Signature

```python
@validate_inputs({"gyration_radius": "positive", "radius_monomer": "positive"})
def get_collision_radius_wq2022_rg(
    gyration_radius: Union[NDArray[np.float64], float], radius_monomer: float
) -> Union[NDArray[np.float64], float]: ...
```



## get_collision_radius_wq2022_rg_df

[Show source in collision_radius_module.py:247](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L247)

Calculate the collision radius using the wq2022_rg_df model.

This function uses a fitted model based on fractal dimension (d_f), ratio
(R_g / rₘ), and empirical coefficients. The collision radius (R_c) is:

- R_c = (A × d_f^B × (R_g / rₘ) + C) × rₘ
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B, C are empirical coefficients from Qian et al. (2022).

#### Arguments

- fractal_dimension : Fractal dimension of particle (dimensionless).
- gyration_radius : Radius of gyration of the particle (m).
- radius_monomer : Monomer radius (m).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg_df(2.5, 1.5, 0.1)
# 1.66...
```

#### References

- Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
  "Effects of agglomerate characteristics on their collision kernels in
  the free molecular regime." Journal of Aerosol Science, 159.
  https://doi.org/10.1016/j.jaerosci.2021.105868

#### Signature

```python
@validate_inputs(
    {
        "fractal_dimension": "positive",
        "gyration_radius": "positive",
        "radius_monomer": "positive",
    }
)
def get_collision_radius_wq2022_rg_df(
    fractal_dimension: Union[NDArray[np.float64], float],
    gyration_radius: Union[NDArray[np.float64], float],
    radius_monomer: float,
) -> Union[NDArray[np.float64], float]: ...
```



## get_collision_radius_wq2022_rg_df_k0

[Show source in collision_radius_module.py:302](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L302)

Calculate the collision radius using the wq2022_rg_df_k0 model.

This function uses a fitted expression depending on fractal dimension
(d_f), fractal prefactor (k₀), and ratio (R_g / rₘ). The collision
radius (R_c) is:

- R_c = (A × d_f^B × k₀^C × (R_g / rₘ) + D × k₀ + E) × rₘ
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - k₀ is the fractal prefactor (dimensionless).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B, C, D, E are empirical coefficients from Qian et al. (2022).

#### Arguments

- fractal_dimension : Fractal dimension of particle (dimensionless).
- fractal_prefactor : Fractal prefactor of particle (dimensionless).
- gyration_radius : Radius of gyration (m).
- radius_monomer : Monomer radius (m).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg_df_k0(2.5, 1.2, 1.5, 0.1)
# 1.83...
```

#### References

- Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
  "Effects of agglomerate characteristics on their collision kernels in
  the free molecular regime." Journal of Aerosol Science, 159.
  https://doi.org/10.1016/j.jaerosci.2021.105868

#### Signature

```python
@validate_inputs(
    {
        "fractal_dimension": "positive",
        "fractal_prefactor": "positive",
        "gyration_radius": "positive",
        "radius_monomer": "positive",
    }
)
def get_collision_radius_wq2022_rg_df_k0(
    fractal_dimension: float,
    fractal_prefactor: float,
    gyration_radius: Union[NDArray[np.float64], float],
    radius_monomer: float,
) -> Union[NDArray[np.float64], float]: ...
```



## get_collision_radius_wq2022_rg_df_k0_a13

[Show source in collision_radius_module.py:364](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L364)

Calculate the collision radius using the wq2022_rg_df_k0_a13 model.

This function uses a fitted expression depending on fractal dimension
(d_f), fractal prefactor (k₀), shape anisotropy (A₁₃), and ratio
(R_g / rₘ). The collision radius (R_c) is:

- R_c = (A × d_f^B × k₀^C × (R_g / rₘ) + D × k₀ + E × A₁₃ + F) × rₘ
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - k₀ is the fractal prefactor (dimensionless).
    - A₁₃ is the shape anisotropy parameter (dimensionless).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B, C, D, E, F are empirical coefficients from Qian et al. (2022).

#### Arguments

- fractal_dimension : Fractal dimension of particle (dimensionless).
- fractal_prefactor : Fractal prefactor of particle (dimensionless).
- shape_anisotropy : Shape anisotropy parameter (dimensionless, A₁₃).
- gyration_radius : Radius of gyration (m).
- radius_monomer : Monomer radius (m).

#### Returns

- Collision radius of the particle (m).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg_df_k0_a13(
    2.5, 1.2, 1.82, 1.5, 0.1
)
# 1.82...
```

#### References

- Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
  "Effects of agglomerate characteristics on their collision kernels in
  the free molecular regime." Journal of Aerosol Science, 159.
  https://doi.org/10.1016/j.jaerosci.2021.105868

#### Signature

```python
@validate_inputs(
    {
        "fractal_dimension": "positive",
        "fractal_prefactor": "positive",
        "shape_anisotropy": "positive",
        "gyration_radius": "positive",
        "radius_monomer": "positive",
    }
)
def get_collision_radius_wq2022_rg_df_k0_a13(
    fractal_dimension: float,
    fractal_prefactor: float,
    shape_anisotropy: float,
    gyration_radius: Union[NDArray[np.float64], float],
    radius_monomer: float,
) -> Union[NDArray[np.float64], float]: ...
```


---
# convert_kappa_volumes.md

# Convert Kappa Volumes

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Convert Kappa Volumes

> Auto-generated documentation for [particula.particles.properties.convert_kappa_volumes](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_kappa_volumes.py) module.

## get_kappa_from_volumes

[Show source in convert_kappa_volumes.py:108](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_kappa_volumes.py#L108)

Compute the κ parameter from known volumes of solute and water, given
water activity.

Rearranging κ-Köhler-based relationships, we have:
- κ = ( (1/aw) - 1 ) × (V_water / V_solute).

#### Arguments

- volume_solute : Solute volume (float or NDArray).
- volume_water : Water volume (float or NDArray).
- water_activity : Water activity (float or NDArray, 0 < aw ≤ 1).

#### Returns

- The kappa parameter (float or NDArray).

#### Examples

``` py
import particula as par
kappa_val = par.get_kappa_from_volumes(1e-19, 4e-19, 0.95)
print(kappa_val)
# ~indicative value for the solute's hygroscopicity
```

#### References

- Petters, M. D. & Kreidenweis, S. M. (2007). "A single parameter
  representation of hygroscopic growth and cloud condensation nucleus
  activity." Atmos. Chem. Phys.

#### Signature

```python
def get_kappa_from_volumes(
    volume_solute: Union[float, np.ndarray],
    volume_water: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]: ...
```



## get_solute_volume_from_kappa

[Show source in convert_kappa_volumes.py:19](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_kappa_volumes.py#L19)

Calculate the solute volume from the total solution volume using κ-Köhler
theory.

The relation for κ-Köhler can be written as:
- V_solute = V_total × F
  where F depends on kappa and water activity (aw), ensuring that
  for aw → 0, V_solute → V_total.

#### Arguments

- volume_total : Volume of the total solution (float or NDArray).
- kappa : Kappa parameter (float or NDArray).
- water_activity : Water activity (float or NDArray, 0 < aw ≤ 1).

#### Returns

- Solute volume (float or NDArray).

#### Examples

``` py  title="Example Usage"
import particula as par
v_sol = par.get_solute_from_kappa_volume(1e-18, 0.8, 0.9)
print(v_sol)
# ~some fraction of the total volume
```

#### References

- Petters, M. D. & Kreidenweis, S. M. (2007). "A single parameter
  representation of hygroscopic growth and cloud condensation nucleus
  activity." Atmos. Chem. Phys.

#### Signature

```python
def get_solute_volume_from_kappa(
    volume_total: Union[float, np.ndarray],
    kappa: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]: ...
```



## get_water_volume_from_kappa

[Show source in convert_kappa_volumes.py:65](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_kappa_volumes.py#L65)

Calculate the water volume from the solute volume, κ parameter, and water
activity.

This uses κ-Köhler-type relations where:
- V_water = V_solute × ( κ / (1/aw - 1) ), ensuring that for aw → 0,
  V_water → 0.

#### Arguments

- volume_solute : Volume of solute (float or NDArray).
- kappa : Kappa parameter (float or NDArray).
- water_activity : Water activity (float or NDArray, 0 < aw ≤ 1).

#### Returns

- Water volume (float or NDArray).

#### Examples

``` py title="Example Usage"
import particula as par
v_water = par.get_water_volume_from_kappa(1e-19, 0.5, 0.95)
print(v_water)
# ~some fraction of the solute volume
```

#### References

- Petters, M. D. & Kreidenweis, S. M. (2007). "A single parameter
  representation of hygroscopic growth and cloud condensation nucleus
  activity." Atmos. Chem. Phys.

#### Signature

```python
def get_water_volume_from_kappa(
    volume_solute: Union[float, NDArray[np.float64]],
    kappa: Union[float, NDArray[np.float64]],
    water_activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_water_volume_in_mixture

[Show source in convert_kappa_volumes.py:149](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_kappa_volumes.py#L149)

Calculate the water volume in a solute-water mixture from a specified water
volume fraction.

The relationship is:

- V_water = (φ_water × V_solute_dry) / (1 - φ_water)
    - φ_water is the water volume fraction.

#### Arguments

- volume_solute_dry : Volume of the solute (float), excluding water.
- volume_fraction_water : Fraction of water volume in the total mixture
  (float, 0 ≤ φ_water < 1).

#### Returns

- The water volume (float), in the same units as volume_solute_dry.

#### Examples

``` py title="Example Usage"
import particula as par
v_water = par.get_water_volume_in_mixture(100.0, 0.8)
print(v_water)
# 400.0
```

#### References

- "Volume Fractions in Mixture Calculations," Standard Chemistry Texts.

#### Signature

```python
def get_water_volume_in_mixture(
    volume_solute_dry: Union[float, np.ndarray],
    volume_fraction_water: Union[float, np.ndarray],
) -> float: ...
```


---
# convert_mass_concentration.md

# Convert Mass Concentration

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Convert Mass Concentration

> Auto-generated documentation for [particula.particles.properties.convert_mass_concentration](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py) module.

## get_mass_fraction_from_mass

[Show source in convert_mass_concentration.py:165](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L165)

Convert mass concentrations to mass fractions for N components.

The mass fraction is computed by:

- wᵢ = mᵢ / mₜₒₜₐₗ
    - wᵢ is the mass fraction of component i (unitless),
    - mᵢ is the mass concentration of component i (kg/m³),
    - mₜₒₜₐₗ is the total mass concentration of all components (kg/m³).

#### Arguments

- mass_concentrations : Mass concentrations (kg/m³). Can be 1D or 2D.

#### Returns

- Mass fractions (unitless). Rows sum to 1 if input is 2D; returns 1D
  mass fractions if input is 1D.

#### Examples

```py
import numpy as np
import particula as par

mass_conc = np.array([10.0, 30.0, 60.0])  # kg/m³
par.get_mass_fraction(mass_conc)
# Output might be array([0.1, 0.3, 0.6])
```

#### References

- Wikipedia contributors, "Mass fraction (chemistry)," Wikipedia,
  https://en.wikipedia.org/wiki/Mass_fraction_(chemistry).

#### Signature

```python
@validate_inputs({"mass_concentrations": "nonnegative"})
def get_mass_fraction_from_mass(
    mass_concentrations: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_mole_fraction_from_mass

[Show source in convert_mass_concentration.py:9](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L9)

Convert mass concentrations to mole fractions for N components.

The mole fraction is computed using:

- xᵢ = (mᵢ / Mᵢ) / Σⱼ(mⱼ / Mⱼ)
    - xᵢ is the mole fraction of component i,
    - mᵢ is the mass concentration of component i (kg/m³),
    - Mᵢ is the molar mass of component i (kg/mol).

#### Arguments

- mass_concentrations : Mass concentrations (kg/m³). Can be 1D or 2D.
- molar_masses : Molar masses (kg/mol). Must match dimensions of
  mass_concentrations.

#### Returns

- Mole fractions (unitless). Rows sum to 1 if input is 2D; returns 1D
  mole fractions if input is 1D.

#### Examples

```py
import numpy as np
import particula as par
mass_conc = np.array([0.2, 0.8])  # kg/m³
mol_masses = np.array([0.018, 0.032])  # kg/mol
get_mole_fraction_from_mass(mass_conc, mol_masses))
# Output might be array([0.379..., 0.620...])
```

#### References

- Wikipedia contributors, "Mole fraction," Wikipedia,
  https://en.wikipedia.org/wiki/Mole_fraction.

#### Signature

```python
@validate_inputs({"mass_concentrations": "nonnegative", "molar_masses": "positive"})
def get_mole_fraction_from_mass(
    mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## get_volume_fraction_from_mass

[Show source in convert_mass_concentration.py:84](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L84)

Convert mass concentrations to volume fractions for N components.

The volume fraction is determined by:

- ϕᵢ = vᵢ / vₜₒₜₐₗ
    - ϕᵢ is the volume fraction of component i (unitless),
    - vᵢ is the volume of component i (m³),
    - vₜₒₜₐₗ is the total volume of all components (m³).

Volumes computed from mass concentration (mᵢ) and density (ρᵢ) using:
- vᵢ = mᵢ / ρᵢ.

#### Arguments

- mass_concentrations : Mass concentrations (kg/m³). Can be 1D or 2D.
- densities : Densities (kg/m³). Must match the shape of
  mass_concentrations.

#### Returns

- Volume fractions (unitless). Rows sum to 1 if input is 2D; returns 1D
  volume fractions if input is 1D.

#### Examples

```py
import numpy as np
import particula as par

mass_conc = np.array([[1.0, 2.0], [0.5, 0.5]])  # kg/m³
dens = np.array([1000.0, 800.0])               # kg/m³
par.get_volume_fraction_from_mass(mass_conc, dens))
# Output:
# array([[0.444..., 0.555...],
#        [0.5     , 0.5     ]])
```

#### References

- Wikipedia contributors, "Volume fraction," Wikipedia,
  https://en.wikipedia.org/wiki/Volume_fraction.

#### Signature

```python
@validate_inputs({"mass_concentrations": "nonnegative", "densities": "positive"})
def get_volume_fraction_from_mass(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```


---
# convert_mole_fraction.md

# Convert Mole Fraction

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Convert Mole Fraction

> Auto-generated documentation for [particula.particles.properties.convert_mole_fraction](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mole_fraction.py) module.

## get_mass_fractions_from_moles

[Show source in convert_mole_fraction.py:11](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mole_fraction.py#L11)

Convert mole fractions to mass fractions for N components.

The relationship between mass fraction (wᵢ) and mole fraction (xᵢ) is:

- wᵢ = (xᵢ × Mᵢ) / Σⱼ(xⱼ × Mⱼ)
    - wᵢ is the mass fraction of component i (unitless),
    - xᵢ is the mole fraction of component i (unitless),
    - Mᵢ is the molecular weight of component i (kg/mol),
    - Σⱼ(xⱼ × Mⱼ) is the total mass (per total moles).

#### Arguments

mole_fractions : Mole fractions (unitless). Can be 1D or 2D.
    If 2D, each row is treated as a set of mole fractions for N
    components.
molecular_weights : Molecular weights (kg/mol). Must match the shape of
    ``mole_fractions`` in the last dimension.

#### Returns

- Mass fractions (unitless). Rows sum to 1 if input is 2D; returns 1D
  mass fractions if input is 1D.

#### Examples

``` py title="Example 1: 1D"
import numpy as np
import particula as par
x_1d = np.array([0.2, 0.5, 0.3])    # mole fractions
mw_1d = np.array([18.0, 44.0, 28.0])  # molecular weights
par.get_mass_fractions_from_moles(x_1d, mw_1d)
# Output: ([0.379..., 0.620..., 0.0])
```

``` py title="Example 2: 2D"
import numpy as np
import particula as par
x_2d = np.array([
    [0.2, 0.5, 0.3],
    [0.3, 0.3, 0.4]
])
mw_2d = np.array([18.0, 44.0, 28.0])
par.get_mass_fractions_from_moles(x_2d, mw_2d)
```

#### References

- Wikipedia contributors, "Mass fraction (chemistry)," Wikipedia,
  https://en.wikipedia.org/wiki/Mass_fraction_(chemistry).

#### Signature

```python
@validate_inputs({"mole_fractions": "nonnegative", "molecular_weights": "positive"})
def get_mass_fractions_from_moles(
    mole_fractions: NDArray[np.float64], molecular_weights: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```


---
# convert_size_distribution.md

# Convert Size Distribution

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Convert Size Distribution

> Auto-generated documentation for [particula.particles.properties.convert_size_distribution](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py) module.

## ConversionStrategy

[Show source in convert_size_distribution.py:39](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L39)

#### Methods

    - `-` *convert* - Convert distribution data between input and output scales.
Defines an interface for conversion strategies between particle size
distribution formats.

All subclasses must implement the convert method to perform the actual
conversion logic.

#### Examples

``` py title="Subclass Example"
class CustomStrategy(ConversionStrategy):
    def convert(self, diameters, concentration, inverse=False):
        # Custom conversion logic here
        return concentration
```

#### Signature

```python
class ConversionStrategy: ...
```

### ConversionStrategy().convert

[Show source in convert_size_distribution.py:58](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L58)

Convert distribution data from one scale to another.

#### Arguments

- diameters : Array of particle diameters.
- concentration : The distribution data corresponding to these
    diameters.
- inverse : If True, reverse the direction of the conversion.

#### Returns

- np.ndarray of converted distribution data.

#### Raises

- NotImplementedError : If not overridden by a subclass.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## DNdlogDPtoPDFConversionStrategy

[Show source in convert_size_distribution.py:189](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L189)

Conversion strategy for converting between dn/dlogdp and PDF formats,
possibly through an intermediate PMF conversion.

#### Examples

``` py title="Example Usage"
strategy = DNdlogDPtoPDFConversionStrategy()
result_pdf = strategy.convert(diameters, dn_dlogdp_data)
# result_pdf is now in PDF format
```

#### Signature

```python
class DNdlogDPtoPDFConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### DNdlogDPtoPDFConversionStrategy().convert

[Show source in convert_size_distribution.py:202](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L202)

Convert between dn/dlogdp and PDF formats through an intermediate
PMF step.

#### Arguments

- diameters : Array of particle diameters.
- concentration : Distribution data in dn/dlogdp or PDF format.
- inverse : If True, convert from PDF to dn/dlogdp; otherwise the
    opposite.

#### Returns

- np.ndarray of the distribution in the target format.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## DNdlogDPtoPMFConversionStrategy

[Show source in convert_size_distribution.py:119](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L119)

Conversion strategy for converting between dn/dlogdp and PMF formats.

#### Examples

``` py title="Example Usage"
strategy = DNdlogDPtoPMFConversionStrategy()
result = strategy.convert(diameters, dn_dlogdp_conc)
# result is now in PMF format
```

#### Signature

```python
class DNdlogDPtoPMFConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### DNdlogDPtoPMFConversionStrategy().convert

[Show source in convert_size_distribution.py:131](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L131)

Perform the conversion between dn/dlogdp and PMF formats.

#### Arguments

- diameters : Array of particle diameters.
- concentration : Distribution data in dn/dlogdp or PMF format.
- inverse : If True, convert from PMF to dn/dlogdp; otherwise the
    opposite.

#### Returns

- np.ndarray of the distribution in the target format.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## PMFtoPDFConversionStrategy

[Show source in convert_size_distribution.py:154](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L154)

Conversion strategy for converting between PMF and PDF formats.

#### Examples

``` py title="Example Usage"
strategy = PMFtoPDFConversionStrategy()
result_pdf = strategy.convert(diameters, PMF_data, inverse=False)
# result_pdf is now in PDF format
```

#### Signature

```python
class PMFtoPDFConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### PMFtoPDFConversionStrategy().convert

[Show source in convert_size_distribution.py:166](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L166)

Perform the conversion between PMF and PDF formats.

#### Arguments

- diameters : Array of particle diameters.
- concentration : Distribution data in PMF or PDF format.
- inverse : If True, convert from PDF to PMF; otherwise from PMF
    to PDF.

#### Returns

- np.ndarray of the distribution in the target format.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## SameScaleConversionStrategy

[Show source in convert_size_distribution.py:84](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L84)

Conversion strategy that returns the input concentration unchanged.

No conversion is performed because the input and output scales are the
same.

#### Examples

```py title="Example Usage"
strategy = SameScaleConversionStrategy()
result = strategy.convert(diameters, concentration)
# result is identical to concentration
```

#### Signature

```python
class SameScaleConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### SameScaleConversionStrategy().convert

[Show source in convert_size_distribution.py:99](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L99)

Return the concentration unchanged, since no conversion is needed.

#### Arguments

- diameters : Array of particle diameters (unused).
- concentration : The original distribution data.
- inverse : Flag indicating direction (unused).

#### Returns

- np.ndarray identical to the input concentration.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## SizerConverter

[Show source in convert_size_distribution.py:237](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L237)

A converter that composes a ConversionStrategy to transform
particle size distribution data between formats.
[might not be needed or used, -kyle]

#### Examples

``` py title="Example Usage"
diameters = [1e-7, 1e-6, 1e-5]
concentration = [1e6, 1e5, 1e4]

strategy = DNdlogDPtoPMFConversionStrategy()
converter = SizerConverter(strategy)
new_conc = converter.convert(diameters, concentration)
```

#### Signature

```python
class SizerConverter:
    def __init__(self, strategy: ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### SizerConverter().convert

[Show source in convert_size_distribution.py:262](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L262)

Convert the particle size distribution data using the assigned
strategy.

#### Arguments

- diameters : Array of particle diameters.
- concentration : Distribution data.
- inverse : If True, reverse the conversion direction
    (if supported).

#### Returns

- np.ndarray of the converted distribution.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## get_distribution_conversion_strategy

[Show source in convert_size_distribution.py:284](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L284)

Factory function to obtain a conversion strategy based on the input and
output scales.

#### Arguments

- input_scale : Scale of the input distribution, e.g.
    'dn/dlogdp' or 'pmf'.
- output_scale : Desired scale of the output distribution, e.g.
    'pmf' or 'pdf'.

#### Returns

- A ConversionStrategy object supporting the requested conversion.

#### Raises

- ValueError : If scales are invalid or unsupported.

#### Examples

``` py title="Example Usage"
strategy = get_distribution_conversion_strategy('dn/dlogdp', 'pdf')
converter = SizerConverter(strategy)
converted_data = converter.convert(diameters, concentration)
```

#### Signature

```python
def get_distribution_conversion_strategy(
    input_scale: str, output_scale: str
) -> ConversionStrategy: ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)



## get_distribution_in_dn

[Show source in convert_size_distribution.py:346](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L346)

Convert the sizer data between dn/dlogdp and d_num formats.

If inverse=False, this function applies:
- d_num = dn_dlogdp × (log10(upper / lower))
    - The bin width is determined by upper and lower diameter limits,
      with log10 scaling.

If inverse=True, it reverts:
- dn/dlogdp = d_num / (log10(upper / lower))

#### Arguments

- diameter : Array of particle diameters.
- dn_dlogdp : Array representing either dn/dlogdp or d_num.
- inverse : If True, converts from d_num to dn/dlogdp; otherwise the
    opposite.

#### Returns

- A np.ndarray of the converted distribution.

#### Examples

```py
import numpy as np
from particula.util.size_distribution_convert import convert_sizer_dn

diam = np.array([1e-7, 2e-7, 4e-7])
dn_logdp = np.array([1e6, 1e5, 1e4])
result = convert_sizer_dn(diam, dn_logdp, inverse=False)
print(result)
# Output: d_num format for each diameter bin
```

#### References

- "dN/dlogD_p and dN/dD_p," TSI Application Note PR-001, 2010.
    [link](
    https://tsi.com/getmedia/1621329b-f410-4dce-992b-e21e1584481a/
    PR-001-RevA_Aerosol-Statistics-AppNote?ext=.pd)

#### Signature

```python
def get_distribution_in_dn(
    diameter: np.ndarray, dn_dlogdp: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## get_pdf_distribution_in_pmf

[Show source in convert_size_distribution.py:404](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L404)

Convert the distribution data between a probability density function (PDF)
and a probability mass spectrum (PMF).

The conversion uses:
- y_pdf = y_PMF / Δx
- y_PMF = y_pdf * Δx
  - Δx is the bin width, determined by consecutive differences in x_array.

#### Arguments

- x_array : An array of diameters/radii for the distribution bins.
- distribution : The original distribution data (PMF or PDF).
- to_pdf : If True, convert from PMF to PDF; if False, from PDF to PMF.

#### Returns

- A np.ndarray of the converted distribution data.

#### Examples

```py
import numpy as np
import particula as par
x_vals = np.array([1.0, 2.0, 3.0])
PMF = np.array([10.0, 5.0, 2.5])
pdf = par.get_pdf_distribution_in_pmf(x_vals, PMF, to_pdf=True)
print(pdf)
# Output: [10.  5.  2.5] / [1.0, 1.0, ...] = ...
```

#### References

- Detailed bin width discussion in: TSI Application Note
  "Aerosol Statistics and Densities."

#### Signature

```python
def get_pdf_distribution_in_pmf(
    x_array: np.ndarray, distribution: np.ndarray, to_pdf: bool = True
) -> np.ndarray: ...
```


---
# coulomb_enhancement.md

# Coulomb Enhancement

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Coulomb Enhancement

> Auto-generated documentation for [particula.particles.properties.coulomb_enhancement](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py) module.

## get_coulomb_continuum_limit

[Show source in coulomb_enhancement.py:144](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py#L144)

Calculate the continuum-limit Coulomb enhancement factor, Γ_c.

The continuum-limit factor is computed by:

- Γ_c =
    ϕ_E / [1 - exp(-ϕ_E)]    if ϕ_E ≠ 0
    1                        if ϕ_E = 0

where ϕ_E is the Coulomb potential ratio (dimensionless).

#### Arguments

- coulomb_potential : The Coulomb potential ratio ϕ_E (dimensionless).

#### Returns

- The Coulomb enhancement factor in the continuum limit (dimensionless).

#### Examples

``` py title="Example"
import numpy as np
import particula as par
potential = np.array([-0.5, 0.0, 0.5])
par.particles.get_coulomb_continuum_limit(potential)
print(gamma_cont)
# Output: array([...])
```

#### References

- Equation (6b): Gopalakrishnan, R., & Hogan, C. J. (2012).
  Coulomb-influenced collisions in aerosols and dusty plasmas.
  Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
@validate_inputs({"coulomb_potential": "finite"})
def get_coulomb_continuum_limit(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coulomb_enhancement_ratio

[Show source in coulomb_enhancement.py:23](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py#L23)

Calculate the Coulomb potential ratio, ϕ_E, for particle-particle
interactions.

The potential ratio is computed using:

- ϕ_E = - (qᵢ × qⱼ × e²) / [4π ε₀ (rᵢ + rⱼ) k_B T]
    - ϕ_E is the Coulomb potential ratio (dimensionless).
    - qᵢ, qⱼ are the charges (dimensionless, e.g. the number of electrons).
    - e is the elementary charge in coulombs (C).
    - ε₀ is the electric permittivity of free space (F·m⁻¹).
    - rᵢ, rⱼ are the particle radii (m).
    - k_B is the Boltzmann constant (J·K⁻¹).
    - T is the temperature (K).

#### Arguments

- radius : Radius of the particles (m).
- charge : Number of integer charges on the particles (dimensionless).
- temperature : System temperature (K).
- ratio_lower_limit : Lower limit to clip the potential ratio for very
  large negative (repulsive) values.

#### Returns

- The Coulomb potential ratio (dimensionless).

#### Examples

``` py title="Example"
import numpy as np
import particula as par
par.particles.get_coulomb_enhancement_ratio(
    radius=np.array([1e-7, 2e-7]),
    charge=np.array([1, 2]),
    temperature=298.15,
    ratio_lower_limit=-200
)
# Output: array([...])
```

#### References

- Equation (7): Gopalakrishnan, R., & Hogan, C. J. (2012).
  Coulomb-influenced collisions in aerosols and dusty plasmas.
  Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
@validate_inputs({"particle_radius": "nonnegative"})
def get_coulomb_enhancement_ratio(
    particle_radius: Union[float, NDArray[np.float64]],
    charge: Union[int, NDArray[np.float64]] = 0,
    temperature: float = 298.15,
    ratio_lower_limit: float = -200,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_coulomb_kinetic_limit

[Show source in coulomb_enhancement.py:100](https://github.com/uncscode/particula/blob/main/particula/particles/properties/coulomb_enhancement.py#L100)

Calculate the kinetic-limit Coulomb enhancement factor, Γₖᵢₙ.

The kinetic-limit factor is computed by:

- Γₖᵢₙ =
    1 + ϕ_E      if ϕ_E ≥ 0
    exp(ϕ_E)     if ϕ_E < 0

where ϕ_E is the Coulomb potential ratio (dimensionless).

#### Arguments

- coulomb_potential : The Coulomb potential ratio ϕ_E (dimensionless).

#### Returns

- The Coulomb enhancement factor in the kinetic limit (dimensionless).

#### Examples

``` py title="Example"
import numpy as np
import particula as par
potential = np.array([-0.5, 0.0, 0.5])
par.particles.get_coulomb_kinetic_limit(potential)
# Output: array([...])
```

#### References

- Equations (6d) and (6e): Gopalakrishnan, R., & Hogan, C. J. (2012).
  Coulomb-influenced collisions in aerosols and dusty plasmas.
  Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
@validate_inputs({"coulomb_potential": "finite"})
def get_coulomb_kinetic_limit(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# diffusion_coefficient.md

# Diffusion Coefficient

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Diffusion Coefficient

> Auto-generated documentation for [particula.particles.properties.diffusion_coefficient](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusion_coefficient.py) module.

## get_diffusion_coefficient

[Show source in diffusion_coefficient.py:22](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusion_coefficient.py#L22)

Calculate the diffusion coefficient of a particle based on temperature
and aerodynamic mobility.

The diffusion coefficient (D) can be computed using:

- D = k_B T × B
    - D is the diffusion coefficient in m²/s,
    - k_B is the Boltzmann constant in J/K,
    - T is the temperature in Kelvin,
    - B is the aerodynamic mobility in m²/s.

#### Arguments

- temperature : Temperature in Kelvin (K).
- aerodynamic_mobility : Aerodynamic mobility in m²/s.
- boltzmann_constant : Boltzmann constant in J/K.

#### Returns

- The diffusion coefficient of the particle in m²/s.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_diffusion_coefficient(
    temperature=300.0, aerodynamic_mobility=1.0e-8
)
# Output: ...
```

#### References

- Einstein, A. (1905). "On the movement of small particles suspended
  in stationary liquids required by the molecular-kinetic theory of
  heat." Annalen der Physik, 17(8), 549–560.
- "Stokes-Einstein equation," Wikipedia,
  https://en.wikipedia.org/wiki/Stokes%E2%80%93Einstein_equation

#### Signature

```python
@validate_inputs({"temperature": "positive", "aerodynamic_mobility": "nonnegative"})
def get_diffusion_coefficient(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
    boltzmann_constant: float = BOLTZMANN_CONSTANT,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [BOLTZMANN_CONSTANT](../../util/constants.md#boltzmann_constant)



## get_diffusion_coefficient_via_system_state

[Show source in diffusion_coefficient.py:72](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusion_coefficient.py#L72)

Calculate the diffusion coefficient from system state parameters.

This function determines the diffusion coefficient (D) of a particle by:
1. Computing gas properties (dynamic viscosity, mean free path),
2. Determining particle slip correction and aerodynamic mobility,
3. Calling get_diffusion_coefficient() to get D.

#### Arguments

- particle_radius : Particle radius in meters (m).
- temperature : System temperature in Kelvin (K).
- pressure : System pressure in Pascals (Pa).

#### Returns

- The diffusion coefficient of the particle in m²/s.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_diffusion_coefficient_via_system_state(
    particle_radius=1.0e-7,
    temperature=298.15,
    pressure=101325
)
# Output: ...
```

#### References

- Millikan, R. A. (1923). "On the elementary electrical charge and the
  Avogadro constant." Physical Review, 2(2), 109–143. [check]
- "Mass Diffusion," Wikipedia,
  https://en.wikipedia.org/wiki/Diffusion#Mass_diffusion

#### Signature

```python
def get_diffusion_coefficient_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# diffusive_knudsen_module.md

# Diffusive Knudsen Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Diffusive Knudsen Module

> Auto-generated documentation for [particula.particles.properties.diffusive_knudsen_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusive_knudsen_module.py) module.

## get_diffusive_knudsen_number

[Show source in diffusive_knudsen_module.py:13](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusive_knudsen_module.py#L13)

Compute the diffusive Knudsen number for particle-particle interactions.

The *diffusive* Knudsen number (Kn_d) differs from the standard Knudsen
number. It represents the ratio of the mean particle persistence
distance to the effective Coulombic interaction scale. Mathematically:

- Kn_d = [ √(k_B × T × μ_red) / f_red ] / [ (rᵢ + rⱼ) × (Γ_c / Γ_k) ]
    - k_B is the Boltzmann constant (J/K).
    - T is the temperature (K).
    - μ_red is the reduced mass of particles (kg).
    - f_red is the reduced friction factor (dimensionless).
    - rᵢ + rⱼ is the sum of radii for the interacting particles (m).
    - Γ_c is the continuum-limit Coulomb enhancement factor(dimensionless).
    - Γ_k is the kinetic-limit Coulomb enhancement factor (dimensionless).

#### Arguments

- particle_radius : Radius of the particle(s) in meters (m).
- particle_mass : Mass of the particle(s) in kilograms (kg).
- friction_factor : Friction factor(s) (dimensionless).
- coulomb_potential_ratio : Coulomb potential ratio (dimensionless),
  zero if no charge.
- temperature : Temperature of the system in Kelvin (K).

#### Returns

- The diffusive Knudsen number, either a float or NDArray[np.float64].

#### Examples

```py title="Single Particle Example"
import numpy as np
import particula as par
par.particles.get_diffusive_knudsen_number(
    particle_radius=1e-7,
    particle_mass=1e-17,
    friction_factor=0.8,
    coulomb_potential_ratio=0.3,
    temperature=300
)
# Output: 0.12...
```

```py title="Multiple Particles Example"
import numpy as np
import particula as par
# Multiple particles example
radius_arr = np.array([1e-7, 2e-7])
mass_arr = np.array([1e-17, 2e-17])
friction_arr = np.array([0.8, 1.1])
potential_arr = np.array([0.3, 0.5])
par.particles.par.get_diffusive_knudsen_number(
    radius_arr, mass_arr, friction_arr, potential_arr
)
# Output: array([...])
```

#### References

- Chahl, H. S., & Gopalakrishnan, R. (2019). "High potential, near free
  molecular regime Coulombic collisions in aerosols and dusty plasmas."
  Aerosol Science and Technology, 53(8), 933-957.
  https://doi.org/10.1080/02786826.2019.1614522
- Gopalakrishnan, R., & Hogan, C. J. (2012). "Coulomb-influenced
  collisions in aerosols and dusty plasmas." Physical Review E, 85(2).
  https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "particle_mass": "nonnegative",
        "friction_factor": "nonnegative",
    }
)
def get_diffusive_knudsen_number(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    friction_factor: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]] = 0.0,
    temperature: float = 298.15,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# friction_factor_module.md

# Friction Factor Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Friction Factor Module

> Auto-generated documentation for [particula.particles.properties.friction_factor_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/friction_factor_module.py) module.

## get_friction_factor

[Show source in friction_factor_module.py:18](https://github.com/uncscode/particula/blob/main/particula/particles/properties/friction_factor_module.py#L18)

Calculate the friction factor for a particle in a fluid.

This friction factor (f) is the proportionality constant between
the fluid velocity and the resulting drag force on the particle.
The formula used is:

- f = (6πμ r) / C
    - f is the friction factor (N·s/m),
    - μ is the dynamic viscosity of the fluid (Pa·s),
    - r is the radius of the particle (m),
    - C is the slip correction factor (dimensionless).

#### Arguments

- particle_radius : Radius of the particle in meters (m).
- dynamic_viscosity : Dynamic viscosity of the fluid in Pa·s.
- slip_correction : Slip correction factor (dimensionless).

#### Returns

- The friction factor of the particle in N·s/m.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_friction_factor(
    particle_radius=1e-7,
    dynamic_viscosity=1.8e-5,
    slip_correction=1.1
)
# Output: ...
```

#### References

- Zhang, C., Thajudeen, T., Larriba, C., Schwartzentruber, T. E.,
  & Hogan, C. J. (2012). "Determination of the Scalar Friction Factor
  for Nonspherical Particles and Aggregates Across the Entire Knudsen
  Number Range by Direct Simulation Monte Carlo (DSMC)."
  Aerosol Science and Technology, 46(10), 1065-1078.
  https://doi.org/10.1080/02786826.2012.690543

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "dynamic_viscosity": "positive",
        "slip_correction": "positive",
    }
)
def get_friction_factor(
    particle_radius: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    slip_correction: Union[float, NDArray[np.float64]],
): ...
```


---
# inertia_time.md

# Inertia Time

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Inertia Time

> Auto-generated documentation for [particula.particles.properties.inertia_time](https://github.com/uncscode/particula/blob/main/particula/particles/properties/inertia_time.py) module.

## get_particle_inertia_time

[Show source in inertia_time.py:12](https://github.com/uncscode/particula/blob/main/particula/particles/properties/inertia_time.py#L12)

Compute the particle inertia time (τ_p).

The particle inertia time represents the response time of a particle to
changes in fluid velocity, given by:

- τ_p = (2 / 9) × (ρ_p / ρ_f) × (r² / ν)
    - τ_p is the particle inertia time in seconds (s).
    - ρ_p is the particle density (kg/m³).
    - ρ_f is the surrounding fluid density (kg/m³).
    - r is the particle radius (m).
    - ν is the kinematic viscosity of the fluid (m²/s).

#### Arguments

- particle_radius : Particle radius in meters (m).
- particle_density : Density of the particle in kg/m³.
- fluid_density : Density of the fluid in kg/m³.
- kinematic_viscosity : Kinematic viscosity of the fluid in m²/s.

#### Returns

- The particle inertia time in seconds (s). Returned as either a float
    or NDArray[np.float64].

#### Examples

```py title="Example"
import particula as par
par.particles.get_particle_inertia_time(
    particle_radius=1e-6,
    particle_density=1000.0,
    fluid_density=1.225,
    kinematic_viscosity=1.5e-5
)
# Output: ...
```

#### References

- Ayala, O., Rosa, B., & Wang, L. P. (2008). "Effects of turbulence on
  the geometric collision rate of sedimenting droplets. Part 2. Theory
  and parameterization." New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075016

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "particle_density": "positive",
        "fluid_density": "positive",
        "kinematic_viscosity": "positive",
    }
)
def get_particle_inertia_time(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# kelvin_effect_module.md

# Kelvin Effect Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Kelvin Effect Module

> Auto-generated documentation for [particula.particles.properties.kelvin_effect_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kelvin_effect_module.py) module.

## get_kelvin_radius

[Show source in kelvin_effect_module.py:12](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kelvin_effect_module.py#L12)

Compute the Kelvin radius (rₖ) to account for curvature effects on vapor
pressure.

The Kelvin radius is defined by:

- rₖ = (2 × σ × M) / (R × T × ρ)
    - rₖ is Kelvin radius in meters (m).
    - σ is the effective surface tension in N/m.
    - M is the molar mass in kg/mol.
    - R is the universal gas constant in J/(mol·K).
    - T is the temperature in Kelvin (K).
    - ρ is the effective density in kg/m³.

#### Arguments

- effective_surface_tension : Surface tension of the mixture (N/m).
- effective_density : Effective density of the mixture (kg/m³).
- molar_mass : Molar mass (kg/mol).
- temperature : Temperature of the system (K).

#### Returns

- Kelvin radius in meters (float or NDArray[np.float64]).

#### Examples

``` py title="Example"
import numpy as np
import particula as par
par.particles.get_kelvin_radius(
    effective_surface_tension=0.072,
    effective_density=1000.0,
    molar_mass=0.018,
    temperature=298.15
)
# Output: ...
```

#### References

- "Kelvin equation," Wikipedia,
  https://en.wikipedia.org/wiki/Kelvin_equation

#### Signature

```python
@validate_inputs(
    {
        "effective_surface_tension": "positive",
        "effective_density": "positive",
        "molar_mass": "positive",
        "temperature": "positive",
    }
)
def get_kelvin_radius(
    effective_surface_tension: Union[float, NDArray[np.float64]],
    effective_density: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_kelvin_term

[Show source in kelvin_effect_module.py:71](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kelvin_effect_module.py#L71)

Compute the Kelvin exponential term to account for curvature effects.

The Kelvin term (K) is given by:

- K = exp(rₖ / rₚ)
    - K is dimensionless.
    - rₖ is the Kelvin radius (m).
    - rₚ is the particle radius (m).

#### Arguments

- particle_radius : Radius of the particle (m).
- kelvin_radius_value : Precomputed Kelvin radius (m).

#### Returns

- Dimensionless exponential factor adjusting vapor pressure.

#### Examples

``` py title="Example"
import particula as par
par.particles.get_kelvin_term(
    particle_radius=1e-7,
    kelvin_radius_value=2e-7
)
print(kv_term)
# Output: ...
```

#### References

- Donahue, N. M., et al. (2013). "How do organic vapors contribute to
  new-particle formation?" Faraday Discussions, 165, 91–104.
  https://doi.org/10.1039/C3FD00046J. [check]

#### Signature

```python
@validate_inputs(
    {"particle_radius": "nonnegative", "kelvin_radius_value": "nonnegative"}
)
def get_kelvin_term(
    particle_radius: Union[float, NDArray[np.float64]],
    kelvin_radius_value: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# knudsen_number_module.md

# Knudsen Number Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Knudsen Number Module

> Auto-generated documentation for [particula.particles.properties.knudsen_number_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/knudsen_number_module.py) module.

## get_knudsen_number

[Show source in knudsen_number_module.py:13](https://github.com/uncscode/particula/blob/main/particula/particles/properties/knudsen_number_module.py#L13)

Calculate the Knudsen number (Kn) from the gas mean free path and particle
radius.

The Knudsen number (Kn) indicates whether a flow is in the continuum
regime or the free molecular regime. It is computed by:

- Kn = λ / r
    - Kn is the Knudsen number (dimensionless),
    - λ is the mean free path in meters (m),
    - r is the particle radius in meters (m).

#### Arguments

- mean_free_path : Mean free path of the gas molecules in meters (m).
- particle_radius : Radius of the particle in meters (m).

#### Returns

- The Knudsen number, which is the ratio of the mean free path to the
    particle radius.

#### Examples

``` py title="Example Usage"
import particula as par
par.particles.get_knudsen_number(6.5e-8, 1.0e-7)
# Output: 0.65
```

#### References

- Knudsen number, Wikipedia,
  https://en.wikipedia.org/wiki/Knudsen_number

#### Signature

```python
@validate_inputs({"mean_free_path": "nonnegative", "particle_radius": "nonnegative"})
def get_knudsen_number(
    mean_free_path: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# lognormal_size_distribution.md

# Lognormal Size Distribution

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Lognormal Size Distribution

> Auto-generated documentation for [particula.particles.properties.lognormal_size_distribution](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py) module.

## get_lognormal_pdf_distribution

[Show source in lognormal_size_distribution.py:16](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py#L16)

Compute a lognormal probability density function (PDF) for given modes.

This function superimposes multiple lognormal PDFs, each with its own mode,
geometric standard deviation, and particle count. It then returns their sum
across the provided x_values. Mathematically, for each mode i:

- PDFᵢ(x) = (1 / [x · ln(gsdᵢ) · √(2π)]) ×
             exp(- [ln(x) - ln(modeᵢ)]² / [2 · (ln(gsdᵢ))² ])

#### Arguments

- x_values : 1D array of the size points at which the PDF is evaluated.
- mode : Array of lognormal mode (scale) values for each mode.
- geometric_standard_deviation : Array of GSD values for each mode.
- number_of_particles : Number of particles in each mode.

#### Returns

- 1D array of the total PDF values summed across all modes.

#### Examples

```py title="Example"
import numpy as np
import particula as par
x_vals = np.linspace(1e-9, 1e-6, 100)
par.particles.get_lognormal_pdf_distribution(
    x_values=x_vals,
    mode=np.array([5e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.5, 2.0]),
    number_of_particles=np.array([1e9, 5e9])
)
# Output: [...]
```

#### References

- [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
- [Probability Density Function Wikipedia](https://en.wikipedia.org/wiki/Probability_density_function)
- [Scipy Lognorm Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

#### Signature

```python
@validate_inputs(
    {
        "x_values": "nonnegative",
        "mode": "positive",
        "geometric_standard_deviation": "positive",
        "number_of_particles": "positive",
    }
)
def get_lognormal_pdf_distribution(
    x_values: NDArray[np.float64],
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_lognormal_pmf_distribution

[Show source in lognormal_size_distribution.py:94](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py#L94)

Compute a lognormal probability mass function (PMF) for given modes.

This function first calculates the lognormal PDF using
get_lognormal_pdf_distribution(), then converts it to a PMF by
integrating (or summing) over x_values. The result reflects discrete mass
(probability) distribution across the given size points.

#### Arguments

- x_values : 1D array of size points at which the PMF is evaluated.
- mode : Array of lognormal mode (scale) values for each mode.
- geometric_standard_deviation : Array of GSD values for each mode.
- number_of_particles : Number of particles in each mode.

#### Returns

- 1D array of the total PMF values summed across all modes.

#### Examples

```py title="Example"
import numpy as np
import particula as par
x_vals = np.linspace(1e-9, 1e-6, 100)
par.particles.get_lognormal_pmf_distribution(
    x_values=x_vals,
    mode=np.array([5e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.5, 2.0]),
    number_of_particles=np.array([1e9, 5e9])
)
# Output: [...]
```

#### References

- [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
- [Probability Mass Function Wikipedia](https://en.wikipedia.org/wiki/Probability_mass_function)
- [Scipy Lognorm Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

#### Signature

```python
@validate_inputs(
    {
        "x_values": "nonnegative",
        "mode": "positive",
        "geometric_standard_deviation": "positive",
        "number_of_particles": "positive",
    }
)
def get_lognormal_pmf_distribution(
    x_values: NDArray[np.float64],
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_lognormal_sample_distribution

[Show source in lognormal_size_distribution.py:168](https://github.com/uncscode/particula/blob/main/particula/particles/properties/lognormal_size_distribution.py#L168)

Generate random samples from a lognormal distribution for given modes.

This function uses scipy.stats.lognorm.rvs() to draw samples for each mode,
with a specified scale (mode) and shape (GSD). The total samples are then
combined according to the relative number of particles in each mode.

#### Arguments

- mode : Array of lognormal mode (scale) values for each mode.
- geometric_standard_deviation : Array of GSD values for each mode.
- number_of_particles : Number of particles for each mode.
- number_of_samples : Total number of random samples to generate.

#### Returns

- 1D array of sampled particle sizes, combining all modes.

#### Examples

```py title="Example"
import numpy as np
import particula as par
par.particles.get_lognormal_sample_distribution(
    mode=np.array([5e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.5, 2.0]),
    number_of_particles=np.array([1e9, 5e9]),
    number_of_samples=10_000
)
# Output: [...]
```

#### References

- [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
- [Probability Density Function Wikipedia](https://en.wikipedia.org/wiki/Probability_density_function)
- [Scipy Lognorm Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

#### Signature

```python
@validate_inputs(
    {
        "mode": "positive",
        "geometric_standard_deviation": "positive",
        "number_of_particles": "positive",
        "number_of_samples": "positive",
    }
)
def get_lognormal_sample_distribution(
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
    number_of_samples: int,
    upper_bound: float = np.inf,
    lower_bound: float = 0,
) -> NDArray[np.float64]: ...
```


---
# mean_thermal_speed_module.md

# Mean Thermal Speed Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Mean Thermal Speed Module

> Auto-generated documentation for [particula.particles.properties.mean_thermal_speed_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/mean_thermal_speed_module.py) module.

## get_mean_thermal_speed

[Show source in mean_thermal_speed_module.py:14](https://github.com/uncscode/particula/blob/main/particula/particles/properties/mean_thermal_speed_module.py#L14)

Calculate the mean thermal speed of a particle in a fluid.

The mean thermal speed (v) is derived from kinetic theory and is given by:

- v = √( (8 × k_B × T) / (π × m) )
    - v is the mean thermal speed in m/s,
    - k_B is the Boltzmann constant in J/K,
    - T is the temperature in Kelvin (K),
    - m is the particle mass in kilograms (kg).

#### Arguments

- particle_mass : The mass of the particle(s) in kg.
- temperature : The temperature of the system in Kelvin (K).

#### Returns

- The mean thermal speed in m/s, as either a float or an
    NDArray[np.float64].

#### Examples

``` py title="Example"
import particula as par
par.particles.get_mean_thermal_speed(1e-17, 298)
# Output: ...
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
  Physics, Section 9.5.3 Mean Free Path of an Aerosol Particle,
  Equation 9.87.

#### Signature

```python
@validate_inputs({"particle_mass": "nonnegative", "temperature": "positive"})
def get_mean_thermal_speed(
    particle_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# partial_pressure_module.md

# Partial Pressure Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Partial Pressure Module

> Auto-generated documentation for [particula.particles.properties.partial_pressure_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/partial_pressure_module.py) module.

## get_partial_pressure_delta

[Show source in partial_pressure_module.py:9](https://github.com/uncscode/particula/blob/main/particula/particles/properties/partial_pressure_module.py#L9)

Calculate the difference in partial pressure between gas and particle
phase, considering the Kelvin effect.

- Δp = p_gas − (p_particle × K)
    - p_gas is the partial pressure in the gas phase,
    - p_particle is the partial pressure in the particle phase,
    - K is the Kelvin term (dimensionless).

#### Arguments

- partial_pressure_gas : Partial pressure of the species in the gas
    phase.
- partial_pressure_particle : Partial pressure of the species in
    the particle phase.
- kelvin_term : Dimensionless Kelvin effect factor due to particle
    curvature.

#### Returns

- The difference in partial pressure, as either a float or
    NDArray[np.float64].

#### Examples

``` py title="Example"
import particula as par
par.particles.get_partial_pressure_delta(
    partial_pressure_gas=1000.0,
    partial_pressure_particle=900.0,
    kelvin_term=1.01
)
# Output: 1000.0 - (900.0 * 1.01) = 91.0
```

#### References

- [Kelvin effect, Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)
- [Partial pressure, Wikipedia](https://en.wikipedia.org/wiki/Partial_pressure)

#### Signature

```python
def get_partial_pressure_delta(
    partial_pressure_gas: Union[float, NDArray[np.float64]],
    partial_pressure_particle: Union[float, NDArray[np.float64]],
    kelvin_term: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# reynolds_number.md

# Reynolds Number

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Reynolds Number

> Auto-generated documentation for [particula.particles.properties.reynolds_number](https://github.com/uncscode/particula/blob/main/particula/particles/properties/reynolds_number.py) module.

## get_particle_reynolds_number

[Show source in reynolds_number.py:12](https://github.com/uncscode/particula/blob/main/particula/particles/properties/reynolds_number.py#L12)

Calculate the Reynolds number (Reₚ) of a particle in a fluid.

This dimensionless quantity characterizes the flow regime:

- Reₚ = (2 × a × vₚ) / ν
    - a is the particle radius in meters (m).
    - vₚ is the particle velocity in meters/second (m/s).
    - ν is the kinematic viscosity in square meters/second (m²/s).

#### Arguments

- particle_radius : Particle radius (m).
- particle_velocity : Particle velocity relative to the fluid (m/s).
- kinematic_viscosity : Kinematic viscosity of the fluid (m²/s).

#### Returns

- Dimensionless Reynolds number (float or NDArray[np.float64]).

#### Examples

```py title="Example"
import particula as par
par.particles.get_particle_reynolds_number(
    particle_radius=1e-6,
    particle_velocity=0.1,
    kinematic_viscosity=1.5e-5
)
# Output: ...
```

#### References

- [Reynolds number, Wikipedia](https://en.wikipedia.org/wiki/Reynolds_number)
- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
    Physics,
- **Stokes Flow (Viscous Dominated, Re_p < 1)**:
    - Particles follow the fluid closely (e.g., aerosols).
- **Transitional Flow (1 < Re_p < 1000)**:
    - Both **viscous and inertial forces** contribute to flow behavior.
    - Intermediate drag corrections apply.
- **Turbulent Flow (Re_p > 1000)**:
    - **Inertial forces dominate**, resulting in vortex shedding and
        wake formation.
    - Applies to **large, fast-moving particles**
        (e.g., raindrops, large sediment).

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "positive",
        "particle_velocity": "positive",
        "kinematic_viscosity": "positive",
    }
)
def get_particle_reynolds_number(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# settling_velocity.md

# Settling Velocity

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Settling Velocity

> Auto-generated documentation for [particula.particles.properties.settling_velocity](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py) module.

## _drag_coefficient

[Show source in settling_velocity.py:407](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L407)

Return drag coefficient c_d given a Reynolds number Re.

#### Arguments

- reynolds_number : Reynolds number [-].

#### Returns

- Drag coefficient c_d [-].

#### Signature

```python
def _drag_coefficient(reynolds_number: float) -> float: ...
```



## _velocity_mismatch

[Show source in settling_velocity.py:428](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L428)

Calculate the mismatch between predicted and actual velocities.

#### Arguments

- velocity : Current estimate of particle velocity [m/s].
- radius : Particle radius [m].
- rho_p : Particle density [kg/m³].
- fluid_density : Fluid density [kg/m³].
- dynamic_viscosity : Dynamic viscosity of the fluid [Pa·s].
- gravitational_acceleration : Gravitational acceleration [m/s²].

#### Returns

- Squared difference between predicted and actual velocities.

#### Signature

```python
def _velocity_mismatch(
    velocity: float,
    radius: float,
    rho_p: float,
    fluid_density: float,
    kinematic_viscosity: float,
    gravitational_acceleration: float,
) -> float: ...
```



## get_particle_settling_velocity

[Show source in settling_velocity.py:31](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L31)

Calculate the settling velocity of a particle in a fluid using Stokes' law.

The settling velocity (vₛ) is given by the equation:

- vₛ = (2 × r² × (ρₚ − ρ_f) × g × C_c) / (9 × μ)
    - vₛ : Settling velocity in m/s
    - r : Particle radius in m
    - ρₚ : Particle density in kg/m³
    - ρ_f : Fluid density in kg/m³
    - g : Gravitational acceleration in m/s²
    - C_c : Cunningham slip correction factor (dimensionless)
    - μ : Dynamic viscosity in Pa·s

#### Arguments

- particle_radius : The radius of the particle in meters.
- particle_density : The density of the particle in kg/m³.
- slip_correction_factor : Account for non-continuum effects
    (dimensionless).
- dynamic_viscosity : Dynamic viscosity of the fluid in Pa·s.
- gravitational_acceleration : Gravitational acceleration in m/s².
- fluid_density : The fluid density in kg/m³. Defaults to 0.0.

#### Returns

- Settling velocity of the particle in m/s.

#### Examples

```py title="Array Input Example"
import numpy as np
import particula as par
par.particles.particle_settling_velocity(
    particle_radius=np.array([1e-6, 1e-5, 1e-4]),
    particle_density=np.array([1000, 2000, 3000]),
    slip_correction_factor=np.array([1, 1, 1]),
    dynamic_viscosity=1.0e-3
)
# Output: array([...])
```

#### References

- "Stokes' Law," Wikipedia,
  https://en.wikipedia.org/wiki/Stokes%27_law

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "particle_density": "positive",
        "slip_correction_factor": "positive",
        "dynamic_viscosity": "nonnegative",
    }
)
def get_particle_settling_velocity(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    gravitational_acceleration: float = STANDARD_GRAVITY,
    fluid_density: float = 0.0,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [STANDARD_GRAVITY](../../util/constants.md#standard_gravity)



## get_particle_settling_velocity_via_inertia

[Show source in settling_velocity.py:104](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L104)

Calculate gravitational settling velocity using particle inertia time.

The settling velocity (vₛ) is determined by:

- vₛ = (g × τₚ × C_c) / f(Reₚ)
    - g is gravitational acceleration (m/s²).
    - τₚ is particle inertia time (s).
    - C_c is the Cunningham slip correction factor (dimensionless).
    - f(Reₚ) is the drag correction factor, 1 + 0.15 × Reₚ^0.687.
    - Reₚ is particle Reynolds number (dimensionless).

#### Arguments

- particle_inertia_time : Particle inertia time in seconds (s).
- particle_radius : Particle radius in meters (m).
- relative_velocity : Relative velocity between particle and fluid
    (m/s).
- slip_correction_factor : Cunningham slip correction factor
    (dimensionless).
- gravitational_acceleration : Gravitational acceleration (m/s²).
- kinematic_viscosity : Kinematic viscosity of the fluid (m²/s).

#### Returns

- Particle settling velocity in m/s.

#### Examples

```py title="Example"
import particula as par
par.particles.get_particle_settling_velocity_via_inertia(
    particle_inertia_time=0.002,
    particle_radius=1.0e-6,
    relative_velocity=0.1,
    slip_correction_factor=1.05,
    gravitational_acceleration=9.81,
    kinematic_viscosity=1.5e-5
)
# Output: ...
```

#### References

- Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008).
  "Effects of turbulence on the geometric collision rate of
  sedimenting droplets. Part 1. Results from direct numerical
  simulation." New Journal of Physics, 10.
  https://doi.org/10.1088/1367-2630/10/7/075015

#### Signature

```python
@validate_inputs(
    {
        "particle_inertia_time": "positive",
        "gravitational_acceleration": "positive",
        "slip_correction_factor": "positive",
    }
)
def get_particle_settling_velocity_via_inertia(
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
    relative_velocity: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    gravitational_acceleration: float,
    kinematic_viscosity: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_particle_settling_velocity_via_system_state

[Show source in settling_velocity.py:181](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L181)

Compute the particle settling velocity based on system state parameters.

This function calculates the dynamic viscosity from temperature, the mean
free path from the same system state, and the Knudsen number of the
particle, then applies the slip correction factor. Finally, it returns
the settling velocity from Stokes' law with slip correction.

#### Arguments

- particle_radius : Particle radius in meters (m).
- particle_density : Particle density in kg/m³.
- temperature : Temperature of the system in Kelvin (K).
- pressure : Pressure of the system in Pascals (Pa).

#### Returns

- Settling velocity of the particle in m/s.

#### Examples

```py title="System State Example"
import particula as par
par.particles.particle_settling_velocity_via_system_state(
    particle_radius=1e-6,
    particle_density=1200,
    temperature=298.15,
    pressure=101325
)
# Output: ...
```

#### References

- Gas viscosity property estimation:
  https://en.wikipedia.org/wiki/Viscosity#Gases
- Slip correction and Knudsen number relations from:
  Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric
  Chemistry and Physics. Wiley-Interscience.

#### Signature

```python
def get_particle_settling_velocity_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## get_particle_settling_velocity_with_drag

[Show source in settling_velocity.py:253](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L253)

Calculate the particle's terminal settling velocity with a full drag model.

For low Reynolds numbers (Re < re_threshold), the Stokes settling velocity
(with slip correction) is used:

- vₛ(Stokes) = (2/9) × [r² × (ρₚ − ρ_f) × g × C_c] / μ

For higher Reynolds numbers, a force-balance approach is solved
numerically, using a variable drag coefficient (c_d).

- vₛ = fminbound(mismatch, 0, v_upper)
    - mismatch = (v_pred - v)²
    - v_pred = sqrt((8 × r × (ρₚ - ρ_f) × g) / (3 × ρ_f × c_d))

#### Arguments

- particle_radius : Particle radius (m).
- particle_density : Particle density (kg/m³).
- fluid_density : Fluid density (kg/m³).
- dynamic_viscosity : Fluid dynamic viscosity (Pa·s).
- slip_correction_factor : Slip correction factor, dimensionless.
- gravitational_acceleration : Gravitational acceleration (m/s²),
  default is 9.80665.
- re_threshold : Reynolds number threshold (dimensionless),
  default is 0.1.
- tol : Numeric tolerance for solver (dimensionless),
  default is 1e-6.
- max_iter : Maximum function evaluations in numeric solver,
  default is 100.

#### Returns

- Terminal settling velocity (m/s). Scalar or NDArray.

#### Examples

```py title="Example"
import numpy as np
import particula as par
r_array = np.array([1e-6, 5e-5, 2e-4])
rho_array = np.array([1500, 2000, 1850])
par.particles.get_particle_settling_velocity_with_drag(
    particle_radius=r_array,
    particle_density=rho_array,
    fluid_density=1.225,
    dynamic_viscosity=1.8e-5,
    slip_correction_factor=np.array([1.0, 0.95, 1.1])
)
# Output: array([...])
```

#### References

- "Drag Coefficient," Wikipedia,
  https://en.wikipedia.org/wiki/Drag_coefficient
- Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry
  and Physics*, 3rd ed., John Wiley & Sons.

#### Signature

```python
@validate_inputs(
    {
        "particle_radius": "positive",
        "particle_density": "positive",
        "fluid_density": "positive",
        "dynamic_viscosity": "nonnegative",
    }
)
def get_particle_settling_velocity_with_drag(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: float,
    dynamic_viscosity: float,
    slip_correction_factor: Union[float, NDArray[np.float64]],
    gravitational_acceleration: float = STANDARD_GRAVITY,
    re_threshold: float = 0.1,
    tol: float = 1e-06,
    max_iter: int = 100,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [STANDARD_GRAVITY](../../util/constants.md#standard_gravity)


---
# slip_correction_module.md

# Slip Correction Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Slip Correction Module

> Auto-generated documentation for [particula.particles.properties.slip_correction_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/slip_correction_module.py) module.

## get_cunningham_slip_correction

[Show source in slip_correction_module.py:11](https://github.com/uncscode/particula/blob/main/particula/particles/properties/slip_correction_module.py#L11)

Calculate the Cunningham slip correction factor for small particles in a
fluid.

The slip correction factor (C_c) accounts for non-continuum effects on
small particles, correcting for the no-slip assumption used in Stokes'
law. It is calculated using:

- C_c = 1 + Kn × (1.257 + 0.4 × exp(-1.1 / Kn))
    - Kn is the dimensionless Knudsen number.

#### Arguments

- knudsen_number : Knudsen number (dimensionless).

#### Returns

- Slip correction factor (dimensionless).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_cunningham_slip_correction(0.1)
# Output: ...
```

#### References

- "Cunningham correction factor," Wikipedia,
  https://en.wikipedia.org/wiki/Cunningham_correction_factor

#### Signature

```python
@validate_inputs({"knudsen_number": "nonnegative"})
def get_cunningham_slip_correction(
    knudsen_number: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# special_functions.md

# Special Functions

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Special Functions

> Auto-generated documentation for [particula.particles.properties.special_functions](https://github.com/uncscode/particula/blob/main/particula/particles/properties/special_functions.py) module.

## get_debye_function

[Show source in special_functions.py:13](https://github.com/uncscode/particula/blob/main/particula/particles/properties/special_functions.py#L13)

Calculate the generalized Debye function for a given input.

The Debye function can be expressed as follows:

- Dₙ(x) = (n / xⁿ) ∫[tⁿ / (exp(t) - 1)] dt  from t = 0 to x
    - x is a dimensionless variable.
    - n is the exponent (default is 1).

#### Arguments

- variable : Upper limit of integration; can be float or NDArray.
- integration_points : Number of points for numerical integration
  (default 1000).
- n : Exponent in the Debye function formula (default 1).

#### Returns

- Debye function value(s). If the input is a float, returns a float.
  If the input is an array, returns an array of the same shape.

#### Examples

``` py title="Debye function with n=1 for a single float value"
import particula as par
par.particles.get_debye_function(1.0)
# Output: 0.7765038970390566
```

``` py title="Debye function with n=2 for a single float value"
import particula as par
par.particles.get_debye_function(1.0, n=2)
# Output: 0.6007582206816492
```

``` py title="Debye function with n=1 for a numpy array"
import particula as par
par.particles.get_debye_function(np.array([1.0, 2.0, 3.0]))
# Output: [0.84140566 0.42278434 0.28784241]
```

#### References

- [Debye function](https://en.wikipedia.org/wiki/Debye_function)
- [Wolfram MathWorld: Debye Functions](https://mathworld.wolfram.com/DebyeFunctions.html)

#### Signature

```python
@validate_inputs({"variable": "finite"})
def get_debye_function(
    variable: Union[float, NDArray[np.float64]],
    integration_points: int = 1000,
    n: int = 1,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# stokes_number.md

# Stokes Number

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Stokes Number

> Auto-generated documentation for [particula.particles.properties.stokes_number](https://github.com/uncscode/particula/blob/main/particula/particles/properties/stokes_number.py) module.

## get_stokes_number

[Show source in stokes_number.py:12](https://github.com/uncscode/particula/blob/main/particula/particles/properties/stokes_number.py#L12)

Compute the Stokes number (St) to measure particle inertia relative to
fluid flow.

The Stokes number is a dimensionless parameter reflecting how much a
particle resists following changes in the fluid’s motion. If St >> 1,
particle inertia dominates; if St << 1, the particle closely follows
fluid flow. Mathematically:

- St = τ_p / τ_k
    - St : Stokes number (dimensionless),
    - τ_p : Particle inertia time [s],
    - τ_k : Kolmogorov timescale [s].

#### Arguments

- particle_inertia_time : Particle inertia time in seconds (s).
- kolmogorov_time : Kolmogorov timescale in seconds (s).

#### Returns

- Dimensionless Stokes number (float or NDArray[np.float64]).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_stokes_number(1e-3, 2e-3)
# Output: 0.5
```

#### References

- [Stokes number, Wikipedia](https://en.wikipedia.org/wiki/Stokes_number)
- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
  Physics, 3rd ed., Wiley-Interscience.

#### Signature

```python
@validate_inputs({"particle_inertia_time": "positive", "kolmogorov_time": "positive"})
def get_stokes_number(
    particle_inertia_time: Union[float, NDArray[np.float64]],
    kolmogorov_time: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# vapor_correction_module.md

# Vapor Correction Module

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Vapor Correction Module

> Auto-generated documentation for [particula.particles.properties.vapor_correction_module](https://github.com/uncscode/particula/blob/main/particula/particles/properties/vapor_correction_module.py) module.

## get_vapor_transition_correction

[Show source in vapor_correction_module.py:14](https://github.com/uncscode/particula/blob/main/particula/particles/properties/vapor_correction_module.py#L14)

Calculate the Fuchs–Sutugin vapor transition correction factor.

This correction factor (f) accounts for the transition regime between free
molecular flow and continuum diffusion when computing mass or heat
transport.

Mathematically:

- f(Kn, α) = [0.75·α·(1+Kn)] / [Kn² + Kn + 0.283·α·Kn + 0.75·α]
    - Kn is the Knudsen number (dimensionless),
    - α is the mass accommodation coefficient (dimensionless).

#### Arguments

- knudsen_number : Dimensionless Knudsen number.
- mass_accommodation : Mass accommodation coefficient (dimensionless).

#### Returns

- Transition correction factor (float or NDArray[np.float64]).

#### Examples

``` py title="Example"
import particula as par
par.particles.get_vapor_transition_correction(
    knudsen_number=0.1, mass_accommodation=1.0
)
# Output: 0.73...
```

#### References

- Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry
  and Physics, Ch. 12. Equation 12.43.
- Fuchs, N. A., & Sutugin, A. G. (1971). *High-Dispersed Aerosols*.
  In *Topics in Current Aerosol Research*, Elsevier, pp. 1–60.

#### Signature

```python
@validate_inputs({"knudsen_number": "nonnegative", "mass_accommodation": "nonnegative"})
def get_vapor_transition_correction(
    knudsen_number: Union[float, NDArray[np.float64]],
    mass_accommodation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```


---
# representation.md

# Representation

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Representation

> Auto-generated documentation for [particula.particles.representation](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py) module.

## ParticleRepresentation

[Show source in representation.py:22](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L22)

Everything needed to represent a particle or a collection of particles.

Represents a particle or a collection of particles, encapsulating the
strategy for calculating mass, radius, and total mass based on a
specified particle distribution, density, and concentration. This class
allows for flexibility in representing particles.

#### Attributes

- strategy : The computation strategy for particle representations.
- activity : The activity strategy for the partial pressure
    calculations.
- surface : The surface strategy for surface tension and Kelvin effect.
- distribution : The distribution data for the particles, which could
    represent sizes, masses, or another relevant metric.
- density : The density of the material from which the particles are
    made.
- concentration : The concentration of particles within the
    distribution.
- charge : The charge on each particle.
- volume : The air volume for simulation of particles in the air,
    default is 1 m^3. This is only used in ParticleResolved Strategies.

#### Methods

- get_strategy : Return the distribution strategy (optionally cloned).
- get_strategy_name : Return the name of the distribution strategy.
- get_activity : Return the activity strategy (optionally cloned).
- get_activity_name : Return the name of the activity strategy.
- get_surface : Return the surface strategy (optionally cloned).
- get_surface_name : Return the name of the surface strategy.
- get_distribution : Return the distribution array (optionally cloned).
- get_density : Return the density array (optionally cloned).
- get_concentration : Return the concentration array (optionally cloned).
- get_total_concentration : Return the total concentration (1/m^3).
- get_charge : Return the per-particle charge (optionally cloned).
- get_volume : Return the representation volume in m^3 (optionally cloned).
- get_species_mass : Return the mass per species, in kg.
- get_mass : Return the array of total particle masses, in kg.
- get_mass_concentration : Return the total mass concentration in kg/m^3.
- get_radius : Return the array of particle radii in meters.
- add_mass : Add mass to the distribution in each bin.
- add_concentration : Add concentration to the distribution in each bin.
- collide_pairs : Collide pairs of indices (ParticleResolved strategies).

#### Signature

```python
class ParticleRepresentation:
    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
        concentration: NDArray[np.float64],
        charge: NDArray[np.float64],
        volume: float = 1,
    ): ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)
- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)
- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)

### ParticleRepresentation().__str__

[Show source in representation.py:87](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L87)

Returns a string representation of the particle representation.

#### Returns

- A string representation of the particle representation.

#### Examples

``` py title="Get String Representation"
str_rep = str(particle_representation)
print(str_rep)
```

#### Signature

```python
def __str__(self) -> str: ...
```

### ParticleRepresentation().add_concentration

[Show source in representation.py:455](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L455)

Add concentration to the particle distribution.

#### Arguments

- added_concentration : The concentration to be added per bin
    (1/m^3).
- added_distribution : Optional distribution array to merge into
  the existing distribution. If None, the current distribution
  is reused.

#### Examples

``` py title="Add Concentration"
particle_representation.add_concentration(added_concentration)
```

#### Signature

```python
def add_concentration(
    self,
    added_concentration: NDArray[np.float64],
    added_distribution: Optional[NDArray[np.float64]] = None,
) -> None: ...
```

### ParticleRepresentation().add_mass

[Show source in representation.py:436](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L436)

Add mass to the particle distribution and update parameters.

#### Arguments

- added_mass : The mass to be added per distribution bin, in kg.

#### Examples

``` py title="Add Mass"
particle_representation.add_mass(added_mass)
```

#### Signature

```python
def add_mass(self, added_mass: NDArray[np.float64]) -> None: ...
```

### ParticleRepresentation().collide_pairs

[Show source in representation.py:490](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L490)

Collide pairs of particles, used for ParticleResolved Strategies.

#### Arguments

- indices : The indices of the particles to collide.

#### Examples

``` py title="Collide Pairs"
particle_representation.collide_pairs(indices)
```

#### Signature

```python
def collide_pairs(self, indices: NDArray[np.int64]) -> None: ...
```

### ParticleRepresentation().get_activity

[Show source in representation.py:145](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L145)

Return the activity strategy used for partial pressure calculations.

#### Arguments

- clone : If True, then return a deepcopy of the activity strategy.

#### Returns

- The activity strategy used for partial
  pressure calculations.

#### Examples

``` py title="Get Activity Strategy"
activity = particle_representation.get_activity()
```

#### Signature

```python
def get_activity(self, clone: bool = False) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)

### ParticleRepresentation().get_activity_name

[Show source in representation.py:165](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L165)

Return the name of the activity strategy used for partial pressure
calculations.

#### Returns

- The name of the activity strategy used for partial
  pressure calculations.

#### Examples

``` py title="Get Activity Strategy Name"
activity_name = particle_representation.get_activity_name()
print(activity_name)
```

#### Signature

```python
def get_activity_name(self) -> str: ...
```

### ParticleRepresentation().get_charge

[Show source in representation.py:299](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L299)

Return the charge per particle.

#### Arguments

- clone : If True, then return a copy of the charge array.

#### Returns

- The charge of the particles (dimensionless).

#### Examples

``` py title="Get Charge Array"
charge = particle_representation.get_charge()
```

#### Signature

```python
def get_charge(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_concentration

[Show source in representation.py:257](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L257)

Return the volume concentration of the particles.

For ParticleResolved Strategies, this is the number of
particles per self.volume. Otherwise, it's per 1/m^3.

#### Arguments

- clone : If True, then return a copy of the concentration array.

#### Returns

- The concentration of the particles in 1/m^3.

#### Examples

``` py title="Get Concentration Array"
concentration = particle_representation.get_concentration()
```

#### Signature

```python
def get_concentration(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_density

[Show source in representation.py:238](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L238)

Return the density of the particles.

#### Arguments

- clone : If True, then return a copy of the density array.

#### Returns

- The density of the particles.

#### Examples

``` py title="Get Density Array"
density = particle_representation.get_density()
```

#### Signature

```python
def get_density(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_distribution

[Show source in representation.py:219](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L219)

Return the distribution of the particles.

#### Arguments

- clone : If True, then return a copy of the distribution array.

#### Returns

- The distribution of the particles.

#### Examples

``` py title="Get Distribution Array"
distribution = particle_representation.get_distribution()
```

#### Signature

```python
def get_distribution(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_mass

[Show source in representation.py:358](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L358)

Return the mass of the particles as calculated by the strategy.

#### Arguments

- clone : If True, then return a copy of the mass array.

#### Returns

- The mass of the particles in kg.

#### Examples

``` py title="Get Mass"
mass = particle_representation.get_mass()
```

#### Signature

```python
def get_mass(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_mass_concentration

[Show source in representation.py:379](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L379)

Return the total mass per volume of the simulated particles.

The mass concentration is calculated from the distribution
and concentration arrays.

#### Arguments

- clone : If True, then return a copy of the mass concentration
  value.

#### Returns

- The mass concentration in kg/m^3.

#### Examples

``` py title="Get Mass Concentration"
mass_concentration = (
    particle_representation.get_mass_concentration()
)
print(mass_concentration)
```

#### Signature

```python
def get_mass_concentration(self, clone: bool = False) -> np.float64: ...
```

### ParticleRepresentation().get_radius

[Show source in representation.py:415](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L415)

Return the radius of the particles as calculated by the strategy.

#### Arguments

- clone : If True, then return a copy of the radius array.

#### Returns

- The radius of the particles in meters.

#### Examples

``` py title="Get Radius"
radius = particle_representation.get_radius()
```

#### Signature

```python
def get_radius(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_species_mass

[Show source in representation.py:337](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L337)

Return the masses per species in the particles.

#### Arguments

- clone : If True, then return a copy of the computed mass array.

#### Returns

- The mass of the particles per species in kg.

#### Examples

``` py title="Get Species Mass"
species_mass = particle_representation.get_species_mass()
```

#### Signature

```python
def get_species_mass(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_strategy

[Show source in representation.py:110](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L110)

Return the strategy used for particle representation.

#### Arguments

- clone : If True, then return a deepcopy of the strategy.

#### Returns

- The strategy used for particle
    representation.

#### Examples

``` py title="Get Strategy"
strategy = particle_representation.get_strategy()
```

#### Signature

```python
def get_strategy(self, clone: bool = False) -> DistributionStrategy: ...
```

#### See also

- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)

### ParticleRepresentation().get_strategy_name

[Show source in representation.py:130](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L130)

Return the name of the strategy used for particle representation.

#### Returns

- The name of the strategy used for particle representation.

#### Examples

``` py title="Get Strategy Name"
strategy_name = particle_representation.get_strategy_name()
print(strategy_name)
```

#### Signature

```python
def get_strategy_name(self) -> str: ...
```

### ParticleRepresentation().get_surface

[Show source in representation.py:182](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L182)

Return the surface strategy used for surface tension and Kelvin effect.

#### Arguments

- clone : If True, then return a deepcopy of the surface strategy.

#### Returns

- The surface strategy used for surface tension
  and Kelvin effect.

#### Examples

``` py title="Get Surface Strategy"
surface = particle_representation.get_surface()
```

#### Signature

```python
def get_surface(self, clone: bool = False) -> SurfaceStrategy: ...
```

#### See also

- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)

### ParticleRepresentation().get_surface_name

[Show source in representation.py:202](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L202)

Return the name of the surface strategy used for surface tension and
Kelvin effect.

#### Returns

- The name of the surface strategy used for surface tension
  and Kelvin effect.

#### Examples

``` py title="Get Surface Strategy Name"
surface_name = particle_representation.get_surface_name()
print(surface_name)
```

#### Signature

```python
def get_surface_name(self) -> str: ...
```

### ParticleRepresentation().get_total_concentration

[Show source in representation.py:279](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L279)

Return the total concentration of the particles.

#### Arguments

- clone : If True, then return a copy of the concentration array.

#### Returns

- The total number concentration of the particles in 1/m^3.

#### Examples

``` py title="Get Total Concentration"
total_concentration = (
    particle_representation.get_total_concentration()
)
print(total_concentration)
```

#### Signature

```python
def get_total_concentration(self, clone: bool = False) -> np.float64: ...
```

### ParticleRepresentation().get_volume

[Show source in representation.py:318](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py#L318)

Return the volume used for the particle representation.

#### Arguments

- clone : If True, then return a copy of the volume value.

#### Returns

- The volume of the particles in m^3.

#### Examples

``` py title="Get Volume"
volume = particle_representation.get_volume()
```

#### Signature

```python
def get_volume(self, clone: bool = False) -> float: ...
```


---
# representation_builders.md

# Representation Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Representation Builders

> Auto-generated documentation for [particula.particles.representation_builders](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py) module.

## BuilderActivityStrategyMixin

[Show source in representation_builders.py:126](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L126)

Mixin class for setting the activity_strategy attribute.

#### Methods

- set_activity_strategy : Assign the activity strategy (e.g., ideal mass,
    ideal molar, kappa-parameter).

#### Signature

```python
class BuilderActivityStrategyMixin:
    def __init__(self): ...
```

### BuilderActivityStrategyMixin().set_activity_strategy

[Show source in representation_builders.py:138](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L138)

Assign the activity strategy for the particle representation.

#### Arguments

- activity_strategy : An ActivityStrategy instance (e.g.,
  ActivityIdealMass, ActivityIdealMolar).
- activity_strategy_units : Not used (for interface consistency).

#### Returns

- self : For method chaining.

#### Signature

```python
def set_activity_strategy(
    self,
    activity_strategy: ActivityStrategy,
    activity_strategy_units: Optional[str] = None,
): ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)



## BuilderDistributionStrategyMixin

[Show source in representation_builders.py:88](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L88)

Mixin class for setting the distribution_strategy attribute.

#### Methods

- set_distribution_strategy : Assign the distribution strategy
    (e.g., mass-based, radii-based).

#### Signature

```python
class BuilderDistributionStrategyMixin:
    def __init__(self): ...
```

### BuilderDistributionStrategyMixin().set_distribution_strategy

[Show source in representation_builders.py:100](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L100)

Assign the distribution strategy for the particle representation.

#### Arguments

- distribution_strategy : A DistributionStrategy instance
  (e.g., mass-based bins, radius-based bins).
- distribution_strategy_units : Not used
    (for interface consistency).

#### Returns

- self : For method chaining.

#### Signature

```python
def set_distribution_strategy(
    self,
    distribution_strategy: DistributionStrategy,
    distribution_strategy_units: Optional[str] = None,
): ...
```

#### See also

- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)



## BuilderSurfaceStrategyMixin

[Show source in representation_builders.py:53](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L53)

Mixin class for setting the surface_strategy attribute.

#### Methods

- set_surface_strategy : Assign the surface strategy controlling
    surface tension or other surface-related properties.

#### Signature

```python
class BuilderSurfaceStrategyMixin:
    def __init__(self): ...
```

### BuilderSurfaceStrategyMixin().set_surface_strategy

[Show source in representation_builders.py:65](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L65)

Assign the surface strategy for the particle representation.

#### Arguments

- surface_strategy : A SurfaceStrategy instance defining surface
  tension or other surface-related properties.
- surface_strategy_units : Not used (for interface consistency).

#### Returns

- self : For method chaining.

#### Signature

```python
def set_surface_strategy(
    self, surface_strategy: SurfaceStrategy, surface_strategy_units: Optional[str] = None
): ...
```

#### See also

- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)



## ParticleMassRepresentationBuilder

[Show source in representation_builders.py:160](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L160)

Builder for ParticleRepresentation objects using mass-based distributions.

#### Attributes

- distribution_strategy : The DistributionStrategy
    (e.g., mass-based bins).
- activity_strategy : The ActivityStrategy (e.g., ideal mass).
- surface_strategy : The SurfaceStrategy
    (e.g., surface tension calculations).
- mass : The total or per-bin mass of particles in kg.
- density : The particle density in kg/m^3.
- concentration : The number concentration in 1/m^3.
- charge : Number of charges per particle (dimensionless).

#### Methods

- set_distribution_strategy : Assign the distribution strategy.
- set_activity_strategy : Assign the activity strategy.
- set_surface_strategy : Assign the surface strategy.
- set_mass : Assign the mass of the particles.
- set_density : Assign the density of the particles.
- set_concentration : Assign the number concentration.
- set_charge : Assign the charge of the particles.
- build : Validate and return a ParticleRepresentation with
    mass-based distribution data.

#### Signature

```python
class ParticleMassRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](#builderdistributionstrategymixin)
- [BuilderMassMixin](../builder_mixin.md#buildermassmixin)
- [BuilderSurfaceStrategyMixin](#buildersurfacestrategymixin)

### ParticleMassRepresentationBuilder().build

[Show source in representation_builders.py:215](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L215)

Validate all required parameters and return a ParticleRepresentation.

#### Returns

- ParticleRepresentation : An object configured to represent
  mass-based particle distributions, activity, and surface
  properties.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## ParticleRadiusRepresentationBuilder

[Show source in representation_builders.py:236](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L236)

Builder for ParticleRepresentation objects using radius-based
distributions.

#### Attributes

- distribution_strategy : The DistributionStrategy (e.g.,
    radius-based bins).
- activity_strategy : The ActivityStrategy (e.g., ideal mass).
- surface_strategy : The SurfaceStrategy (e.g., surface tension
    calculations).
- radius : The radius of the particles in meters.
- density : The particle density in kg/m^3.
- concentration : The number concentration in 1/m^3.
- charge : Number of charges per particle (dimensionless).

#### Methods

- set_distribution_strategy : Assign the distribution strategy.
- set_activity_strategy : Assign the activity strategy.
- set_surface_strategy : Assign the surface strategy.
- set_radius : Assign the radius of the particles.
- set_density : Assign the density of the particles.
- set_concentration : Assign the number concentration.
- set_charge : Assign the charge of the particles.
- build : Validate and return a ParticleRepresentation with
  radius-based distribution data.

#### Signature

```python
class ParticleRadiusRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](#builderdistributionstrategymixin)
- [BuilderRadiusMixin](../builder_mixin.md#builderradiusmixin)
- [BuilderSurfaceStrategyMixin](#buildersurfacestrategymixin)

### ParticleRadiusRepresentationBuilder().build

[Show source in representation_builders.py:292](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L292)

Validate all required parameters and return a ParticleRepresentation.

#### Returns

- ParticleRepresentation : An object configured to represent
  radius-based particle distributions, activity, and surface properties.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## PresetParticleRadiusBuilder

[Show source in representation_builders.py:312](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L312)

Builder for ParticleRepresentation objects with radius-based bins
generated from a lognormal size distribution.

#### Attributes

- mode : Mode(s) of the lognormal distribution in meters.
- geometric_standard_deviation : Geometric standard deviation(s).
- number_concentration : Number concentration(s) in 1/m^3.
- radius_bins : The array of radius bins in meters for the
    distribution.
- distribution_type : The type of lognormal distribution
    ("pdf" or "pmf").

#### Methods

- set_distribution_strategy : Assign the distribution strategy.
- set_activity_strategy : Assign the activity strategy.
- set_surface_strategy : Assign the surface strategy.
- set_radius_bins : Assign radius bin edges in meters.
- set_distribution_type : Choose between "pdf" or "pmf".
- build : Generate the distribution and return a ParticleRepresentation.

#### Signature

```python
class PresetParticleRadiusBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderConcentrationMixin](../builder_mixin.md#builderconcentrationmixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](#builderdistributionstrategymixin)
- [BuilderLognormalMixin](../builder_mixin.md#builderlognormalmixin)
- [BuilderRadiusMixin](../builder_mixin.md#builderradiusmixin)
- [BuilderSurfaceStrategyMixin](#buildersurfacestrategymixin)

### PresetParticleRadiusBuilder().build

[Show source in representation_builders.py:428](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L428)

Generate a lognormal distribution (PDF or PMF) based on
current parameters and return a ParticleRepresentation.

#### Returns

- ParticleRepresentation : An object with radius-based lognormal
  distribution, activity, and surface properties.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)

### PresetParticleRadiusBuilder().set_distribution_type

[Show source in representation_builders.py:403](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L403)

Choose between "pdf" (probability density function) or "pmf"
(probability mass function) for the distribution.

#### Arguments

- distribution_type : Must be either 'pdf' or 'pmf'.
- distribution_type_units : Not used (for interface consistency).

#### Returns

- self : For method chaining.

#### Signature

```python
def set_distribution_type(
    self, distribution_type: str, distribution_type_units: Optional[str] = None
): ...
```

### PresetParticleRadiusBuilder().set_radius_bins

[Show source in representation_builders.py:375](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L375)

Assign an array of radius bin edges.

#### Arguments

- radius_bins : The radius bin edges in meters.
- radius_bins_units : The units of the radius bins. Default is "m".

#### Returns

- self : For method chaining.

#### Signature

```python
@validate_inputs({"radius_bins": "positive"})
def set_radius_bins(
    self, radius_bins: NDArray[np.float64], radius_bins_units: str = "m"
): ...
```



## PresetResolvedParticleMassBuilder

[Show source in representation_builders.py:551](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L551)

Builder for ParticleRepresentation objects with preset parameters
for particle-resolved masses derived from a lognormal size distribution.

Generates random samples of particle radii (lognormal) and converts
them to per-particle masses. Includes defaults for mode, geometric
standard deviation, concentration, and total resolved count.

#### Attributes

- mode : Lognormal mode(s) in meters.
- geometric_standard_deviation : GSD(s).
- number_concentration : Number concentration(s) in 1/m^3.
- particle_resolved_count : Number of resolved particles.
- volume : Volume in m^3 for the representation.

#### Methods

- build : Sample radii from a lognormal distribution, convert to mass,
    and create a ParticleRepresentation.

#### Signature

```python
class PresetResolvedParticleMassBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
    BuilderVolumeMixin,
    BuilderParticleResolvedCountMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](#builderdistributionstrategymixin)
- [BuilderLognormalMixin](../builder_mixin.md#builderlognormalmixin)
- [BuilderParticleResolvedCountMixin](../builder_mixin.md#builderparticleresolvedcountmixin)
- [BuilderSurfaceStrategyMixin](#buildersurfacestrategymixin)
- [BuilderVolumeMixin](../builder_mixin.md#buildervolumemixin)

### PresetResolvedParticleMassBuilder().build

[Show source in representation_builders.py:614](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L614)

Sample particle radii from a lognormal distribution, convert
to mass, and return a ParticleRepresentation with resolved
per-particle masses.

#### Returns

- ParticleRepresentation : Configured for particle-resolved
  masses with the specified distribution, activity, and surface.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)



## ResolvedParticleMassRepresentationBuilder

[Show source in representation_builders.py:468](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L468)

Builder for ParticleRepresentation objects with fully resolved
particle masses (array-based).

Allows setting distribution strategy, mass, density, charge,
volume, etc. No default values are assumed.

#### Attributes

- distribution_strategy : Strategy for the resolved mass distribution.
- activity_strategy : Activity strategy (e.g., ideal mass).
- surface_strategy : Surface strategy (e.g., tension).
- mass : Per-particle or resolved mass in kg.
- density : Particle density in kg/m^3.
- charge : Number of charges (dimensionless).
- volume : Volume of simulation in m^3.

#### Methods

- set_distribution_strategy : Assign the distribution strategy.
- set_activity_strategy : Assign the activity strategy.
- set_surface_strategy : Assign the surface strategy.
- set_mass : Assign the mass of the particles.
- set_density : Assign the density of the particles.
- set_charge : Assign the charge of the particles.
- set_volume : Assign the volume of the particles.
- build : Validate all parameters and return
    a ParticleRepresentation with resolved masses.

#### Signature

```python
class ResolvedParticleMassRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderVolumeMixin,
    BuilderMassMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderActivityStrategyMixin](#builderactivitystrategymixin)
- [BuilderChargeMixin](../builder_mixin.md#builderchargemixin)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderDistributionStrategyMixin](#builderdistributionstrategymixin)
- [BuilderMassMixin](../builder_mixin.md#buildermassmixin)
- [BuilderSurfaceStrategyMixin](#buildersurfacestrategymixin)
- [BuilderVolumeMixin](../builder_mixin.md#buildervolumemixin)

### ResolvedParticleMassRepresentationBuilder().build

[Show source in representation_builders.py:525](https://github.com/uncscode/particula/blob/main/particula/particles/representation_builders.py#L525)

Validate attributes and construct a ParticleRepresentation
with array-based, resolved masses.

#### Returns

- ParticleRepresentation : Configured with the specified
  distribution, activity, and surface strategies.

#### Signature

```python
def build(self) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](./representation.md#particlerepresentation)


---
# representation_factories.md

# Representation Factories

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Representation Factories

> Auto-generated documentation for [particula.particles.representation_factories](https://github.com/uncscode/particula/blob/main/particula/particles/representation_factories.py) module.

## ParticleRepresentationFactory

[Show source in representation_factories.py:20](https://github.com/uncscode/particula/blob/main/particula/particles/representation_factories.py#L20)

Factory for creating particle representation builders.

#### Methods

- get_builders : Return a dictionary of strategy builder instances.
- get_strategy : Obtain a ParticleRepresentation from a chosen builder.

#### Examples

```py title="Factory Usage Example"
import particula as par
factory = par.particles.ParticleRepresentationFactory()
rep = factory.get_strategy("mass")
# rep -> ParticleRepresentation with mass-based distribution
```

#### References

- Builder Pattern,
  [Wikipedia](https://en.wikipedia.org/wiki/Builder_pattern).

#### Signature

```python
class ParticleRepresentationFactory(
    StrategyFactoryABC[
        Union[
            ParticleMassRepresentationBuilder,
            ParticleRadiusRepresentationBuilder,
            PresetParticleRadiusBuilder,
            ResolvedParticleMassRepresentationBuilder,
            PresetResolvedParticleMassBuilder,
        ],
        ParticleRepresentation,
    ]
): ...
```

#### See also

- [ParticleMassRepresentationBuilder](./representation_builders.md#particlemassrepresentationbuilder)
- [ParticleRadiusRepresentationBuilder](./representation_builders.md#particleradiusrepresentationbuilder)
- [ParticleRepresentation](./representation.md#particlerepresentation)
- [PresetParticleRadiusBuilder](./representation_builders.md#presetparticleradiusbuilder)
- [PresetResolvedParticleMassBuilder](./representation_builders.md#presetresolvedparticlemassbuilder)
- [ResolvedParticleMassRepresentationBuilder](./representation_builders.md#resolvedparticlemassrepresentationbuilder)

### ParticleRepresentationFactory().get_builders

[Show source in representation_factories.py:53](https://github.com/uncscode/particula/blob/main/particula/particles/representation_factories.py#L53)

Return a mapping of strategy types to builder instances.

#### Returns

- A dictionary where each key is a strategy type
  ("mass", "radius", etc.) and each value is the
  corresponding builder instance.

#### Examples

```py title="get_builders usage"
import particula as par
factory = par.particles.ParticleRepresentationFactory()
builders = factory.get_builders()
mass_builder = builders["mass"]
# mass_builder -> ParticleMassRepresentationBuilder()
```

#### Signature

```python
def get_builders(self): ...
```


---
# surface_builders.md

# Surface Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Surface Builders

> Auto-generated documentation for [particula.particles.surface_builders](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py) module.

## SurfaceStrategyMassBuilder

[Show source in surface_builders.py:66](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L66)

Builder class for SurfaceStrategyMass objects.

For calculating the Kelvin effect based on mass mixing. Needed for
the effective surface tension calculation.

#### Methods

- set_surface_tension : Set the surface tension in N/m.
- set_density : Set the density in kg/m^3.
- set_parameters : Configure multiple parameters at once.
- build : Validate parameters and return the strategy.

#### Signature

```python
class SurfaceStrategyMassBuilder(
    BuilderABC, BuilderSurfaceTensionMixin, BuilderDensityMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderSurfaceTensionMixin](../builder_mixin.md#buildersurfacetensionmixin)

### SurfaceStrategyMassBuilder().build

[Show source in surface_builders.py:87](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L87)

Validate and return the SurfaceStrategyMass object.

#### Returns

- `SurfaceStrategyMass` - Instance of the SurfaceStrategyMass object.

#### Signature

```python
def build(self) -> SurfaceStrategyMass: ...
```

#### See also

- [SurfaceStrategyMass](./surface_strategies.md#surfacestrategymass)



## SurfaceStrategyMolarBuilder

[Show source in surface_builders.py:26](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L26)

Builder class for SurfaceStrategyMolar objects.

For calculating the Kelvin effect based on molar mass. Needed for
the effective surface tension calculation.

#### Methods

- set_surface_tension : Set the surface tension in N/m.
- set_density : Set the density in kg/m^3.
- set_molar_mass : Set the molar mass in kg/mol.
- set_parameters : Configure multiple parameters at once.
- build : Validate parameters and return the strategy.

#### Signature

```python
class SurfaceStrategyMolarBuilder(
    BuilderABC, BuilderDensityMixin, BuilderSurfaceTensionMixin, BuilderMolarMassMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderMolarMassMixin](../builder_mixin.md#buildermolarmassmixin)
- [BuilderSurfaceTensionMixin](../builder_mixin.md#buildersurfacetensionmixin)

### SurfaceStrategyMolarBuilder().build

[Show source in surface_builders.py:52](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L52)

Validate and return the SurfaceStrategyMolar object.

#### Returns

- `SurfaceStrategyMolar` - Instance of the SurfaceStrategyMolar object.

#### Signature

```python
def build(self) -> SurfaceStrategyMolar: ...
```

#### See also

- [SurfaceStrategyMolar](./surface_strategies.md#surfacestrategymolar)



## SurfaceStrategyVolumeBuilder

[Show source in surface_builders.py:100](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L100)

Builder class for SurfaceStrategyVolume objects.

For calculating the Kelvin effect based on volume mixing. Needed for
the effective surface tension calculation.

#### Methods

- set_surface_tension : Set the surface tension in N/m.
- set_density : Set the density in kg/m^3.
- set_parameters : Configure multiple parameters at once.
- build : Validate parameters and return the strategy.

#### Signature

```python
class SurfaceStrategyVolumeBuilder(
    BuilderABC, BuilderSurfaceTensionMixin, BuilderDensityMixin
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderDensityMixin](../builder_mixin.md#builderdensitymixin)
- [BuilderSurfaceTensionMixin](../builder_mixin.md#buildersurfacetensionmixin)

### SurfaceStrategyVolumeBuilder().build

[Show source in surface_builders.py:121](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L121)

Validate and return the SurfaceStrategyVolume object.

#### Returns

- `SurfaceStrategyVolume` - Instance of the SurfaceStrategyVolume
    object.

#### Signature

```python
def build(self) -> SurfaceStrategyVolume: ...
```

#### See also

- [SurfaceStrategyVolume](./surface_strategies.md#surfacestrategyvolume)


---
# surface_factories.md

# Surface Factories

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Surface Factories

> Auto-generated documentation for [particula.particles.surface_factories](https://github.com/uncscode/particula/blob/main/particula/particles/surface_factories.py) module.

## SurfaceFactory

[Show source in surface_factories.py:17](https://github.com/uncscode/particula/blob/main/particula/particles/surface_factories.py#L17)

Factory for creating surface tension strategy builders.

Creates builder instances for volume-, mass-, or molar-based
surface tension strategies. These strategies calculate surface
tension and the Kelvin effect for species in particulate phases.

#### Methods

- get_builders : Return a mapping of strategy types to builder
    instances.
- get_strategy : Return a strategy instance for the specified type
    ('volume', 'mass', or 'molar').

#### Returns

- SurfaceStrategy : The instance of the requested surface strategy.

#### Raises

- ValueError : If an unknown strategy type is provided or if
  required parameters are missing during check_keys/pre_build_check.

#### Signature

```python
class SurfaceFactory(
    StrategyFactoryABC[
        Union[
            SurfaceStrategyVolumeBuilder,
            SurfaceStrategyMassBuilder,
            SurfaceStrategyMolarBuilder,
        ],
        Union[SurfaceStrategyVolume, SurfaceStrategyMass, SurfaceStrategyMolar],
    ]
): ...
```

#### See also

- [SurfaceStrategyMassBuilder](./surface_builders.md#surfacestrategymassbuilder)
- [SurfaceStrategyMass](./surface_strategies.md#surfacestrategymass)
- [SurfaceStrategyMolarBuilder](./surface_builders.md#surfacestrategymolarbuilder)
- [SurfaceStrategyMolar](./surface_strategies.md#surfacestrategymolar)
- [SurfaceStrategyVolumeBuilder](./surface_builders.md#surfacestrategyvolumebuilder)
- [SurfaceStrategyVolume](./surface_strategies.md#surfacestrategyvolume)

### SurfaceFactory().get_builders

[Show source in surface_factories.py:49](https://github.com/uncscode/particula/blob/main/particula/particles/surface_factories.py#L49)

Return a mapping of strategy types to builder instances.

#### Returns

- Keys are 'volume', 'mass', or 'molar', each
  mapped to the corresponding builder class.

#### Examples

```py title="SurfaceFactory Example"
import particula as par
factory = par.particles.SurfaceFactory()
builders = factory.get_builders()
volume_builder = builders["volume"]
mass_builder = builders["mass"]
molar_builder = builders["molar"]
```

#### Signature

```python
def get_builders(self): ...
```


---
# surface_strategies.md

# Surface Strategies

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Particles](./index.md#particles) / Surface Strategies

> Auto-generated documentation for [particula.particles.surface_strategies](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py) module.

## SurfaceStrategy

[Show source in surface_strategies.py:24](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L24)

Abstract base class for surface strategies.

Implements methods for calculating effective surface tension, density,
and the Kelvin effect in particulate phases.

#### Methods

- effective_surface_tension : Calculate the effective surface tension.
- effective_density : Calculate the effective density.
- get_name : Return the type of the surface strategy.
- kelvin_radius : Calculate the Kelvin radius for curvature effects.
- kelvin_term : Calculate the exponential Kelvin term for vapor pressure.

#### Signature

```python
class SurfaceStrategy(ABC): ...
```

### SurfaceStrategy().effective_density

[Show source in surface_strategies.py:53](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L53)

Calculate the effective density of the species mixture.

#### Arguments

- mass_concentration : Concentration of the species in kg/m^3.

#### Returns

- Effective density in kg/m^3.

#### Signature

```python
@abstractmethod
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategy().effective_surface_tension

[Show source in surface_strategies.py:39](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L39)

Calculate the effective surface tension of the species mixture.

#### Arguments

- mass_concentration : Concentration of the species in kg/m^3.

#### Returns

- Effective surface tension in N/m.

#### Signature

```python
@abstractmethod
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategy().get_name

[Show source in surface_strategies.py:67](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L67)

Return the type of the surface strategy.

#### Signature

```python
def get_name(self) -> str: ...
```

### SurfaceStrategy().kelvin_radius

[Show source in surface_strategies.py:71](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L71)

Calculate the Kelvin radius, which sets the curvature effect on vapor pressure.

#### Arguments

- molar_mass : Molar mass of the species in kg/mol.
- mass_concentration : Concentration of the species in kg/m^3.
- temperature : Temperature of the system in K.

#### Returns

- Kelvin radius in meters.

#### References

- r = 2 × surface_tension × molar_mass / (R × T × density)
  [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

#### Signature

```python
def kelvin_radius(
    self,
    molar_mass: Union[float, NDArray[np.float64]],
    mass_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]: ...
```

### SurfaceStrategy().kelvin_term

[Show source in surface_strategies.py:99](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L99)

Calculate the exponential Kelvin term that adjusts vapor pressure.

#### Arguments

- radius : Particle radius in meters.
- molar_mass : Molar mass of the species in kg/mol.
- mass_concentration : Concentration of the species in kg/m^3.
- temperature : Temperature of the system in K.

#### Returns

- Factor by which vapor pressure is increased.

#### References

- P_eff = P_sat × exp(kelvin_radius / particle_radius)
  [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

#### Signature

```python
def kelvin_term(
    self,
    radius: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    mass_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## SurfaceStrategyMass

[Show source in surface_strategies.py:179](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L179)

Surface tension and density based on mass-fraction weighting.

#### Attributes

- surface_tension : Surface tension array or scalar in N/m.
- density : Density array or scalar in kg/m^3.

#### References

- [Mass Fraction](https://en.wikipedia.org/wiki/Mass_fraction_(chemistry))

#### Signature

```python
class SurfaceStrategyMass(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyMass().effective_density

[Show source in surface_strategies.py:211](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L211)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategyMass().effective_surface_tension

[Show source in surface_strategies.py:199](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L199)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```



## SurfaceStrategyMolar

[Show source in surface_strategies.py:129](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L129)

Surface tension and density based on mole-fraction weighting.

#### Attributes

- surface_tension : Surface tension array or scalar in N/m.
- density : Density array or scalar in kg/m^3.
- molar_mass : Molar mass array or scalar in kg/mol.

#### References

- [Mole Fraction](https://en.wikipedia.org/wiki/Mole_fraction)

#### Signature

```python
class SurfaceStrategyMolar(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyMolar().effective_density

[Show source in surface_strategies.py:165](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L165)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategyMolar().effective_surface_tension

[Show source in surface_strategies.py:152](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L152)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```



## SurfaceStrategyVolume

[Show source in surface_strategies.py:222](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L222)

Surface tension and density based on volume-fraction weighting.

#### Attributes

- surface_tension : Surface tension array or scalar in N/m.
- density : Density array or scalar in kg/m^3.

#### References

- [Volume Fraction](https://en.wikipedia.org/wiki/Volume_fraction)

#### Signature

```python
class SurfaceStrategyVolume(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyVolume().effective_density

[Show source in surface_strategies.py:255](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L255)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategyVolume().effective_surface_tension

[Show source in surface_strategies.py:242](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L242)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```


---
# runnable.md

# Runnable

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Runnable

> Auto-generated documentation for [particula.runnable](https://github.com/uncscode/particula/blob/main/particula/runnable.py) module.

## Runnable

[Show source in runnable.py:27](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L27)

Abstract base class for processes modifying an Aerosol instance.

This class enforces the implementation of a process rate calculation
and an execution method that modifies the Aerosol in-place. Subclasses
must implement both 'rate' and 'execute', which define how the process
affects the Aerosol over a time step.

#### Methods

- `-` *rate* - Calculate and return the rate of the process.
- `-` *execute* - Apply the process logic to the Aerosol over a specified
    time.
- `-` *__or__* - Chain two processes using the '|' operator.

#### Examples

```py title="Defining a Custom Process"
class CustomProcess(Runnable):
    def rate(self, aerosol):
        return 42

    def execute(self, aerosol, time_step, sub_steps=1):
        # Modify aerosol here
        return aerosol
```

#### References

- No references available yet.

#### Signature

```python
class Runnable(ABC): ...
```

### Runnable().__or__

[Show source in runnable.py:102](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L102)

Chain this Runnable with another using the '|' operator.

This method enables an easy way to sequence processes, so they
can be executed in a defined order.

#### Arguments

- other : Another Runnable to append after this one.

#### Returns

- RunnableSequence : A sequence containing both processes.

#### Examples

```py title="Chaining two processes"
combined_process = process1 | process2
final_aerosol = combined_process.execute(aerosol, time_step=1.0)
```

#### Signature

```python
def __or__(self, other: "Runnable"): ...
```

### Runnable().execute

[Show source in runnable.py:76](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L76)

Execute the process, modifying the Aerosol in-place over a time step.

#### Arguments

- aerosol : The Aerosol instance to be updated.
- time_step : The time step size in seconds.
- sub_steps : Number of sub-steps to subdivide the time step,
    default 1.

#### Returns

- The updated Aerosol after this process runs.

#### Examples

```py title="Executing the process"
process = CustomProcess()
updated_aerosol = process.execute(my_aerosol, time_step=1.0)
```

#### Signature

```python
@abstractmethod
def execute(self, aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)

### Runnable().rate

[Show source in runnable.py:57](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L57)

Calculate and return the rate of this process for the given Aerosol.

#### Arguments

- aerosol : The Aerosol instance on which to calculate the rate.

#### Returns

- Any : The computed rate of this process.

#### Examples

```py title="Using the rate method"
process = CustomProcess()
process_rate = process.rate(my_aerosol)
print(process_rate)
```

#### Signature

```python
@abstractmethod
def rate(self, aerosol: Aerosol) -> Any: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)



## RunnableSequence

[Show source in runnable.py:128](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L128)

A sequence of Runnable processes executed in order.

This class maintains a list of processes to be applied sequentially
to an Aerosol. Each process modifies the Aerosol and passes it along
to the next in the sequence.

#### Attributes

- processes : A list of Runnable objects forming the sequence.

#### Methods

- `-` *add_process* - Add a Runnable to the sequence.
- `-` *execute* - Apply each Runnable in the sequence to an Aerosol.
- `-` *__or__* - Chain a new Runnable into this sequence.

#### Examples

```py title="Building and running a RunnableSequence"
sequence = RunnableSequence()
sequence.add_process(CustomProcess())
sequence.add_process(AnotherProcess())
final_aerosol = sequence.execute(aerosol, time_step=2.0)
```

#### Signature

```python
class RunnableSequence:
    def __init__(self): ...
```

### RunnableSequence().__or__

[Show source in runnable.py:196](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L196)

Chain another Runnable into this sequence using the '|' operator.

#### Arguments

- process : The Runnable to add.

#### Returns

- RunnableSequence : This sequence with the new Runnable appended.

#### Examples

```py
sequence = RunnableSequence()
sequence |= CustomProcess()
# or
sequence = sequence | AnotherProcess()
```

#### Signature

```python
def __or__(self, process: Runnable): ...
```

#### See also

- [Runnable](#runnable)

### RunnableSequence().add_process

[Show source in runnable.py:156](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L156)

Add a Runnable to the sequence.

#### Arguments

- process : The Runnable to add.

#### Examples

```py
sequence = RunnableSequence()
sequence.add_process(CustomProcess())
```

#### Signature

```python
def add_process(self, process: Runnable): ...
```

#### See also

- [Runnable](#runnable)

### RunnableSequence().execute

[Show source in runnable.py:171](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171)

Execute all processes in the sequence on the given Aerosol.

Each Runnable in the sequence modifies the Aerosol and passes
it to the next Runnable until all have been executed.

#### Arguments

- aerosol : The Aerosol instance to be updated.
- time_step : The time step size in seconds for each process.

#### Returns

- Aerosol : The resulting Aerosol after all processes run.

#### Examples

```py title="Executing a RunnableSequence"
sequence = RunnableSequence()
final_aerosol = sequence.execute(aerosol, time_step=1.0)
```

#### Signature

```python
def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)


---
# arbitrary_round.md

# Arbitrary Round

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Arbitrary Round

> Auto-generated documentation for [particula.util.arbitrary_round](https://github.com/uncscode/particula/blob/main/particula/util/arbitrary_round.py) module.

## get_arbitrary_round

[Show source in arbitrary_round.py:15](https://github.com/uncscode/particula/blob/main/particula/util/arbitrary_round.py#L15)

Round values to the nearest multiple of a specified base.

The function supports "round", "floor", or "ceil" modes, and can retain
original nonzero values if rounding returns zero.

#### Arguments

- values : The values to be rounded.
- base : Positive float indicating the rounding interval.
- mode : Rounding mode, one of ['round', 'floor', 'ceil'].
- nonzero_edge : If True, zeros after rounding are replaced with the
    original values.

#### Returns

- The input values rounded according to the specified base and mode.

#### Examples

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([1.2, 2.5, 3.7, 4.0])
print(par.get_arbitrary_round(arr, base=1.0, mode='round'))
# Output: [1.  2.  4.  4.]

print(par.get_arbitrary_round(arr, base=0.5, mode='floor'))
# Output: [1.  2.  3.5 4. ]

print(par.get_arbitrary_round(2.5, base=1.0, mode='round'))
# Output: 2.0
```

#### References

- "Rounding," Python Documentation, docs.python.org.
- "NumPy Rounding," NumPy Documentation, NumPy.org.

#### Signature

```python
def get_arbitrary_round(
    values: Union[float, list[float], np.ndarray],
    base: Union[float, np.float64] = 1.0,
    mode: str = "round",
    nonzero_edge: bool = False,
) -> Union[float, NDArray[np.float64]]: ...
```


---
# colors.md

# Colors

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Colors

> Auto-generated documentation for [particula.util.colors](https://github.com/uncscode/particula/blob/main/particula/util/colors.py) module.


---
# constants.md

# Constants

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Constants

> Auto-generated documentation for [particula.util.constants](https://github.com/uncscode/particula/blob/main/particula/util/constants.py) module.

#### Attributes

- `GAS_CONSTANT` - Gas constant in J mol^-1 K^-1 = m^2 kg mol^-1 s^-2 K^-1
  J = kg m^2 s^-2: BOLTZMANN_CONSTANT * AVOGADRO_NUMBER

- `RELATIVE_PERMITTIVITY_AIR_ROOM` - Relative permittivity of air at approx.
  296.15 K and 101325 Pa and 40% RH
  See https://www.osti.gov/servlets/purl/1504063
  Previously known as the "dielectric constant"
  Often denoted as epsilon: 1.000530569

- `RELATIVE_PERMITTIVITY_AIR_STP` - At STP (273.15 K, 1 atm):
  see: https://en.wikipedia.org/wiki/Relative_permittivity: 1.00058986

- `RELATIVE_PERMITTIVITY_AIR` - select one of the two:: RELATIVE_PERMITTIVITY_AIR_ROOM

- `VACUUM_PERMITTIVITY` - Permittivity of free space in F/m
  Also known as the electric constant, permittivity of free space
  Often denoted by epsilon_0: scipy.constants.epsilon_0

- `REF_VISCOSITY_AIR_STP` - These values are used to calculate the dynamic viscosity of air
  Here, REF temperature and viscosity are at STP:
  Standard temperature and pressure (273.15 K and 101325 Pa): 1.716e-05


---
# convert_dtypes.md

# Convert Dtypes

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Convert Dtypes

> Auto-generated documentation for [particula.util.convert_dtypes](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py) module.

## get_coerced_type

[Show source in convert_dtypes.py:16](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L16)

Coerce the given data to the specified dtype if it is not already of that
type.

#### Arguments

- data : The data to be coerced (any type).
- dtype : The desired data type, e.g. float, int, or np.ndarray.

#### Returns

- The data converted to the specified type.

#### Raises

- ValueError : If the data cannot be coerced to the desired dtype.

#### Examples

``` py title="Coerce integer to float"
import particula as par
x = par.get_coerced_type(1, float)
print(x)
# 1.0
```

``` py title="Coerce list to numpy array"
import numpy as np
import particula as par
arr = par.get_coerced_type([1, 2, 3], np.ndarray)
print(arr)
# [1 2 3]
```

#### References

- NumPy Documentation: https://numpy.org/doc/

#### Signature

```python
def get_coerced_type(data, dtype): ...
```



## get_dict_from_list

[Show source in convert_dtypes.py:58](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L58)

Convert a list of strings into a dictionary mapping each string to its
index.

#### Arguments

- list_of_str : A non-empty list of strings.

#### Returns

- A dict where keys are the strings and values are their indices.

#### Raises

- AssertionError : If the list is empty or contains non-string items.

#### Examples

``` py title="Convert list of strings to dictionary"
import particula as par

str_list = ["alpha", "beta", "gamma"]
mapping = par.get_dict_from_list(str_list)
print(mapping)
# {'alpha': 0, 'beta': 1, 'gamma': 2}
```

#### Signature

```python
def get_dict_from_list(list_of_str: list) -> dict: ...
```



## get_shape_check

[Show source in convert_dtypes.py:130](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L130)

Validate or reshape a data array to ensure compatibility with a time array
and header list.

If data is 2D, the function attempts to align the time dimension with one
of the axes. If data is 1D, the header list must have exactly one entry.

#### Arguments

- time : 1D array of time values.
- data : 1D or 2D array of data values.
- header : List of headers corresponding to the data dimensions.

#### Returns

- A possibly reshaped data array ensuring alignment with time and
  header constraints.

#### Raises

- ValueError : If the header length does not match the data shape,
  or if data is 1D but header has more than one entry.

#### Examples

``` py
import numpy as np
import particula as par
time_array = np.arange(0, 10)
data_2d = np.random.rand(10, 5)
headers = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']
reshaped_data = par.get_shape_check(time_array, data_2d, headers)
print(reshaped_data.shape)
# Should be (10, 5)
```

#### Signature

```python
def get_shape_check(time: np.ndarray, data: np.ndarray, header: list) -> np.ndarray: ...
```



## get_values_of_dict

[Show source in convert_dtypes.py:93](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L93)

Retrieve a list of index values from a dictionary for the specified keys.

#### Arguments

- key_list : The keys to look up in the dictionary.
- dict_to_check : The dictionary from which values are retrieved.

#### Returns

- A list of values corresponding to the given keys.

#### Raises

- KeyError : If any key in key_list is not found in dict_to_check.

#### Examples

``` py
import particula as par
my_dict = {'a': 1, 'b': 2, 'c': 3}
vals = par.get_values_of_dict(['a', 'c'], my_dict)
print(vals)
# [1, 3]
```

#### Signature

```python
def get_values_of_dict(
    key_list: List[str], dict_to_check: Dict[str, Any]
) -> List[Any]: ...
```


---
# convert_units.md

# Convert Units

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Convert Units

> Auto-generated documentation for [particula.util.convert_units](https://github.com/uncscode/particula/blob/main/particula/util/convert_units.py) module.

## get_unit_conversion

[Show source in convert_units.py:29](https://github.com/uncscode/particula/blob/main/particula/util/convert_units.py#L29)

Convert a numeric value or unit expression from one unit to another using
Pint.

For simple multiplicative units, if no value is provided, this function
returns the conversion factor. For units with an offset
(e.g., temperatures), or if a value is supplied, a fully converted
numeric value is returned instead.

#### Arguments

- old : A string representing the current unit (e.g., "m", "degC").
- new : A string representing the target unit.
- value : An optional numeric value to convert. If omitted, returns the
    conversion factor between old and new.

#### Raises

- ImportError : If Pint is not installed. Install it using:
    `pip install pint`.

#### Returns

- A float representing either the conversion factor or the fully
  converted value in the target unit.

#### Examples

``` py title="Example Multi-Unit Conversion"
import particula as par
factor = par.get_unit_conversion("ug/m^3", "kg/m^3")
print(factor)
# 1e-9
```

``` py title="Example Temperature Conversion"
import particula as par
degF = par.get_unit_conversion("degC", "degF", value=25)
print(degF)
# ~77.0
```

#### References

- Pint documentation: https://pint.readthedocs.io/

#### Signature

```python
def get_unit_conversion(old: str, new: str, value: Optional[float] = None) -> float: ...
```


---
# src_lf2013_coagulation.md

# Src Lf2013 Coagulation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Util](../index.md#util) / [Lf2013 Coagulation](./index.md#lf2013-coagulation) / Src Lf2013 Coagulation

> Auto-generated documentation for [particula.util.lf2013_coagulation.src_lf2013_coagulation](https://github.com/uncscode/particula/blob/main/particula/util/lf2013_coagulation/src_lf2013_coagulation.py) module.

## lf2013_coag_full

[Show source in src_lf2013_coagulation.py:13](https://github.com/uncscode/particula/blob/main/particula/util/lf2013_coagulation/src_lf2013_coagulation.py#L13)

calculate ion--particle coagulation according to lf2013

#### Signature

```python
def lf2013_coag_full(
    ion_type="air",
    particle_type="conductive",
    temperature_val=298.15,
    pressure_val=101325,
    charge_vals=None,
    radius_vals=None,
): ...
```


---
# machine_limit.md

# Machine Limit

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Machine Limit

> Auto-generated documentation for [particula.util.machine_limit](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py) module.

## get_safe_exp

[Show source in machine_limit.py:13](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L13)

Compute the exponential of each element in the input array, with overflow
protection.

The exponential is calculated using:
    - y = exp(x), where x is clipped to avoid exceeding machine limits.

#### Arguments

- value : Array-like of values to exponentiate.

#### Returns

- np.ndarray of exponentiated values, with machine-level clipping.

#### Examples

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([0, 10, 1000])
print(par.get_safe_exp(arr))
# Output: [1.00000000e+000 2.20264658e+004 1.79769313e+308]
```

#### References

- "Floating Point Arithmetic," NumPy Documentation, NumPy.org.

#### Signature

```python
def get_safe_exp(value: ArrayLike) -> np.ndarray: ...
```



## get_safe_log

[Show source in machine_limit.py:45](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L45)

Compute the natural logarithm of each element in the input array, with
underflow protection.

The natural log is calculated using:
    - y = ln(x), where x is clipped away from zero to maintain positivity.

#### Arguments

- value : Array-like of values for logarithm calculation.

#### Returns

- np.ndarray of natural logarithms, with machine-level clipping.

#### Examples

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([1e-320, 1.0, 10.0])
print(get_safe_log(arr))
# Output: [-7.40545337e+02  0.00000000e+00  2.30258509e+00]
```

#### References

- "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.

#### Signature

```python
def get_safe_log(value: ArrayLike) -> np.ndarray: ...
```



## get_safe_log10

[Show source in machine_limit.py:77](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L77)

Compute the base-10 logarithm of each element in the input array, with
underflow protection.

The base-10 log is calculated using:
    - y = log10(x), where x is clipped away from zero to maintain positivity.

#### Arguments

- value : Array-like of values for base-10 logarithm calculation.

#### Returns

- np.ndarray of base-10 logarithms, with machine-level clipping.

#### Examples

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([1e-320, 1.0, 1000.0])
print(par.get_safe_log10(arr))
# Output: [-320.           0.           3.        ]
```

#### References

- "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.

#### Signature

```python
def get_safe_log10(value: ArrayLike) -> np.ndarray: ...
```



## get_safe_power

[Show source in machine_limit.py:109](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L109)

Compute the power (base ** exponent) with overflow protection.

The power is computed as: result = exp(exponent * log(base))
where the intermediate value is clipped to avoid overflow beyond the
machine limits. This function assumes that `base` contains positive values.
The behavior for non-positive bases is undefined.

#### Arguments

- base : Array-like of positive base values.
- exponent : Array-like of exponents.

#### Returns

- np.ndarray of power values, computed with machine-level clipping.

#### Examples

``` py title="Example Usage"
import numpy as np
import particula as par

base = np.array([1, 2, 3])
exponent = np.array([1, 2, 3])
print(par.get_safe_power(base, exponent))
# Output: [ 1.  4. 27.]
```

#### References

- "Floating Point Arithmetic," NumPy Documentation, NumPy.org.

#### Signature

```python
@validate_inputs({"base": "positive"})
def get_safe_power(base: ArrayLike, exponent: ArrayLike) -> np.ndarray: ...
```


---
# reduced_quantity.md

# Reduced Quantity

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Reduced Quantity

> Auto-generated documentation for [particula.util.reduced_quantity](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py) module.

## get_reduced_self_broadcast

[Show source in reduced_quantity.py:71](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L71)

Return a square matrix of pairwise reduced values using a single array.

Each element is calculated by broadcasting the array with its transpose:
- r_ij = (α_i × α_j) / (α_i + α_j),
    - r_ij is the reduced quantity between α_i and α_j.

#### Arguments

- alpha_array : A 1D array for pairwise reduced value calculations.

#### Returns

- A 2D square matrix of pairwise reduced values.

#### Examples

``` py title="Example"
from particula.util.reduced_quantity import get_reduced_self_broadcast
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
print(get_reduced_self_broadcast(arr))
# Output: [[0.5       0.6666667 0.75     ]
#          [0.6666667 1.        1.2      ]
#          [0.75      1.2       1.5      ]]
```

#### References

- [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)

#### Signature

```python
def get_reduced_self_broadcast(
    alpha_array: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## get_reduced_value

[Show source in reduced_quantity.py:15](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L15)

Return the reduced value of two parameters.

The reduced value is computed using:
- r = (α × β) / (α + β),
    - r is the reduced quantity,
    - α, β are the input parameters.

#### Arguments

- alpha : The first parameter (scalar or array).
- beta : The second parameter (scalar or array).

#### Returns

- The element-wise reduced quantity, zero if (α+β)=0.

#### Raises

- ValueError : If arrays have incompatible shapes.

#### Examples

``` py title="Example"
from particula.util.reduced_quantity import get_reduced_value
import numpy as np

print(get_reduced_value(3.0, 6.0))
# Output: 2.0

arrA = np.array([1.0, 2.0, 3.0])
arrB = np.array([2.0, 5.0, 10.0])
print(get_reduced_value(arrA, arrB))
# Output: [0.666..., 1.428..., 2.142...]
```

#### References

- [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)

#### Signature

```python
def get_reduced_value(
    alpha: Union[float, NDArray[np.float64]], beta: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```


---
# refractive_index_mixing.md

# Refractive Index Mixing

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Refractive Index Mixing

> Auto-generated documentation for [particula.util.refractive_index_mixing](https://github.com/uncscode/particula/blob/main/particula/util/refractive_index_mixing.py) module.

## get_effective_refractive_index

[Show source in refractive_index_mixing.py:11](https://github.com/uncscode/particula/blob/main/particula/util/refractive_index_mixing.py#L11)

Calculate the effective refractive index of a two-solute mixture.

The calculation uses volume-weighted molar refraction, described by:
- r_eff = (v0 / (v0 + v1)) * ((m0 - 1) / (m0 + 2)) +
          (v1 / (v0 + v1)) * ((m1 - 1) / (m1 + 2))
    - r_eff is the effective molar refraction,
    - m0, m1 are the refractive indices of each solute,
    - v0, v1 are the volumes of each solute.

Then the resulting refractive index is:
- n_eff = (2 × r_eff + 1) / (1 - r_eff).

#### Arguments

- m_zero : Refractive index of solute 0 (float or complex).
- m_one : Refractive index of solute 1 (float or complex).
- volume_zero : Volume of solute 0.
- volume_one : Volume of solute 1.

#### Returns

- Effective refractive index of the mixture (float or complex).

#### Examples

``` py title="Example"
import particula as par
n_mix = par.get_effective_refractive_index(1.33, 1.50, 2.0, 1.0)
print(n_mix)
# Output: ~1.382
```

#### References

- Y. Liu & P. H. Daum, "Relationship of refractive index to mass
  density and self-consistency mixing rules for multicomponent
  mixtures like ambient aerosols," Journal of Aerosol Science,
  vol. 39(11), pp. 974–986, 2008.
  - `DOI` - 10.1016/j.jaerosci.2008.06.006
- Wikipedia contributors, "Refractive index," Wikipedia.

#### Signature

```python
def get_effective_refractive_index(
    m_zero: Union[float, complex],
    m_one: Union[float, complex],
    volume_zero: float,
    volume_one: float,
) -> Union[float, complex]: ...
```


---
# validate_inputs.md

# Validate Inputs

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Validate Inputs

> Auto-generated documentation for [particula.util.validate_inputs](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py) module.

## validate_finite

[Show source in validate_inputs.py:98](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L98)

Validate that a numeric array or scalar has no infinities or NaNs.

#### Arguments

- value : Array-like numeric values to check.
- name : The argument name, used in the error message.

#### Raises

- ValueError : If any element is inf or NaN.

#### Signature

```python
def validate_finite(value, name): ...
```



## validate_inputs

[Show source in validate_inputs.py:113](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L113)

A decorator to validate function inputs against specified constraints.

The constraints are defined by a dictionary of argument names and their
validation types (e.g., "positive", "negative", "nonnegative", etc.). If
any argument violates its constraint, a ValueError is raised.

#### Arguments

- dict_args : Dictionary {argument_name: constraint_type}, where the
    constraint_type is one of:
    - "positive" : Must be strictly > 0.
    - "negative" : Must be strictly < 0.
    - "nonpositive" : Must be <= 0.
    - "nonnegative" : Must be >= 0.
    - "nonzero" : Must be != 0.
    - "finite" : Must not contain inf or NaN.

#### Returns

- A decorator that applies the specified input validations.

#### Examples

``` py
from particula.util.validate_inputs import validate_inputs

@validate_inputs({"mass": "positive", "temperature": "nonnegative"})
def some_function(mass, temperature):
    return mass * temperature
```

#### Signature

```python
def validate_inputs(dict_args): ...
```



## validate_negative

[Show source in validate_inputs.py:38](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L38)

Validate that a numeric array or scalar is strictly negative.

#### Arguments

- value : Array-like numeric values to check.
- name : The argument name, used in the error message.

#### Raises

- ValueError : If any element is >= 0.

#### Signature

```python
def validate_negative(value, name): ...
```



## validate_nonnegative

[Show source in validate_inputs.py:68](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L68)

Validate that a numeric array or scalar is nonnegative (>= 0).

#### Arguments

- value : Array-like numeric values to check.
- name : The argument name, used in the error message.

#### Raises

- ValueError : If any element is < 0.

#### Signature

```python
def validate_nonnegative(value, name): ...
```



## validate_nonpositive

[Show source in validate_inputs.py:53](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L53)

Validate that a numeric array or scalar is nonpositive (<= 0).

#### Arguments

- value : Array-like numeric values to check.
- name : The argument name, used in the error message.

#### Raises

- ValueError : If any element is > 0.

#### Signature

```python
def validate_nonpositive(value, name): ...
```



## validate_nonzero

[Show source in validate_inputs.py:83](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L83)

Validate that a numeric array or scalar is nonzero.

#### Arguments

- value : Array-like numeric values to check.
- name : The argument name, used in the error message.

#### Raises

- ValueError : If any element is 0.

#### Signature

```python
def validate_nonzero(value, name): ...
```



## validate_positive

[Show source in validate_inputs.py:23](https://github.com/uncscode/particula/blob/main/particula/util/validate_inputs.py#L23)

Validate that a numeric array or scalar is strictly positive.

#### Arguments

- value : Array-like numeric values to check.
- name : The argument name, used in the error message.

#### Raises

- ValueError : If any element is <= 0.

#### Signature

```python
def validate_positive(value, name): ...
```
