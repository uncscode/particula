# particula.execution.cpu_configuration

*module*

Non-executable identity for supported native CPU configurations.

GPU plans intentionally impose GPU-specific support limits. CPU restart uses
an exact signature of known CPU physics objects instead. The signature is data
only: callers supply the actual strategy when resuming, and no type named in a
file is imported or instantiated. Unknown custom physics is rejected.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_configuration.py#L1-L1)

## particula.execution.cpu_configuration.NATIVE_PROCESSES

*attribute* · *module attribute*

```python
NATIVE_PROCESSES = (MassCondensation, Coagulation, WallLoss, Dilution, Nucleation)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_configuration.py#L38-L44)

## particula.execution.cpu_configuration._CONFIGURATION_TYPES

*attribute* · *module attribute*

```python
_CONFIGURATION_TYPES = (EnvironmentForcing, *NATIVE_PROCESSES, BrownianCoagulationStrategy, DilutionStrategy, wl.SphericalWallLossStrategy, wl.RectangularWallLossStrategy, wl.ChargedWallLossStrategy, NucleationSourceConfig, NucleationSpeciesMap, NucleationCommitConfig, ExhaustionControls, ns.ActivationNucleationStrategy, ns.KineticNucleationStrategy, ns.ClosedInterval, ns.NucleationValidityDomain, ns.InjectionComposition, ns.FormationMetadata, cs.CondensationIsothermal, cs.CondensationIsothermalStaggered, cs.CondensationLatentHeat, CondensationSpeciesMap, ac.ActivityIdealMass, ac.ActivityIdealMolar, ac.ActivityIdealVolume, ac.ActivityKappaParameter, ac.ActivityNonIdealBinary, ac.ActivityBAT, su.SurfaceStrategyMass, su.SurfaceStrategyMolar, su.SurfaceStrategyVolume, vp.ConstantVaporPressureStrategy, vp.AntoineVaporPressureStrategy, vp.ClausiusClapeyronStrategy, vp.WaterBuckStrategy, vp.TableVaporPressureStrategy, vp.ArblasterLiquidVaporPressureStrategy, vp.LiquidClausiusHybridStrategy, lh.ConstantLatentHeat, lh.LinearLatentHeat, lh.PowerLawLatentHeat)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_configuration.py#L46-L87)

## particula.execution.cpu_configuration._freeze

*function*

```python
def _freeze(value)
```

Represent configuration as data, without retaining live objects.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_configuration.py#L90-L130)

## particula.execution.cpu_configuration.condensation_identity

*function*

```python
def condensation_identity(strategy)
```

Retain all declared CPU transport, mapping and thermodynamic inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_configuration.py#L133-L135)

## particula.execution.cpu_configuration.cpu_process_identity

*function*

```python
def cpu_process_identity(process)
```

Retain native configuration without imposing GPU capability limits.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_configuration.py#L138-L142)
