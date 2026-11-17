# particula.execution._gpu_gas_forcing

*module*

Selective gas forcing without returning evolving gas state to the host.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_gas_forcing.py#L1-L1)

## particula.execution._gpu_gas_forcing._R

*attribute* · *module attribute*

```python
_R = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_gas_forcing.py#L17-L17)

## particula.execution._gpu_gas_forcing._commit_targets

*function*

```python
def _commit_targets(values: FloatMatrix, columns: IntVector, gas: FloatMatrix)
```

Overwrite only the explicitly prescribed species.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_gas_forcing.py#L48-L52)

## particula.execution._gpu_gas_forcing._evaluate_targets

*function*

```python
def _evaluate_targets(values: FloatMatrix, columns: IntVector, modes: IntVector, pressures: FloatVector, molar_mass: FloatVector, temperature: FloatVector, invalid: IntVector)
```

Convert prescribed ratios to kg/m3 using device temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_gas_forcing.py#L20-L45)

## particula.execution._gpu_gas_forcing.apply_species_forcing

*function*

```python
def apply_species_forcing(owner, vapor, concentrations, saturation_ratios)
```

Validate all named targets before changing any physical state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_gas_forcing.py#L55-L132)
