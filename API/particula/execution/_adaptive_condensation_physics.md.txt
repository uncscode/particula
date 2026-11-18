# particula.execution._adaptive_condensation_physics

*module*

Native material-resolved isothermal rates for adaptive device integration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_adaptive_condensation_physics.py#L1-L1)

## particula.execution._adaptive_condensation_physics._AIR

*attribute* · *module attribute*

```python
_AIR = wp.constant(wp.float64(MOLECULAR_WEIGHT_AIR))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_adaptive_condensation_physics.py#L15-L15)

## particula.execution._adaptive_condensation_physics._R

*attribute* · *module attribute*

```python
_R = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_adaptive_condensation_physics.py#L14-L14)

## particula.execution._adaptive_condensation_physics.material_rates

*function*

```python
def material_rates(masses: FloatTensor, counts: FloatMatrix, density: FloatVector, gas: FloatMatrix, vapor: FloatMatrix, molar: FloatVector, tension: FloatVector, surface_density: FloatVector, surface_reference: FloatVector, phases: IntVector, references: FloatVector, activity_density: FloatVector, kappas: FloatVector, bat_table: FloatVector, activity_mode: wp.int32, water: wp.int32, accommodation: FloatVector, diffusion: FloatVector, mean_free_path: FloatVector, temperature: FloatVector, forward: IntVector, rate: FloatTensor)
```

Evaluate all exchanging lanes, including phase-weighted surface tension.

Masses are kg per particle, gas kg/m3, densities kg/m3, diffusivity m2/s,
tensions N/m, molar masses kg/mol, temperature K, and rates kg/s.
The caller validates the physical trial before accepting it.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_adaptive_condensation_physics.py#L18-L112)
