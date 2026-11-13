# First CPU simulation


This native single-box example initializes a lognormal PMF and performs one
10-second condensation step. The retained
100 kg/m³ vapor concentration is an exaggerated illustration, not a calibrated
atmospheric case or an accuracy/convergence claim.

```python
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import CondensationSpeciesMap

particles = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete",
    radius_bins=np.logspace(-9, -4, 250),
    mode=[100e-9], geometric_standard_deviation=[1.2],
    number_concentration=[1e8], density=1000.0,
)
gas = par.gas.GasData(
    name=["organic"], molar_mass=np.array([0.180]),
    concentration=np.array([[100.0]]),  # kg/m³
    partitioning=np.array([True]),
)
environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[1.0]],
)
aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)
initial_particle_mass = particles.species_mass_inventory.copy()

strategy = par.dynamics.CondensationIsothermal(
    molar_mass=gas.molar_mass.copy(),
    diffusion_coefficient=2e-5, accommodation_coefficient=1.0,
    activity_strategy=par.particles.ActivityIdealMass(),
    surface_strategy=par.particles.SurfaceStrategyVolume(
        surface_tension=0.072, density=1000.0
    ),
    vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(100.0),
    species_map=CondensationSpeciesMap(("organic",), ((0, 0),)),
)
process = par.dynamics.MassCondensation(strategy)
result = process.execute(aerosol, time_step=10.0)
print(result.particles.species_mass_density)  # kg/m³
```

The PMF stores bin number densities and uses the canonical 1 m³ reference
volume. See [native construction](../data-model/construction.md)
and [access and replacement](../data-model/ownership.md).


Next: [process guides](../processes/index.md) or
[examples](../../examples/index.md).
