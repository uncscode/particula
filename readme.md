# Particula

A simple, fast, and powerful particle simulator for aerosol science.

**Development version:** 0.3.0 · **Requires:** Python 3.12+

[Documentation](https://uncscode.github.io/particula/) | [Examples](https://uncscode.github.io/particula/examples/) | [PyPI](https://pypi.org/project/particula/)

## Installation

```bash
pip install particula
```

or via conda:

```bash
conda install -c conda-forge particula
```

These commands install the published release. To work with the 0.3.0 source
checkout, including the native APIs below, install from the repository root:

```bash
python -m pip install -e '.[dev,extra]'
```

The source version does not indicate that 0.3.0 has been published. See the
[release validation record](https://uncscode.github.io/particula/contribute/validation/releases/v0.3.0/)
for revision-specific evidence and outstanding release gates.

## Quick Start

This example uses the 0.3.0 native data model: `ParticleData`, `GasData`, and
`EnvironmentData` held by an `Aerosol`. Install the source checkout as above
when using a published release that predates these APIs.

```python
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import CondensationSpeciesMap

particles = par.particles.create_resolved_particle_data(
    masses=[1e-18, 2e-18], density=1000.0,
    concentration=[1.0, 1.0], volume=1e-6,  # counts and m³
)
gas = par.gas.GasData(
    name=["organic"], molar_mass=np.array([0.180]),
    concentration=np.array([[1e-6]]),  # kg/m³
    partitioning=np.array([True]),
)
environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[1.0]],
)
aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)

strategy = par.dynamics.CondensationIsothermal(
    molar_mass=gas.molar_mass.copy(),
    diffusion_coefficient=2e-5, accommodation_coefficient=1.0,
    activity_strategy=par.particles.ActivityIdealMass(),
    surface_strategy=par.particles.SurfaceStrategyVolume(density=1000.0),
    vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e-3),
    species_map=CondensationSpeciesMap(("organic",), ((0, 0),)),
)
condensation = par.dynamics.MassCondensation(strategy)
coagulation = par.dynamics.Coagulation(
    par.dynamics.BrownianCoagulationStrategy("particle_resolved")
)
aerosol = (condensation | coagulation).execute(
    aerosol, time_step=10.0, sub_steps=10
)
print(aerosol.particles.number_density)  # m⁻³, not raw counts
```

This single-box illustration retains explicit population and species meaning.
See [native access and replacement](https://uncscode.github.io/particula/guide/data-model/ownership/)
for ownership, validation and non-unit-volume inventory checks.

## Documentation

| Looking for… | Read next |
| --- | --- |
| Installation and local documentation | [Installation guide](https://uncscode.github.io/particula/guide/getting-started/installation/) |
| Tutorials and examples | [Examples](https://uncscode.github.io/particula/examples/) |
| Scientific processes | [Process guide](https://uncscode.github.io/particula/guide/processes/) |
| Scientific background | [Theory](https://uncscode.github.io/particula/theory/) |
| API reference | [API](https://uncscode.github.io/particula/api/) |
| Upgrading to 0.3.0 | [Migration guide](https://uncscode.github.io/particula/guide/migration/v0.3.0/) |
| AI assistants | [Particula Helper](https://uncscode.github.io/particula/guide/ai-assistants/) |
| Contributing and code structure | [Contributor guide](https://uncscode.github.io/particula/contribute/CONTRIBUTING/), [architecture](https://uncscode.github.io/particula/contribute/architecture/) |

## Citation

If you use Particula in your research, please cite:
> Particula [Computer software]. DOI: [10.5281/zenodo.6634653](https://doi.org/10.5281/zenodo.6634653)

## License

MIT
