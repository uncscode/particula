# Describe a particle population

Choose an example by the information you have and the quantity you need to infer.
The distinction between a particle property and a population measure runs through
all of them.

| Starting question | Example | Connection |
| --- | --- | --- |
| Why do two spectra of the same population look different? | [Size distributions](aerosol_distributions.py) | Density, bin number, and coordinate conversion |
| How accurately does mobility recover a reference diameter? | [Dry PSL mobility sizing](../../../studies/literature/dry-psl-mobility/psl_mobility.py) | Measured slip and transfer of a one-point calibration |
| How do I initialize from radii or component masses? | [Particle construction](particle_data.py) | Units and population-weighted diagnostics |
| What changes when I use bins or represented counts? | [Native storage](distribution_tutorial.py) | Number density versus physical inventory |
| How does composition change equilibrium pressure? | [Activity models](../../thermodynamics/particle-properties/activity_tutorial.py) | Mixing assumptions and effective activity |
| How does curvature affect the particle surface? | [Surface properties](../../thermodynamics/particle-properties/particle_surface_tutorial.py) | Effective properties and Kelvin correction |
| How can I evaluate activity directly on arrays? | [Activity functions](../../thermodynamics/particle-properties/activity_functions.py) | Numerical functions beneath the strategy interface |

Begin with the first two pages for a new simulation. Use the property-model
examples when composition or curvature matters to your research question, then
carry those assumptions into [process calculations](../../processes/index.md).
