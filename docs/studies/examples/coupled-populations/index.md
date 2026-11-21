# Coupled populations

<a id="connect-processes-to-a-research-question"></a>

These worked simulations combine population construction, thermodynamic inputs,
time evolution, and diagnostics. Choose a question, then inspect the example's
assumptions and validation boundary before adapting its parameters. A successful
run or an attractive size distribution alone does not establish a reliable
material model.

<div class="study-figures">

![Schematic of a wet, dry, then wet chamber. Water shells shrink and regrow around the same-sized seeds. Fewer seeds remain airborne as more reach the chamber wall.](../../../images/coupled-populations/chamber-multiple-cycles.svg)\
[Separate water cycling from seed loss in a chamber](chambers/cloud_chamber_multi_cycle.py)

![A surviving seed keeps its core while its water shell expands in humid air and contracts in dry air. The three drawings follow one illustrative trajectory.](../../../images/coupled-populations/chamber-single-cycle.svg)\
[Follow droplets through one wet–dry chamber cycle](chambers/cloud_chamber_single_cycle.py)

![A smooth size spectrum sits above species inventories that drift in opposite directions from their common starting reference during fixed-composition binned coagulation. The marks show directions schematically.](../../../images/coupled-populations/organic-binned-inventories.svg)\
[When a size-distribution plot hides a composition error](emissions/organic_partitioning_and_coagulation.py)

![Cooling moves organic material from vapor onto pre-existing seeds. The schematic shows fewer vapor marks and a thicker organic coating around the same dark core.](../../../images/coupled-populations/cooling-organic-uptake.svg)\
[Follow organic uptake onto seeds during imposed cooling](emissions/soot_formation_in_flames.py)

![One water-grown organic and soot population supplies four copies: condensation alone, or continued condensation with Brownian collisions, Brownian plus differential settling, or turbulent shear. Branches depict the comparison design.](../../../images/coupled-populations/biomass-collision-branches.svg)\
[How do collision models change a water-grown mixed aerosol?](emissions/biomass_burning_cloud_interactions.py)

![The balance is level in the mass audit. The identity check fails: the retained lookup assigned hexachlorophene to the requested lysozyme. The illustration conveys a limitation in material assignment.](../../../images/coupled-populations/respiratory-material-identity.svg)\
[Audit a respiratory-droplet illustration before using its predictions](emissions/cough_droplets_partitioning.py)

![At the same initial dry particle number density and temperature, the higher-pressure chamber retains more particles. The gauge needles and particle densities illustrate the direction of the simulated pressure contrast.](../../../images/coupled-populations/dry-chamber-pressure.svg)\
[Dry charged aerosols: ground level, mesopause and pressurized chambers](environment/gpu_dry_environment_sweep.py)

</div>

<!-- STUDY_LIST_START -->
- [Chamber cycles](chambers/index.md)
- [Charged aerosols and ensembles](charged-aerosols/index.md)
- [Dry aerosols and humid air with ions](environment/index.md)
- [Emissions and partitioning](emissions/index.md)
- [Forcing and heterogeneous boxes](forcing/index.md)
<!-- STUDY_LIST_END -->
