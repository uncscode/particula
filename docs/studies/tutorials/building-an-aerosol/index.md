# Building an aerosol

<a id="build-a-meaningful-initial-state"></a>

<a id="start-with-the-meaning-of-the-state"></a>

Before choosing a process, decide what the population represents. A value can
be a per-particle mass, a number density, or a represented count; similar arrays
can therefore describe different physical systems.

Begin with the population measure. Per-particle mass, number density, and represented particle count answer different questions even when they have the same array shape. Build a particle population, set gas abundances in the same species order as the material model, then assemble the aerosol. The state-access tutorial covers live arrays, detached values, and replacement.

<!-- STUDY_LIST_START -->
- [What survives a data transfer?](containers/data_containers.py)
- [From a population to an evolving aerosol](native_aerosol.py)
- [Change a model state without losing its meaning](native_state_access.py)
- [Gas and environment](gas/index.md)
- [Particle populations](particles/index.md)
<!-- STUDY_LIST_END -->
