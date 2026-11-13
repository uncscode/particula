# Containers

## Historical terminology — unsupported before-state

This table is migration history, not executable instructions. Legacy facade
constructors, `.data` wrapping, `from_data`, `from_representation`,
`to_representation`, `from_species` and `to_species` are not supported native
construction or migration steps. Do not strip a native tag to cross a bridge.
The particle facade, its builders/factory and both particle conversion bridges
have been removed from the v0.3.0 development tree. Import native particle
initializers or `ParticleDataBuilder` directly from `particula.particles`.
The `from_species()` and `to_species()` gas bridges have also been removed.
Construct `GasData` or use `GasDataBuilder`, preserving full ordered species
and mixed partitioning masks; use `copy()` for detached native storage.
`Atmosphere` and `AtmosphereBuilder` are removed as well: hold `GasData` and
`EnvironmentData` in the native `Aerosol`, and replace them together when
changing the full gas-species layout.

| Historical API | Native replacement |
|---|---|
| `ParticleRepresentation.get_species_mass()` | `particles.masses` |
| `get_radius()` / `get_mass()` | `particles.radii` / `particles.total_mass` |
| `get_concentration()` | Choose raw counts or `slot_concentration_density` explicitly |
| `get_total_concentration()` | `particles.number_density` |
| `GasSpecies.get_concentration()` | `gas.concentration`, retaining the box axis |
| `Atmosphere` nested inside an aerosol | Separate `GasData` and `EnvironmentData` |
| Facade-owned activity, surface and vapor-pressure strategies | Process-owned scientific configuration |

Old numerical normalization errors are not parity targets. Determine the
original physical measure before initializing native storage.

