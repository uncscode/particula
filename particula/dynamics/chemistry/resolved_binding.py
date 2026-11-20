"""Immutable physical layout for bulk material reactions and gas uptake."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from particula.dynamics.chemistry.binding import _compile_network
from particula.dynamics.chemistry.mechanism import ReactionMechanism

GAS, BULK, UPTAKE = range(3)


def _resolve_columns(
    species,
    materials,
    names,
    gas,
    particle_masses,
    declared,
    particle_columns,
    overrides,
):
    n_gas = len(names)
    lanes = []
    for key in species:
        if key in materials:
            lane = particle_columns[key]
            if (
                isinstance(lane, bool)
                or not isinstance(lane, Integral)
                or not 0 <= lane < len(particle_masses)
            ):
                raise ValueError(f"{key}: invalid particle column")
            particle_masses[lane] = declared[key]
            lanes.append(n_gas + int(lane))
        else:
            name = overrides.get(key, key[4:])
            if name not in names:
                raise ValueError(f"{key}: unmapped gas {name!r}")
            lane = names.index(name)
            if key in declared and not np.isclose(
                gas.molar_mass[lane], declared[key], rtol=1e-12, atol=0
            ):
                raise ValueError(
                    f"{key}: gas molar mass differs from mechanism"
                )
            lanes.append(lane)
    return lanes


def _reaction_kinds(mechanism, species, lanes, n_gas):
    indices = dict(zip(species, lanes, strict=True))
    kinds, sources = [], []
    gamma = np.zeros(n_gas)
    for reaction in mechanism.reactions:
        source = indices[reaction.reactants[0][0]]
        if reaction.law == "reactive_uptake":
            kind = UPTAKE
            gamma[source] += reaction.coefficient
        else:
            kind = GAS if source < n_gas else BULK
        kinds.append(kind)
        sources.append(source)
    if np.any(gamma > 1):
        raise ValueError("summed reactive uptake probabilities exceed one")

    return kinds, sources


@dataclass(frozen=True)
class ResolvedBinding:
    """Reaction columns follow all gas lanes, then all particle material lanes.

    Densities are kg/m³ of material and molar masses are kg/mol. Unmapped
    material contributes to geometry without participating in any reaction.
    """

    mechanism: ReactionMechanism
    gas_names: tuple
    gas_masses: tuple
    density: tuple
    particle_masses: tuple
    indices: tuple
    net: tuple
    orders: tuple
    parameters: tuple
    kinds: tuple
    sources: tuple

    @classmethod
    def create(cls, mechanism, aerosol, particle_columns, gas_overrides):
        """Resolve names once and check units, stoichiometry and collisions."""
        p, gas = aerosol.particles, aerosol.gas
        if p.distribution_type != "particle_resolved":
            raise ValueError("chemistry requires particle_resolved state")
        names = tuple(gas.name)
        if len(set(names)) != len(names):
            raise ValueError("gas names must be unique")
        species = dict(mechanism.species)
        declared = dict(mechanism.declared_masses)
        materials = {n for n in species if n.startswith("particle:")}
        if set(particle_columns) != materials:
            raise ValueError(
                "particle_columns must explicitly map each particle species"
            )
        overrides = {} if gas_overrides is None else dict(gas_overrides)
        if overrides.keys() - (species.keys() - materials):
            raise ValueError("gas overrides reference unknown species")
        particle_masses = np.ones(p.n_species)
        n_gas = len(names)
        lanes = _resolve_columns(
            species,
            materials,
            names,
            gas,
            particle_masses,
            declared,
            particle_columns,
            overrides,
        )
        if len(set(lanes)) != len(lanes):
            raise ValueError("species bindings must use distinct columns")
        masses = np.concatenate((gas.molar_mass, particle_masses))
        if not np.all(np.isfinite(masses) & (masses > 0)):
            raise ValueError("molar masses must be finite and positive")
        net, orders, rates = _compile_network(mechanism, species, masses[lanes])
        kinds, sources = _reaction_kinds(mechanism, species, lanes, n_gas)

        def layout(rows):
            table = np.zeros((len(rows), masses.size), dtype=int)
            table[:, lanes] = rows
            return tuple(tuple(map(int, row)) for row in table)

        return cls(
            mechanism,
            names,
            tuple(map(float, gas.molar_mass)),
            tuple(map(float, p.density)),
            tuple(map(float, particle_masses)),
            tuple(lanes),
            layout(net),
            layout(orders),
            rates,
            tuple(kinds),
            tuple(sources),
        )

    def validate(self, aerosol):
        """Require the ordered metadata used at binding."""
        p, gas = aerosol.particles, aerosol.gas
        if (
            p.distribution_type != "particle_resolved"
            or tuple(gas.name) != self.gas_names
            or tuple(gas.molar_mass) != self.gas_masses
            or tuple(p.density) != self.density
        ):
            raise ValueError(
                "resolved layout or metadata changed; rebind chemistry"
            )
        p.validate_representation()
        active = p.concentration > 0
        if np.any(active != np.any(p.masses > 0, axis=2)) or np.any(
            p.charge[~active] != 0
        ):
            raise ValueError("invalid active or free material slot")
        if not np.all(np.isfinite(p.density) & (p.density > 0)) or not np.all(
            np.isfinite(p.volume) & (p.volume > 0)
        ):
            raise ValueError(
                "chemistry requires positive finite density and box volume"
            )
        if not np.all(
            np.isfinite(aerosol.environment.temperature)
            & (aerosol.environment.temperature > 0)
        ):
            raise ValueError("chemistry requires positive finite temperature")
        if np.any(~np.isfinite(gas.concentration)) or np.any(
            gas.concentration < 0
        ):
            raise ValueError("chemistry requires finite nonnegative gas")
