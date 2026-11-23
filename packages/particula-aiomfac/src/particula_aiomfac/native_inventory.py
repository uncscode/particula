"""Publish closed species equilibrium into explicit native material lanes."""

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import Avogadro

from .reservoirs import SpeciesInventory, _ion_mw


@dataclass(frozen=True)
class SpeciesLayout:
    """Native material indices by physical species, and one bulk population.

    Each box contains one internally equilibrating population. Reserve two
    equivalent slots to represent a fractional ensemble mean charge with
    adjacent integer charge states. The gas is shared by this population.
    Gas names come from the reaction mapping.
    """

    liquid_neutrals: dict
    liquid_ions: dict
    solids: dict
    surface_ions: dict = field(default_factory=dict)
    residual_ions: dict = field(default_factory=dict)
    slots: tuple = (0,)


def _validate_lanes(p, fields, slots):
    lanes = [i for mapping in fields.values() for i in mapping.values()]
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        for i in (*lanes, *slots)
    ):
        raise ValueError("material lanes and population slots must be integers")
    if sorted(lanes) != list(range(p.masses.shape[-1])):
        raise ValueError("map every physical material lane exactly once")
    if (
        not 1 <= len(slots) <= 2
        or len(set(slots)) != len(slots)
        or any(i < 0 or i >= p.masses.shape[1] for i in slots)
    ):
        raise ValueError(
            "reserve one or two distinct slots per bulk population"
        )
    if np.any(np.delete(p.concentration, slots, axis=1)):
        raise ValueError(
            "every occupied slot must belong to the bulk population"
        )


def _layout(aerosol, layout):
    from particula.aerosol import Aerosol

    if not isinstance(aerosol, Aerosol) or not isinstance(
        layout, SpeciesLayout
    ):
        raise TypeError(
            "native equilibrium requires an Aerosol and SpeciesLayout"
        )
    p, g = aerosol.particles, aerosol.gas
    if p.distribution_type not in ("discrete", "particle_resolved"):
        raise ValueError("bulk remapping requires discrete or resolved slots")
    if aerosol.fields.particle_names():
        raise ValueError(
            "map particle extension fields before bulk charge redistribution"
        )
    fields = {k: v for k, v in vars(layout).items() if k != "slots"}
    slots = list(layout.slots)
    _validate_lanes(p, fields, slots)
    if any(
        not x.flags.writeable
        for x in (p.masses, p.concentration, p.charge, g.concentration)
    ):
        raise ValueError(
            "native mass, number, charge and gas arrays must be writable"
        )
    return fields, slots


def _gas_lanes(g, provider, reactions):
    gas_names = list(reactions)
    gas_lanes = [list(g.name).index(name) for name in gas_names]
    if any(not g.partitioning[i] for i in gas_lanes):
        raise ValueError("mapped gas species must participate in partitioning")
    molecular_masses = {
        **dict(
            zip(
                provider.prepared.ions,
                _ion_mw(provider.prepared.ions),
                strict=True,
            )
        ),
        **dict(
            zip(
                provider.prepared.neutrals,
                provider.prepared.neutral_mw,
                strict=True,
            )
        ),
    }
    for name, lane in zip(gas_names, gas_lanes, strict=True):
        reference = sum(
            nu * molecular_masses[k]
            for k, nu in reactions[name].aqueous.items()
        )
        if not np.isclose(reference, g.molar_mass[lane], rtol=2e-4, atol=0):
            raise ValueError(
                f"native gas molar mass disagrees with reaction {name}"
            )
    return gas_names, gas_lanes


def _publish_particle_classes(
    number, mass, charge, slots, destination, charged_lanes, z, mw
):
    # Separate excess charge from balanced surface/dry ion mass. Rescale
    # just this excess in the two classes to give exact integer charges.
    excess = np.zeros_like(destination)
    positive = sum(
        destination[i] * z[k] / mw[k] * Avogadro
        for k, i in charged_lanes
        if z[k] > 0
    )
    negative = sum(
        -destination[i] * z[k] / mw[k] * Avogadro
        for k, i in charged_lanes
        if z[k] < 0
    )
    avg = positive - negative
    for key, lane in charged_lanes:
        if z[key] * avg > 0:
            excess[lane] = (
                destination[lane] * abs(avg) / max(positive, negative)
            )
    rounded = round(avg)
    if abs(avg - rounded) < 1e-12:
        weights, levels = [1.0], [rounded]
    else:
        lower = np.floor(avg)
        weights, levels = [lower + 1 - avg, avg - lower], [lower, lower + 1]
    if len(levels) > len(slots):
        raise ValueError(
            "reserve a second slot for adjacent integer charge states"
        )
    represented = number[slots].sum()
    mass[slots], number[slots], charge[slots] = 0.0, 0.0, 0.0
    if destination.sum() == 0:
        weights, levels = [], []
    for i, level, weight in zip(
        slots[: len(levels)], levels, weights, strict=True
    ):
        mass[i] = (
            destination
            if avg == 0
            else destination - excess + excess * (level / avg)
        )
        number[i] = represented * weight
        charge[i] = level


def apply_species_equilibrium(aerosol, provider, layout, **options):
    """Replace exclusive physical inventories after every box has converged."""
    fields, slots = _layout(aerosol, layout)
    p, g = aerosol.particles, aerosol.gas
    gas_names, gas_lanes = _gas_lanes(
        g, provider, options.get("gas_reactions", {})
    )
    density = p.slot_concentration_density
    mass, number, charge, vapor = [
        x.copy() for x in (p.masses, p.concentration, p.charge, g.concentration)
    ]
    z = dict(zip(provider.prepared.ions, provider.prepared.charge, strict=True))
    mw = dict(
        zip(
            provider.prepared.ions, _ion_mw(provider.prepared.ions), strict=True
        )
    )
    reports = []
    for box in range(len(mass)):
        count = density[box, slots].sum()
        if count <= 0:
            raise ValueError(
                "bulk equilibrium needs a nonempty seed population"
            )
        total = density[box, slots] @ mass[box, slots] * 1e9
        inventory = SpeciesInventory(
            **{
                k: {name: total[idx] for name, idx in mapping.items()}
                for k, mapping in fields.items()
            },
            gases={
                name: vapor[box, i] * 1e9
                for name, i in zip(gas_names, gas_lanes, strict=True)
            },
        )
        charged_lanes = [
            (key, lane)
            for k in ("surface_ions", "residual_ions")
            for key, lane in fields[k].items()
        ]
        initial_charge = sum(
            total[lane] * 1e-9 * z[key] / mw[key] for key, lane in charged_lanes
        )
        recorded = density[box, slots] @ charge[box, slots] / Avogadro
        if abs(recorded - initial_charge) > 1e-8 * max(
            abs(recorded), abs(initial_charge), count / Avogadro
        ):
            raise ValueError(
                "native charge disagrees with surface and residual ions"
            )
        out = provider.equilibrate_species(
            species_inventory=inventory,
            temperature=float(aerosol.environment.temperature[box]),
            **options,
        )
        destination = np.zeros_like(total)
        for k, mapping in fields.items():
            for key, value in getattr(out.inventory, k).items():
                if value and key not in mapping:
                    raise ValueError(f"missing native {k} material for {key}")
                if key in mapping:
                    destination[mapping[key]] = value * 1e-9 / count
        _publish_particle_classes(
            number[box],
            mass[box],
            charge[box],
            slots,
            destination,
            charged_lanes,
            z,
            mw,
        )
        vapor[box, gas_lanes] = [
            out.inventory.gases[name] * 1e-9 for name in gas_names
        ]
        reports.append(out)
    np.copyto(p.masses, mass)
    np.copyto(p.concentration, number)
    np.copyto(p.charge, charge)
    np.copyto(g.concentration, vapor)
    return tuple(reports)
