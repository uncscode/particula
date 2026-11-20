"""Resolve native gas identities and compile molar kinetic coefficients."""

from dataclasses import dataclass

import numpy as np

from particula.dynamics.chemistry.mechanism import ReactionMechanism
from particula.util.constants import GAS_CONSTANT


def _resolve_species(mechanism, gas, overrides):
    names = tuple(gas.name)
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("gas layout requires unique nonblank names")
    overrides = {} if overrides is None else dict(overrides)
    species = dict(mechanism.species)
    if overrides.keys() - species.keys():
        raise ValueError("overrides reference unknown mechanism species")
    indices = []
    for name in species:
        target = overrides.get(name, name[4:])
        if target not in names:
            raise ValueError(f"unmapped mechanism species {name}: {target!r}")
        indices.append(names.index(target))
    if len(set(indices)) != len(indices):
        raise ValueError("two mechanism species map to the same gas column")
    masses = tuple(float(gas.molar_mass[i]) for i in indices)
    if not all(np.isfinite(m) and m > 0 for m in masses):
        raise ValueError("bound species need finite positive gas molar masses")
    return names, species, indices, masses


@dataclass(frozen=True)
class BoundMechanism:
    """Immutable layout and mechanism shared by CPU and device preparation.

    Matrix rows are reactions and columns follow the mechanism species order.
    Concentrations supplied to rate and Jacobian evaluation are mol/m³ of air.
    """

    mechanism: ReactionMechanism
    gas_names: tuple[str, ...]
    indices: tuple[int, ...]
    molar_masses: tuple[float, ...]
    stoichiometry: tuple
    orders: tuple
    parameters: tuple

    @classmethod
    def from_gas(cls, mechanism, gas, overrides):
        """Compile layout indices, balance checks and rate parameters."""
        names, species, indices, masses = _resolve_species(
            mechanism, gas, overrides
        )
        stoichiometry, orders, parameters = [], [], []
        for reaction in mechanism.reactions:
            lhs, rhs = dict(reaction.reactants), dict(reaction.products)
            coeffs = tuple(rhs.get(n, 0) - lhs.get(n, 0) for n in species)
            # A large catalyst present on both sides must not mask an
            # imbalance in the species actually consumed and produced.
            mass_in = sum(
                m * -v for m, v in zip(masses, coeffs, strict=True) if v < 0
            )
            mass_out = sum(
                m * v for m, v in zip(masses, coeffs, strict=True) if v > 0
            )
            if not np.isclose(mass_in, mass_out, rtol=1e-12, atol=0):
                raise ValueError(
                    f"reaction {reaction.name}: molar masses do not balance"
                )
            participants = lhs.keys() | rhs.keys()
            if all(species[n] for n in participants):
                atoms: dict[str, int] = {}
                for n, coefficient in zip(species, coeffs, strict=True):
                    for element, count in species[n]:
                        atoms[element] = (
                            atoms.get(element, 0) + coefficient * count
                        )
                if any(atoms.values()):
                    raise ValueError(
                        f"reaction {reaction.name}: elemental imbalance"
                    )
            stoichiometry.append(coeffs)
            orders.append(
                tuple(dict(reaction.orders).get(n, 0) for n in species)
            )
            parameters.append(
                (
                    reaction.coefficient,
                    reaction.exponent,
                    reaction.energy,
                    reaction.reference_temperature,
                )
            )
        return cls(
            mechanism,
            names,
            tuple(indices),
            masses,
            tuple(stoichiometry),
            tuple(orders),
            tuple(parameters),
        )

    def validate(self, gas):
        """Reject a stale ordered layout or mass before evaluating chemistry."""
        if (
            tuple(gas.name) != self.gas_names
            or tuple(gas.molar_mass[i] for i in self.indices)
            != self.molar_masses
        ):
            raise ValueError(
                "gas layout or molar masses changed; rebind the mechanism"
            )

    def kinetics(self, temperature):
        """Build per-box molar rate and Jacobian evaluators at fixed kelvin."""
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("chemistry requires a finite positive temperature")
        coefficient, exponent, energy, reference = np.array(self.parameters).T
        rates = (
            coefficient
            * (temperature / reference) ** exponent
            * np.exp(-energy / (GAS_CONSTANT * temperature))
        )
        if not np.all(np.isfinite(rates)):
            raise ValueError("nonfinite Arrhenius coefficient")
        return MolarKinetics(
            np.array(self.stoichiometry).T, np.array(self.orders), rates
        )


@dataclass(frozen=True)
class MolarKinetics:
    """Rate of concentration change in mol/m³/s and its state derivative."""

    net: np.ndarray
    orders: np.ndarray
    coefficients: np.ndarray

    def rate(self, time, state):
        """Evaluate all reaction extents before summing species tendencies."""
        return self.net @ (
            self.coefficients * np.prod(state**self.orders, axis=1)
        )

    def jacobian(self, time, state):
        """Evaluate the rate derivative even at zero concentration."""
        jac = np.zeros((state.size, state.size))
        for j in range(state.size):
            powers = self.orders.copy()
            powers[:, j] = np.maximum(0, powers[:, j] - 1)
            jac[:, j] = self.net @ (
                self.coefficients
                * self.orders[:, j]
                * np.prod(state**powers, axis=1)
            )
        return jac
