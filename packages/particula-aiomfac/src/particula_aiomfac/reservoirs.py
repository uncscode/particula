"""Mass-owning solid, dissolved, surface, dry and gas species inventories.

Gas reactions use an ideal gas at the stated pressure standard and the
AIOMFAC aqueous reference conventions. Charged transfer additionally requires
an explicitly parameterized interfacial ion population and capacitance.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import Avogadro, R, elementary_charge
from scipy.optimize import least_squares, linprog
from scipy.special import logsumexp

from ._equilibrium import ConvergenceError
from .model import _tables

FARADAY = Avogadro * elementary_charge


@dataclass(frozen=True)
class SpeciesInventory:
    """Disjoint mass reservoirs, all in µg/m³ of air.

    Solids are named formula units. Ions (in bulk solution, on the surface,
    or in a dry residue) use AIOMFAC ion IDs. Each atom belongs to just one
    reservoir. All calculations use private copies.
    """

    liquid_neutrals: dict = field(default_factory=dict)
    liquid_ions: dict = field(default_factory=dict)
    solids: dict = field(default_factory=dict)
    gases: dict = field(default_factory=dict)
    surface_ions: dict = field(default_factory=dict)
    residual_ions: dict = field(default_factory=dict)

    @property
    def total_mass(self):
        """Total physical mass in µg/m³, including the gas."""
        return sum(sum(values.values()) for values in vars(self).values())


@dataclass(frozen=True)
class GasReaction:
    """Dissolution reaction for one gas at a specified temperature.

    Positive stoichiometry designates aqueous products and negative values
    aqueous reactants. Use neutral names and integer AIOMFAC ion IDs.
    K = product(a_i**nu_i)/(p/p_standard) at zero electric potential.
    """

    aqueous: dict
    log_k: float  # natural logarithm
    temperature_k: float
    charge: int = 0
    standard_pressure_pa: float = 101325.0


# Dimensionless activities, unit standard molality, atmosphere pressure scale.
# Fountoukis and Nenes (2007), Table 2, doi:10.5194/acp-7-4639-2007.
HCL_298 = GasReaction({205: 1, 242: 1}, np.log(1.971e6), 298.15)
NH3_298 = GasReaction(
    {204: 1, 205: -1}, np.log(57.64 * 1.805e-5 / 1.010e-14), 298.15
)


@dataclass(frozen=True)
class IonSurface:
    """Finite ideal adsorption layer sharing a capacitor to the gas reference.

    Site capacity is mol sites/m³ air, capacitance is F/m³ air, and each
    adsorbed ion occupies one site. log_binding maps ion IDs to ln(K_ads)
    in the same electrochemical convention as the gas reactions.
    """

    sites_mol_m3: float
    capacitance_f_m3: float
    log_binding: dict


@dataclass(frozen=True)
class SpeciesEquilibrium:
    """Physical mass and charge balances plus the condensed equilibrium."""

    inventory: SpeciesInventory
    condensed: object
    mass_balance_error: float
    charge_balance_mol_m3: float
    phase_potential_v: float
    surface_potential_v: float
    residual: float


def _ion_mw(ids):
    tables = _tables()
    return (
        np.array(
            [
                tables["SMWC"][i - 201] if i <= 240 else tables["SMWA"][i - 241]
                for i in ids
            ]
        )
        * 1e-3
    )


def _mass_vector(mapping, keys):
    if not set(mapping) <= set(keys):
        raise ValueError(
            f"undeclared inventory species: {set(mapping) - set(keys)}"
        )
    values = np.array([mapping.get(key, 0.0) for key in keys], dtype=float)
    if np.any(~np.isfinite(values) | (values < 0)):
        raise ValueError("species masses must be finite and nonnegative")
    return values


def _reaction_matrix(reactions, keys, charge, temperature):
    matrix = np.zeros((len(keys), len(reactions)))
    for j, (name, rxn) in enumerate(reactions.items()):
        if rxn.temperature_k != temperature:
            raise ValueError(
                f"no gas equilibrium constant for {name} at {temperature} K"
            )
        if (
            not np.isfinite(rxn.log_k)
            or not np.isfinite(rxn.standard_pressure_pa)
            or rxn.standard_pressure_pa <= 0
            or not rxn.aqueous
        ):
            raise ValueError(
                "invalid gas equilibrium constant or reference pressure"
            )
        for key, nu in rxn.aqueous.items():
            if key not in keys or not np.isfinite(nu) or nu == 0:
                raise ValueError(
                    f"invalid aqueous stoichiometry for {name}: {key}"
                )
            matrix[keys.index(key), j] = nu
        if charge @ matrix[:, j] != rxn.charge:
            raise ValueError(f"unbalanced charge for gas reaction {name}")
    return matrix


def _direct_vapor_names(reactions, neutral_names):
    neutral_gases = {}
    for name, rxn in reactions.items():
        if len(rxn.aqueous) != 1:
            break
        key, nu = next(iter(rxn.aqueous.items()))
        if key not in neutral_names or nu != 1 or key in neutral_gases:
            break
        neutral_gases[key] = name
    return neutral_gases


def _analytical_feed(p, names, conserved, mw):
    """Convert conserved neutral and ion moles to analytical masses."""
    feed = np.zeros(p.ncomponents)
    for i, key in enumerate(p.neutrals):
        feed[names.index(key)] = conserved[i] * mw[i] * 1e9
    levels = conserved[len(p.neutrals) :]
    if np.any(levels):
        real = p.salt_indices < p.ncomponents
        scale = levels.max()
        pair = (
            linprog(
                np.zeros(sum(real)),
                A_eq=p.stoich[real].T,
                b_eq=levels / scale,
                bounds=(0, None),
                method="highs",
            )
            if np.any(real)
            else None
        )
        if (
            pair is None
            or not pair.success
            or not np.allclose(
                pair.x @ p.stoich[real],
                levels / scale,
                rtol=1e-10,
                atol=1e-14,
            )
        ):
            raise ValueError(
                "closed neutral vapors require a balanced analytical salt basis"
            )
        feed[p.salt_indices[real]] = pair.x * scale * p.salt_mw[real] * 1e9
    return feed


def _neutral_partition(
    provider,
    conserved,
    matrix,
    mw,
    gas_mw,
    reactions,
    neutral_gases,
    temperature,
    options,
    liquid,
):
    """Share the flat-surface neutral solver, including the vapor-only limit."""
    p = provider.prepared
    neutral_names = list(p.neutrals)
    n = len(neutral_names)
    gas_names = list(reactions)
    names = tuple(provider.mixture.components)
    water_exchanges = names[provider.water_index] in neutral_gases
    if not water_exchanges and (
        np.any(conserved[n:])
        or any(
            conserved[i] and key not in neutral_gases
            for i, key in enumerate(neutral_names)
        )
    ):
        # This inventory contains permanent condensed material. Let the
        # physical-species solver keep water and ions in their original lanes.
        return None
    feed = _analytical_feed(p, names, conserved, mw)
    saturations = {
        key: reactions[name].standard_pressure_pa
        * np.exp(-reactions[name].log_k)
        * mw[neutral_names.index(key)]
        * 1e9
        / (R * temperature)
        for key, name in neutral_gases.items()
    }
    if water_exchanges:
        bulk = provider.solve_partitioning(
            inventories=feed,
            temperature=temperature,
            saturation_concentrations=saturations,
            **options,
        )
        vapor = bulk.gas_concentrations
        result = bulk.solid_liquid
        vapor_only = bulk.liquids is None
        error = bulk.vapor_residual
    else:
        from .partitioning import _vapor_only_state

        # Every occupied component can enter the gas. Test the tangent plane
        # at that boundary before asking the wet solver for a positive liquid.
        cstar = np.array([saturations.get(key, 0.0) for key in names])
        backend = provider.equilibrium_backend(
            options["execution"], options["device"]
        )

        def state(masses):
            physical = liquid(
                masses[p.neutral_indices] * 1e-9 / mw[:n], conserved[n:]
            )
            return physical.liquids, physical

        boundary = _vapor_only_state(
            provider, backend, state, feed, cstar, feed, temperature, None
        )
        if boundary is None:
            return None
        vapor, _, phases, result = boundary
        vapor_only = phases is None
        error = 0.0

    gas = np.zeros(len(gas_names))
    for key, name in neutral_gases.items():
        j = gas_names.index(name)
        gas[j] = vapor[names.index(key)] * 1e-9 / gas_mw[j]
    if result is None:
        condensed = conserved - matrix @ gas
        # Avoid roundoff from independently converted trace inventories.
        if vapor_only:
            condensed[:] = 0.0
        result = liquid(np.maximum(condensed[:n], 0.0), condensed[n:])
    return result, gas, error


def _log_activities(chemistry):
    """Return the aqueous chemical potentials in the reaction matrix order."""
    with np.errstate(divide="ignore"):
        return np.r_[
            np.log(chemistry.neutral_activities),
            np.log(chemistry.ion_molalities) + chemistry.ion_log_coefficients,
        ]


def _gas_amounts(reactions, matrix, active, potentials, phi, temperature):
    """Evaluate ideal vapor concentrations in mol/m³ at liquid potential phi."""
    gas = np.zeros(len(reactions))
    options = tuple(reactions.values())
    for j in np.flatnonzero(active):
        rxn = options[j]
        # Absent bystanders can have undefined potentials and zero transfer.
        used = matrix[:, j] != 0
        affinity = potentials[used] @ matrix[used, j] + rxn.charge * phi
        pressure = rxn.standard_pressure_pa * np.exp(
            np.clip(affinity - rxn.log_k, -745, 100)
        )
        gas[j] = pressure / (R * temperature)
    return gas


def _reactive_vapors(
    model, liquid, conserved, reactions, matrix, temperature, surface
):
    """Solve gas and surface exchange on the conserved aqueous basis."""
    from .solids import conserved_species_matrix

    transform = conserved_species_matrix(model)
    total = transform @ conserved
    columns = transform @ matrix
    gas = np.zeros(matrix.shape[1])
    present = np.all((columns <= 0) | (total[:, None] > 0), axis=0)
    changing = np.any(columns[:, present] != 0, axis=1) & (total > 0)
    zero_neutral = np.zeros(len(model.neutrals))
    zero_ion = np.zeros(len(model.ions))
    options = tuple(reactions.values())
    adsorption = zero_ion.copy()
    ion_columns = transform[:, len(zero_neutral) :]
    charges = np.array([rxn.charge for rxn in options])
    q0 = conserved[len(zero_neutral) :] @ model.charge
    if surface is not None:
        sites = np.array([model.ions.index(k) for k in surface.log_binding])
        binding = np.array(list(surface.log_binding.values()))
        changing |= np.any(ion_columns[:, sites] != 0, axis=1) & (total > 0)
        qcap = surface.capacitance_f_m3 * R * temperature / FARADAY**2

    def balance(log_fraction):
        amounts = total.copy()
        phi, psi = log_fraction[-2:] if surface is not None else (0.0, 0.0)
        amounts[changing] *= np.exp(log_fraction[: sum(changing)])
        state = liquid(zero_neutral, zero_ion, conserved_moles=amounts)
        potentials = _log_activities(state.liquids.alpha)
        gas[:] = _gas_amounts(
            reactions, matrix, present, potentials, phi, temperature
        )
        if surface is not None:
            log_sites = (
                binding
                + potentials[len(zero_neutral) + sites]
                + model.charge[sites] * (phi - psi)
            )
            adsorption[sites] = surface.sites_mol_m3 * np.exp(
                log_sites - logsumexp(np.r_[0.0, log_sites])
            )
        residual = (
            amounts[changing]
            + columns[changing] @ gas
            + ion_columns[changing] @ adsorption
            - total[changing]
        ) / total[changing]
        if surface is not None:
            charge = adsorption @ model.charge
            residual = np.r_[
                residual,
                (charge + charges @ gas - q0) / surface.sites_mol_m3,
                (charge - psi * qcap) / max(qcap, surface.sites_mol_m3),
            ]
        return residual, state

    guess = np.zeros(sum(changing) + 2 * (surface is not None))
    if guess.size:
        # Start at the known physical inventory, testing progressively
        # depleted states in the log coordinates of the reacting solutes.
        fit = least_squares(
            lambda x: balance(x)[0],
            guess,
            bounds=(-75.0, 20.0),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=180,
        )
        guess = fit.x
    residual, state = balance(guess)
    error = float(np.max(abs(residual), initial=0.0))
    if not np.isfinite(error) or error > 1e-7:
        raise ConvergenceError(f"reactive gas balance residual {error:g}")
    phi, psi = guess[-2:] if surface is not None else (0.0, 0.0)
    volts = R * temperature / FARADAY
    return state, gas, adsorption, phi * volts, psi * volts, error


def _active_species(aqueous, matrix, gas0, n, surface0, dry):
    # Active analytical species include any reaction product made available
    # by a gas feed. A signed master inventory is valid for acid-base reactions.
    active = (aqueous > 0) | (np.abs(matrix) @ gas0 > 0)
    active[n:] |= (surface0 + dry) > 0
    while True:
        previous = active.copy()
        for column in matrix.T:
            if np.all(active[column > 0]):
                active[column < 0] = True
        if np.array_equal(active, previous):
            break
    gas_active = np.array(
        [np.all(active[matrix[:, j] != 0]) for j in range(matrix.shape[1])],
        dtype=bool,
    )
    return active, gas_active


def _validate_surface(surface, surface0, reactions, ions):
    if surface is None and (
        np.any(surface0) or any(g.charge for g in reactions.values())
    ):
        raise ValueError(
            "charged gas transfer requires an explicit ion surface model"
        )
    if surface is not None and (
        not np.isfinite(surface.sites_mol_m3)
        or not np.isfinite(surface.capacitance_f_m3)
        or surface.sites_mol_m3 <= 0
        or surface.capacitance_f_m3 <= 0
        or not surface.log_binding
        or not set(surface.log_binding) <= set(ions)
        or not np.all(np.isfinite(list(surface.log_binding.values())))
    ):
        raise ValueError(
            "invalid surface sites, capacitance or binding constants"
        )


@dataclass
class _SpeciesBudget:
    initial: SpeciesInventory
    model: object
    reactions: dict
    matrix: np.ndarray
    mw: np.ndarray
    gas_mw: np.ndarray
    gas0: np.ndarray
    aqueous: np.ndarray
    dry: np.ndarray
    surface0: np.ndarray
    conserved: np.ndarray

    def assemble(self, result, gas, adsorbed, residue, phi, psi, err):
        p = self.model
        ions = list(p.ions)
        n = len(p.neutrals)
        mw, gas_mw = self.mw, self.gas_mw
        inventory = self.initial
        gas_names = list(self.reactions)
        z = np.r_[np.zeros(n), p.charge]
        conserved = self.conserved
        charge0 = z @ conserved
        matrix = self.matrix
        output = SpeciesInventory(
            dict(
                zip(
                    result.neutral_ids,
                    result.neutral_masses.sum(axis=0),
                    strict=True,
                )
            ),
            dict(
                zip(result.ion_ids, result.ion_masses.sum(axis=0), strict=True)
            ),
            dict(zip(result.solid_names, result.solid_masses, strict=True)),
            dict(zip(gas_names, gas * gas_mw * 1e9, strict=True)),
            dict(zip(ions, adsorbed * mw[n:] * 1e9, strict=True)),
            dict(zip(ions, residue * mw[n:] * 1e9, strict=True)),
        )
        imbalance = abs(output.total_mass - inventory.total_mass) / max(
            inventory.total_mass, 1e-300
        )
        qout = sum(
            result.ion_masses.sum(axis=0)
            * 1e-9
            / _ion_mw(result.ion_ids)
            * np.array(
                [_tables()["Ioncharge"][i - 201] for i in result.ion_ids]
            )
        )
        qout += p.charge @ (adsorbed + residue) + (z @ matrix) @ gas
        charge_error = float(qout - charge0)
        if imbalance > 1e-7 or abs(charge_error) > 1e-8 * max(
            np.abs(conserved) @ np.abs(z),
            np.sum(result.ion_masses * 1e-9 / _ion_mw(result.ion_ids)),
            1e-300,
        ):
            raise ConvergenceError(
                "species equilibrium failed mass or charge closure"
            )
        return SpeciesEquilibrium(
            output, result, imbalance, charge_error, phi, psi, err
        )


def _prepare_budget(provider, inventory, reactions, temperature, solids):
    from .solids import solid_parameters

    p = provider.prepared
    neutral_names = list(p.neutrals)
    ions = list(p.ions)
    keys = neutral_names + ions
    n = len(neutral_names)
    mw = np.r_[p.neutral_mw, _ion_mw(ions)]
    z = np.r_[np.zeros(n), p.charge]
    if p.carbonate and p.carbonate_model == "pinned":
        carbon_name = p.neutrals[p.co2]
        if any(carbon_name in gas.aqueous for gas in reactions.values()):
            raise ValueError(
                "CO2 gas transfer requires the molal Gibbs-consistent reference"
            )
    matrix = _reaction_matrix(reactions, keys, z, temperature)
    gas_names = list(reactions)
    gas_mw = mw @ matrix
    if np.any(gas_mw <= 0):
        raise ValueError("gas reaction must have positive molecular mass")
    gas0 = _mass_vector(inventory.gases, gas_names) * 1e-9 / gas_mw
    aqueous = (
        np.r_[
            _mass_vector(inventory.liquid_neutrals, neutral_names),
            _mass_vector(inventory.liquid_ions, ions),
        ]
        * 1e-9
        / mw
    )
    dry = _mass_vector(inventory.residual_ions, ions) * 1e-9 / mw[n:]
    surface0 = _mass_vector(inventory.surface_ions, ions) * 1e-9 / mw[n:]
    # Input salts carry physical mass only while solid.
    solid_masses = _mass_vector(inventory.solids, list(solids))
    _, formulas = solid_parameters(provider, solids)
    if len(formulas):
        aqueous[n:] += (solid_masses * 1e-9 / (formulas @ mw[n:])) @ formulas
    conserved = aqueous + matrix @ gas0
    conserved[n:] += dry + surface0

    return _SpeciesBudget(
        inventory,
        p,
        reactions,
        matrix,
        mw,
        gas_mw,
        gas0,
        aqueous,
        dry,
        surface0,
        conserved,
    )


def equilibrate_species(
    provider,
    *,
    species_inventory,
    temperature,
    gas_reactions=None,
    solids=None,
    surface=None,
    execution="cpu",
    device=None,
    phase_mode="stable",
):
    """Equilibrate a closed parcel of material per m³ of air at specified T.

    Solid and aqueous subproblems run on the selected liquid backend. The gas
    and interfacial equations are solved on the host. Fixed-volume, ideal-gas,
    instantaneous internal equilibrium and a common temperature are assumed.
    """
    from .solids import solve_ionic_solid_liquid

    p = provider.prepared
    inventory = species_inventory
    if not isinstance(inventory, SpeciesInventory):
        raise TypeError("species_inventory must be a SpeciesInventory")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    reactions = dict(gas_reactions or {})
    solids = dict(solids or {})
    budget = _prepare_budget(
        provider, inventory, reactions, temperature, solids
    )
    n = len(p.neutrals)
    neutral_names, ions = list(p.neutrals), list(p.ions)
    aqueous, dry, surface0 = budget.aqueous, budget.dry, budget.surface0
    conserved, matrix = budget.conserved, budget.matrix
    gas0, mw, gas_mw = budget.gas0, budget.mw, budget.gas_mw
    _validate_surface(surface, surface0, reactions, ions)
    assemble = budget.assemble

    def liquid(neutral, ion, **extra):
        return solve_ionic_solid_liquid(
            provider,
            {} if extra else dict(zip(neutral_names, neutral, strict=True)),
            {} if extra else dict(zip(ions, ion, strict=True)),
            temperature,
            solids,
            execution=execution,
            device=device,
            phase_mode=phase_mode,
            **extra,
        )

    # Direct neutral vapors can equilibrate at the vapor-only boundary.
    neutral_gases = _direct_vapor_names(reactions, neutral_names)
    if (
        reactions
        and surface is None
        and len(neutral_gases) == len(reactions)
        and not p.carbonate
        # A bisulfate recipe with no ionic inventory has only neutral vapors
        # and solvents. A positive-liquid balance can miss complete evaporation.
        and not (p.sulfate and np.any(conserved[n:]))
    ):
        partition = _neutral_partition(
            provider,
            conserved,
            matrix,
            mw,
            gas_mw,
            reactions,
            neutral_gases,
            temperature,
            dict(
                phase_mode=phase_mode,
                solids=solids,
                execution=execution,
                device=device,
            ),
            liquid,
        )
        if partition is not None:
            result, gas, error = partition
            residue = result.dry_ion_masses * 1e-9 / mw[n:]
            return assemble(
                result, gas, np.zeros_like(dry), residue, 0.0, 0.0, error
            )

    if not np.any(aqueous[:n]) and not any(
        any(k in neutral_names for k in g.aqueous) and gas0[j]
        for j, g in enumerate(reactions.values())
    ):
        # Aqueous gas laws have no liquid reference after complete dry-out.
        a = aqueous[n:] + dry + surface0
        result = liquid(aqueous[:n], a)
        residue = result.dry_ion_masses * 1e-9 / mw[n:]
        return assemble(result, gas0, np.zeros_like(a), residue, 0.0, 0.0, 0.0)

    if (p.sulfate or p.carbonate) and (reactions or surface is not None):
        z = np.r_[np.zeros(n), p.charge]
        charge0 = z @ conserved
        if surface is None and abs(charge0) > 5e-12 * max(
            abs(z) @ abs(conserved), 1e-300
        ):
            raise ValueError(
                "the bulk reactive inventory must be electroneutral"
            )
        result, gas, ads, phi, psi, error = _reactive_vapors(
            p, liquid, conserved, reactions, matrix, temperature, surface
        )
        return assemble(result, gas, ads, np.zeros_like(dry), phi, psi, error)

    return _aqueous_gas_exchange(budget, surface, temperature, liquid)


def _surface_lanes(surface, ions, n, active):
    surface_idx = np.array(
        [
            n + ions.index(i)
            for i in (surface.log_binding if surface else ())
            if active[n + ions.index(i)]
        ],
        dtype=int,
    )
    if surface is not None and not len(surface_idx):
        raise ValueError("surface model has no occupied binding species")
    return surface_idx


def _aqueous_gas_exchange(budget, surface, temperature, liquid):
    """Balance unreacted aqueous transfers and any finite interfacial sites."""
    p = budget.model
    ions = list(p.ions)
    keys = list(p.neutrals) + ions
    n = len(p.neutrals)
    z = np.r_[np.zeros(n), p.charge]
    aqueous, dry, surface0 = budget.aqueous, budget.dry, budget.surface0
    conserved, matrix, gas0 = budget.conserved, budget.matrix, budget.gas0
    reactions = budget.reactions
    assemble = budget.assemble
    active, gas_active = _active_species(
        aqueous, matrix, gas0, n, surface0, dry
    )
    pos = np.flatnonzero(active & (z > 0))
    neg = np.flatnonzero(active & (z < 0))
    neutral = np.flatnonzero(active & (z == 0))
    if not np.any(gas_active) and surface is None:
        balanced = conserved[n:]
        result = liquid(conserved[:n], balanced)
        return assemble(
            result, gas0, surface0, np.zeros_like(dry), 0.0, 0.0, 0.0
        )
    if bool(len(pos)) != bool(len(neg)):
        raise ValueError(
            "wet gas/ion coupling requires both cation and anion inventories"
        )
    surface_idx = _surface_lanes(surface, ions, n, active)
    charged = bool(len(surface_idx))
    selected = np.r_[neutral, pos, neg]
    totalscale = np.maximum(np.abs(conserved), np.abs(matrix) @ gas0)
    scale = np.maximum(totalscale, max(totalscale.max(), 1e-300) * 1e-10)
    baseline = np.maximum(aqueous, scale * 0.05)
    independent = np.r_[neutral, pos]
    guess = np.r_[
        np.log(baseline[independent] / scale[independent]),
        np.log(
            (baseline[neg] * abs(z[neg]))[:-1]
            / (baseline[neg[-1]] * abs(z[neg[-1]]))
        )
        if len(neg)
        else np.empty(0),
    ]
    if charged:
        guess = np.r_[guess, 0.0, 0.0]
    zeros = np.zeros(len(ions))

    def evaluate(v):
        b = np.zeros_like(conserved)
        b[independent] = scale[independent] * np.exp(v[: len(independent)])
        weights = np.r_[
            v[len(independent) : len(independent) + len(neg) - 1], 0.0
        ]
        if len(neg):
            b[neg] = (
                np.exp(weights - logsumexp(weights))
                * (b[pos] @ z[pos])
                / (-z[neg])
            )
        result = liquid(b[:n], b[n:])
        loga = _log_activities(result.liquids.alpha)
        phi, psi = v[-2:] if charged else (0.0, 0.0)
        gas = _gas_amounts(
            reactions, matrix, gas_active, loga, phi, temperature
        )
        ads = zeros.copy()
        if charged:
            kads = np.array([surface.log_binding[keys[k]] for k in surface_idx])
            logs = kads + loga[surface_idx] + z[surface_idx] * (phi - psi)
            ads[surface_idx - n] = surface.sites_mol_m3 * np.exp(
                logs - logsumexp(np.r_[0.0, logs])
            )
        remainder = b + matrix @ gas - conserved
        remainder[n:] += ads
        constraints = remainder[selected] / scale[selected]
        if charged:
            qcap = surface.capacitance_f_m3 * R * temperature / FARADAY**2
            constraints = np.r_[
                constraints,
                (ads @ p.charge - psi * qcap) / max(qcap, surface.sites_mol_m3),
            ]
        elif len(neg):
            constraints = constraints[
                :-1
            ]  # last anion balance follows electroneutrality
        return constraints, result, gas, ads, phi, psi

    fit = least_squares(
        lambda v: evaluate(v)[0],
        guess,
        bounds=(-100.0, 100.0),
        xtol=1e-12,
        gtol=1e-12,
        ftol=1e-12,
        max_nfev=240,
        x_scale="jac",
    )
    residual, result, gas, ads, phi, psi = evaluate(fit.x)
    err = float(np.max(abs(residual)))
    if not np.isfinite(err) or err > 1e-7:
        raise ConvergenceError(
            f"unresolved gas/surface/solid equilibrium: scaled residual {err:g}"
        )
    return assemble(
        result,
        gas,
        ads,
        zeros,
        phi * R * temperature / FARADAY,
        psi * R * temperature / FARADAY,
        err,
    )
