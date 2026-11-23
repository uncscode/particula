"""Equilibrate anhydrous crystals with AIOMFAC liquids on a molal ion basis."""

from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

import numpy as np
from scipy.optimize import brentq, linprog

from ._equilibrium import ActivityDomainError, ConvergenceError
from .model import _tables
from .phases import LiquidEquilibrium, component_molar_masses


@dataclass(frozen=True)
class SolidSalt:
    """Dissolution log10(Ksp) on the 1 mol/kg solvent scale, valid over [K, K].

    Stoichiometry comes from the declared analytical salt. A constant over a
    finite interval is an explicit caller approximation. The pure crystal
    standard state has unit activity. Hydrated and multication crystals
    require different reaction quotients.
    """

    log10_ksp: float
    temperature_min_k: float = 298.15
    temperature_max_k: float = 298.15

    def constant(self, temperature):
        """Return log10(Ksp) after checking the temperature domain."""
        values = [
            self.log10_ksp,
            self.temperature_min_k,
            self.temperature_max_k,
        ]
        if not np.isfinite(values).all() or not (
            0 < self.temperature_min_k <= temperature <= self.temperature_max_k
        ):
            raise ValueError(
                "temperature outside the specified solid Ksp domain"
            )
        return self.log10_ksp


# USGS PHREEQC pitzer.dat, Halite and Sylvite dissolution, 25 °C.
# https://github.com/phreeqc-dev/phreeqc3/blob/master/database/pitzer.dat
_MINERALS = {
    "halite": ({202: 1, 242: 1}, SolidSalt(1.57)),
    "sylvite": ({203: 1, 242: 1}, SolidSalt(0.90)),
}


@dataclass(frozen=True)
class SolidLiquidEquilibrium:
    """Mass concentrations in µg/m³ with alpha and beta liquid rows.

    The dissolved analytical vector is an alternative description of the
    physical neutral and ion masses; they must not be added together.
    Saturation is log10(IAP/Ksp), undefined without a liquid. Dry residual
    ions have no assigned crystalline form. The charge error is relative to
    total absolute ionic equivalents in each nonempty physical phase.
    """

    liquids: LiquidEquilibrium | None
    component_names: tuple
    dissolved_masses: np.ndarray
    solid_names: tuple
    solid_masses: np.ndarray
    neutral_ids: tuple
    neutral_masses: np.ndarray
    ion_ids: tuple
    ion_masses: np.ndarray
    dry_ion_masses: np.ndarray
    log_saturation: np.ndarray
    complementarity_error: float
    charge_error: float

    @property
    def residual_ion_masses(self):
        """Dry ion mass with no assigned crystalline phase, in µg/m³."""
        return self.dry_ion_masses


def _ion_mw(model):
    tables = _tables()
    ids = np.array(model.ions, dtype=int)
    return (
        0.001
        * np.r_[
            tables["SMWC"][ids[: model.ncation] - 201],
            tables["SMWA"][ids[model.ncation :] - 241],
        ]
    )


def solid_parameters(provider, solids):
    """Resolve constants and mineral stoichiometry in prepared ion order."""
    model = provider.prepared
    recipe = dict(provider.mixture.components)
    formulas, constants = [], []
    for name, spec in solids.items():
        groups = recipe.get(name)
        if isinstance(spec, str):
            if spec not in _MINERALS:
                raise ValueError(f"unknown solid reference {spec!r}")
            reference, spec = _MINERALS[spec]
            if groups is not None and reference != groups:
                raise ValueError(
                    f"{name!r} differs from the specified mineral formula"
                )
            groups = reference
        if (
            not isinstance(spec, SolidSalt)
            or not groups
            or not set(groups) <= set(model.ions)
        ):
            raise ValueError(
                f"{name!r} requires a declared ion pair and SolidSalt"
            )
        if len(groups) != 2 or groups in formulas:
            raise ValueError(
                "solid formulas must be distinct anhydrous cation–anion pairs"
            )
        formulas.append(groups)
        spec.constant(spec.temperature_min_k)
        constants.append(spec)
    stoich = np.asarray(
        [[f.get(j, 0) for j in model.ions] for f in formulas], dtype=float
    ).reshape(len(formulas), len(model.ions))
    return tuple(constants), stoich


_PREPARED_BASES = WeakKeyDictionary()


def _expanded_basis(provider):
    """Complete the pair basis for residual ions after transfer."""
    model = provider.prepared
    recipe = dict(provider.mixture.components)
    cached = _PREPARED_BASES.get(provider)
    if cached is not None:
        return cached
    # Rebind only the analytical map. Re-preparing all implicit acid/base
    # species as a new chemical recipe could enable a different reaction set.
    salt_rows = []
    salt_lanes = []
    salt_mw = []
    for lane, nu, mw in zip(
        model.salt_indices, model.stoich, model.salt_mw, strict=True
    ):
        if lane < model.ncomponents:
            salt_lanes.append(lane)
            salt_rows.append(nu)
            salt_mw.append(mw)
    neutral_lanes = model.neutral_indices.copy()
    for j, name in enumerate(model.neutrals):
        if neutral_lanes[j] >= model.ncomponents:
            neutral_lanes[j] = len(recipe)
            recipe[name] = {
                173: 1
            }  # the only implicit neutral is dissolved CO₂
    ion_mw = _ion_mw(model)
    for pair, counts in zip(model.pairs, model.pair_counts, strict=True):
        formula = dict(zip(pair, map(int, counts), strict=True))
        if formula in recipe.values():
            continue
        name = f"ion pair {pair[0]}/{pair[1]}"
        if name in recipe:
            raise ValueError(f"reserved analytical name collision: {name}")
        nu = np.array([formula.get(j, 0) for j in model.ions])
        salt_lanes.append(len(recipe))
        salt_rows.append(nu)
        salt_mw.append(nu @ ion_mw)
        recipe[name] = formula
    if len(recipe) != model.ncomponents:
        model = replace(
            model,
            ncomponents=len(recipe),
            virtual_count=0,
            neutral_indices=neutral_lanes,
            salt_indices=np.array(salt_lanes),
            stoich=np.asarray(salt_rows),
            salt_mw=np.asarray(salt_mw),
        )
    result = (
        model,
        tuple(recipe),
        _reaction_basis(model, provider.prepared.ncomponents),
    )
    _PREPARED_BASES[provider] = result
    return result


def _mineral_basis(provider, solids, temperature, execution, device):
    model, names, projection = _expanded_basis(provider)
    specs, stoich = solid_parameters(provider, solids)
    constants = np.asarray([spec.constant(temperature) for spec in specs])
    if model is provider.prepared:
        backend = provider.equilibrium_backend(execution, device)
    elif execution == "cpu" and device in (None, "cpu"):
        backend = model
    elif execution == "cuda":
        from ._cuda import CudaModel

        backend = CudaModel(model, device or "cuda:0")
    else:
        raise ValueError("device must match the selected execution backend")
    return model, backend, names, stoich, constants, projection


def conserved_species_matrix(model):
    """Map physical mol amounts to independent, nonnegative aqueous inventories.

    Columns follow prepared neutrals, then prepared ions. Conserved charge is
    checked separately. Carbonate hydration binds one water in bicarbonate
    or carbonate; hydroxide also carries one equivalent of solvent water.
    Choosing CO2 and sulfate as conserved reference forms makes all entries
    nonnegative even when net acid equivalents are negative.
    """
    n = len(model.neutrals)
    matrix = np.eye(n + len(model.ions))

    def ion(j):
        return n + model.ions.index(j)

    if model.sulfate:
        matrix[ion(261), ion(248)] += 1
        matrix[ion(248), :] = 0
    if model.carbonate:
        for j in (250, 262):
            matrix[model.co2, ion(j)] += 1
            matrix[model.water, ion(j)] += 1
            matrix[ion(j), :] = 0
        matrix[model.water, ion(247)] += 1
        matrix[ion(247), :] = 0
    if model.sulfate or model.carbonate:
        matrix[ion(205), :] = 0
    return matrix[np.any(matrix, axis=1)]


@dataclass
class _ReactiveProjection:
    analytical: np.ndarray
    costs: np.ndarray
    molar_masses: np.ndarray
    original_inverse: np.ndarray

    def mass(self, amounts):
        # Cap each candidate by its own limiting conserved inventory, then
        # nondimensionalize each equality separately to retain trace inputs.
        d = self.analytical
        # A returned speciation state normally belongs to the original recipe.
        # Recover that simple representation before searching auxiliary pairs.
        original = self.original_inverse @ amounts
        if np.all(original >= 0) and np.allclose(
            d[:, : len(original)] @ original, amounts, rtol=2e-12, atol=0
        ):
            mass = np.zeros(d.shape[1])
            mass[: len(original)] = (
                original * self.molar_masses[: len(original)] * 1e9
            )
            return mass
        upper = np.min(
            np.divide(
                amounts[:, None], d, out=np.full_like(d, np.inf), where=d > 0
            ),
            axis=0,
        )
        active = np.flatnonzero(upper > 0)
        present = amounts > 0
        x = np.zeros(d.shape[1])
        if active.size:
            reduced = (
                d[present][:, active] * upper[active] / amounts[present, None]
            )
            fit = linprog(
                self.costs[active],
                A_eq=reduced,
                b_eq=np.ones(sum(present)),
                bounds=(0, 1),
                options={"primal_feasibility_tolerance": 1e-10},
            )
            if not fit.success:
                raise ActivityDomainError(
                    "no feasible analytical composition for this inventory"
                )
            x[active] = fit.x * upper[active]
            if np.max(np.abs((d @ x)[present] / amounts[present] - 1)) > 2e-9:
                raise ConvergenceError(
                    "physical-to-analytical inventory projection failed"
                )
        return x * self.molar_masses * 1e9


def _reaction_basis(model, original_width):
    n = len(model.neutrals)
    c = conserved_species_matrix(model)
    analytical = np.zeros((n + len(model.ions), model.ncomponents))
    analytical[np.arange(n), model.neutral_indices] = 1
    analytical[n:, model.salt_indices] = model.stoich.T
    # Among physically equivalent decompositions, prefer the supplied input
    # materials to auxiliary acid/base and cross-salt coordinates.
    costs = np.ones(model.ncomponents)
    costs[original_width:] = 100.0 * model.ncomponents
    inventory = c @ analytical
    return c, _ReactiveProjection(
        inventory,
        costs,
        component_molar_masses(model),
        np.linalg.pinv(inventory[:, :original_width]),
    )


def _ordered(mapping, names, kind):
    unknown = set(mapping) - set(names)
    if unknown:
        raise ValueError(f"undeclared {kind} identifiers: {unknown}")
    values = np.asarray([mapping.get(k, 0.0) for k in names], dtype=float)
    if values.shape != (len(names),) or np.any(
        ~np.isfinite(values) | (values < 0)
    ):
        raise ValueError(
            f"{kind} inventories must be nonnegative finite mol/m³"
        )
    return values


def _dry_extent(ions, stoich, constants):
    """Assign dry ions to crystals, minimizing Gibbs energy among ties."""
    if not len(stoich) or not np.any(ions):
        return np.zeros(len(stoich)), ions.copy()
    # Scale by each mineral's limiting ion and each row's initial supply.
    # A trace crystal must never be allowed to consume a macroscopic excess.
    limits = np.array([np.min(ions[nu > 0] / nu[nu > 0]) for nu in stoich])
    extent = np.zeros(len(stoich))
    ids = np.flatnonzero(limits)
    used = ions > 0
    if ids.size == 1:
        extent[ids] = limits[ids]
    elif ids.size:
        capacities = stoich[ids] * limits[ids, None]
        constraints = capacities[:, used].T / ions[used, None]
        weights = capacities.sum(axis=1)
        weights /= weights.max()
        settings = dict(
            A_ub=constraints, b_ub=np.ones(sum(used)), bounds=(0, 1)
        )
        first = linprog(-weights, **settings)
        if not first.success:
            raise ConvergenceError("could not close the dry ion balance")
        energy = constants[ids] * (limits[ids] / max(limits[ids]))
        best = linprog(
            energy, A_eq=weights[None, :], b_eq=[-first.fun], **settings
        )
        if not best.success:
            raise ConvergenceError(
                "could not select a feasible dry mineral assemblage"
            )
        extent[ids] = best.x * limits[ids]
    residual = ions - extent @ stoich
    # Resolve trace inventories below the optimizer's objective tolerance.
    for j in np.argsort(constants, kind="stable"):
        bound = stoich[j] > 0
        remaining = max(0.0, np.min(residual[bound] / stoich[j, bound]))
        extent[j] += remaining
        residual -= remaining * stoich[j]
    if np.any(residual < -1e-10 * np.maximum(ions, np.finfo(float).tiny)):
        raise ConvergenceError("negative dry residual ion inventory")
    return extent, np.maximum(residual, 0.0)


@dataclass
class _SolidSystem:
    model: object
    backend: object
    temperature: float
    water_index: int
    total: np.ndarray
    stoich: np.ndarray
    constants: np.ndarray
    transfers: np.ndarray
    projection: _ReactiveProjection | None = None

    def analytical(self, free):
        if self.projection is not None:
            return self.projection.mass(free)
        model = self.model
        # Greedy pairing consumes the limiting ion at each step. The complete
        # cation × anion basis guarantees closure for any neutral inventory.
        mass = self.total.copy()
        left = free.copy()
        for slot, lane in enumerate(model.salt_indices):
            if lane >= len(self.total):
                continue
            nu = model.stoich[slot]
            idx = nu > 0
            amount = np.min(left[idx] / nu[idx])
            mass[lane] = amount * model.salt_mw[slot] * 1e9
            left[idx] = np.maximum(left[idx] - amount * nu[idx], 0)
        if np.sum(left) > 5e-11 * np.sum(free):
            raise ValueError(
                "remaining ions cannot be expressed by the declared ion basis"
            )
        return mass

    def liquid(self, free, stable):
        mass = self.analytical(free)
        if stable:
            phases = self.backend.phase_equilibrium(
                mass, self.temperature, self.water_index
            )
        else:
            state = self.backend.evaluate(
                mass, self.temperature, diagnostics=True
            )
            phases = LiquidEquilibrium(mass, None, state, None, 0.0, np.nan)
        return mass, phases

    def saturation_for(self, state):
        with np.errstate(divide="ignore"):
            log_a = np.where(
                state.ion_defined,
                np.log10(state.ion_molalities)
                + state.ion_log_coefficients / np.log(10),
                -np.inf,
            )
        return (
            np.asarray([nu[nu > 0] @ log_a[nu > 0] for nu in self.stoich])
            - self.constants
        )

    @staticmethod
    def complementarity(si, precip):
        active = precip > 0
        return np.max(np.where(active, abs(si), np.maximum(si, 0)), initial=0.0)

    def _transfer(self, j, nu, available, stable):
        used = nu > 0
        limit = np.min(available[used] / nu[used])
        if limit == 0:
            return available, 0.0
        base = np.maximum(available - limit * nu, 0.0)

        def affinity(amount):
            if amount == 0:
                # A limiting independent ion is absent at this endpoint.
                return -1e10
            _, phases = self.liquid(base + amount * nu, stable)
            return self.saturation_for(phases.alpha)[j]

        lo = 0.0
        hi = min(limit, max(self.total.sum() * 1e-9, np.finfo(float).tiny))
        value = affinity(hi)
        while value < 0 and hi < limit:
            lo, hi = hi, min(2 * hi, limit)
            value = affinity(hi)
        amount = (
            hi
            if value <= 0
            else brentq(
                affinity, lo, hi, xtol=np.nextafter(0.0, 1.0), rtol=1e-12
            )
        )
        return base + amount * nu, limit - amount

    def equilibrate(self, stable, free, precip):
        # Cyclic coordinate solves preserve shared-ion inventories.
        if not len(self.stoich):
            return free, precip
        for _ in range(100):
            for j, nu in enumerate(self.transfers):
                free, precip[j] = self._transfer(
                    j, nu, free + precip[j] * nu, stable
                )
            _, phases = self.liquid(free, stable)
            if (
                self.complementarity(self.saturation_for(phases.alpha), precip)
                <= 1e-7
            ):
                return free, precip
        raise ConvergenceError("solid-liquid complementarity did not converge")

    def wet(self, ions, stable):
        # Refine the homogeneous solution against the stable liquids.
        free, precip = self.equilibrate(
            False, ions.copy(), np.zeros(len(self.stoich))
        )
        mass, phases = self.liquid(free, stable)
        saturation = self.saturation_for(phases.alpha)
        if stable and self.complementarity(saturation, precip) > 1e-7:
            free, precip = self.equilibrate(True, free, precip)
            mass, phases = self.liquid(free, True)
            saturation = self.saturation_for(phases.alpha)
        return mass, phases, saturation, precip


def _physical_liquids(
    system, phases, precip, ionic_mw, phase_neutrals, phase_ions
):
    model = system.model
    charge_error = error = 0.0
    for row, (phase, amounts) in enumerate(
        (
            (phases.alpha, phases.alpha_masses),
            (phases.beta, phases.beta_masses),
        )
    ):
        if phase is None:
            continue
        diag = phase.diagnostics or {}
        if diag.get("precipitated_caso4_mol_per_kg_input", 0.0) > 0:
            raise ActivityDomainError(
                "upstream CaSO4 removal requires explicit solid accounting"
            )
        scale = amounts.sum()
        species = phase.species_moles_per_kg_input
        mol = species[len(model.neutrals) :]
        phase_neutrals[row] = (
            scale * species[: len(model.neutrals)] * model.neutral_mw
        )
        phase_ions[row] = scale * mol * ionic_mw
        denominator = mol @ abs(model.charge)
        if denominator > 0:
            charge_error = max(
                charge_error, abs(mol @ model.charge) / denominator
            )
        error = max(
            error, system.complementarity(system.saturation_for(phase), precip)
        )
    if error > 2e-7 or charge_error > 1e-10:
        raise ConvergenceError(
            "solid saturation or charge residual exceeded tolerance"
        )
    return charge_error, error


def _initial_species(model, neutral_moles, ion_moles, conserved_moles, basis):
    nm = _ordered(neutral_moles, model.neutrals, "neutral")
    ions = _ordered(ion_moles, model.ions, "ion")
    c, projection = basis
    free = None
    if conserved_moles is not None:
        if neutral_moles or ion_moles:
            raise ValueError("supply either physical or conserved inventories")
        free = np.asarray(conserved_moles, dtype=float)
        if free.shape != (len(c),) or np.any(~np.isfinite(free) | (free < 0)):
            raise ValueError("invalid conserved molar inventory")
        projected = projection.mass(free) / (
            component_molar_masses(model) * 1e9
        )
        nm = projected[model.neutral_indices]
        ions = projected[model.salt_indices] @ model.stoich
    return nm, ions, free


def solve_ionic_solid_liquid(
    provider,
    neutral_moles,
    ion_moles,
    temperature,
    solids,
    *,
    execution="cpu",
    device=None,
    phase_mode="stable",
    conserved_moles=None,
):
    """Equilibrate neutral and ionic inventories in mol/m³.

    Map neutral names and integer AIOMFAC ion IDs to mol/m³. Wet inputs
    must be charge balanced. Dry net charge remains on residual ions. The
    solids mapping assigns each physical solid a built-in mineral name or a
    SolidSalt constant for a declared salt. Ions may recombine across input
    salts. At zero solvent, residual ions retain their chemical identities.
    Alternatively, supply conserved_moles in conserved_species_matrix row
    order for an electroneutral inventory after a gas or interfacial trial.
    In that case pass empty neutral_moles and ion_moles mappings.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive in K")
    if phase_mode not in ("stable", "single_liquid"):
        raise ValueError("unknown liquid phase mode")
    model, backend, names, stoich, constants, basis = _mineral_basis(
        provider, solids, temperature, execution, device
    )
    nm, ions, free = _initial_species(
        model, neutral_moles, ion_moles, conserved_moles, basis
    )
    c, projection = basis
    ionic_mw = _ion_mw(model)
    solvent = nm * model.neutral_mw * 1e9
    charge_scale = ions @ np.abs(model.charge)
    if np.any(solvent) and abs(ions @ model.charge) > 5e-12 * charge_scale:
        raise ValueError("the supplied ion inventory must be electroneutral")
    salt_mw = stoich @ ionic_mw
    total = np.zeros(len(names))
    real_neutrals = model.neutral_indices < len(names)
    if np.any(nm[~real_neutrals] > 0):
        raise ValueError("declare neutral inputs explicitly in MixtureSpec")
    total[model.neutral_indices[real_neutrals]] = solvent[real_neutrals]
    dry_ions = np.zeros_like(ions)
    phase_neutrals = np.zeros((2, len(nm)))
    phase_ions = np.zeros((2, len(ions)))
    charge_error = 0.0
    saturation = np.full(len(stoich), np.nan)
    error = 0.0

    if not np.any(solvent):
        precip, dry_ions = _dry_extent(ions, stoich, constants)
        mass, phases = total, None
    else:
        if model.sulfate or model.carbonate or conserved_moles is not None:
            if conserved_moles is None:
                free = c @ np.r_[nm, ions]
            transfers = stoich @ c[:, len(nm) :].T
        else:
            free, transfers, projection = ions, stoich, None
        system = _SolidSystem(
            model,
            backend,
            temperature,
            provider.water_index,
            total,
            stoich,
            constants,
            transfers,
            projection,
        )
        mass, phases, saturation, precip = system.wet(
            free, phase_mode == "stable"
        )
        charge_error, error = _physical_liquids(
            system, phases, precip, ionic_mw, phase_neutrals, phase_ions
        )
    return SolidLiquidEquilibrium(
        phases,
        names,
        mass,
        tuple(solids),
        precip * salt_mw * 1e9,
        model.neutrals,
        phase_neutrals,
        model.ions,
        phase_ions,
        dry_ions * ionic_mw * 1e9,
        saturation,
        float(error),
        float(charge_error),
    )


def solve_solid_liquid(
    provider,
    inventories,
    temperature,
    solids,
    *,
    execution="cpu",
    device=None,
    phase_mode="stable",
):
    """Equilibrate neutral and salt analytical mass concentrations in µg/m³."""
    model = provider.prepared
    mass = np.asarray(inventories, dtype=float)
    if mass.shape != (model.ncomponents,) or np.any(
        ~np.isfinite(mass) | (mass < 0)
    ):
        raise ValueError(
            "supply one nonnegative finite mass for each analytical component"
        )
    amounts = np.pad(
        mass / (component_molar_masses(model) * 1e9), (0, model.virtual_count)
    )
    ions = amounts[model.salt_indices] @ model.stoich
    return solve_ionic_solid_liquid(
        provider,
        dict(zip(model.neutrals, amounts[model.neutral_indices], strict=True)),
        dict(zip(model.ions, ions, strict=True)),
        temperature,
        solids,
        execution=execution,
        device=device,
        phase_mode=phase_mode,
    )
