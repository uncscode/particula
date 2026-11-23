"""Prepared direct activities for analytical dissolved-component mixtures."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files
from math import gcd
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._cuda import CudaModel

import numpy as np

from . import UPSTREAM_REVISION, UPSTREAM_VERSION
from ._equilibrium import ActivityDomainError, ConvergenceError, solve
from ._terms import electrolyte_terms, short_range


class SpeciationRequiredError(ValueError):
    """The declared system needs an equilibrium solver."""


@dataclass(frozen=True)
class MixtureSpec:
    """Ordered mapping of component names to positive integer subgroup counts.

    Each salt is one formula unit with one cation and one anion subgroup.
    """

    components: Mapping

    def __post_init__(self):
        clean = {}
        for name, groups in self.components.items():
            if not isinstance(name, str) or not name.strip() or not groups:
                raise ValueError(f"invalid component {name!r}")
            for group, count in groups.items():
                if any(type(i) is not int or i < 1 for i in (group, count)):
                    raise ValueError(
                        f"{name}: expected positive integer subgroup IDs and counts"
                    )
            clean[name] = MappingProxyType(dict(groups))
        object.__setattr__(self, "components", MappingProxyType(clean))


@dataclass(frozen=True)
class ActivityResult:
    """Arrays preserve leading batch axes. Final axes follow the accompanying IDs.

    Diagnostics SR/MR/LR follow neutral_ids then ion_ids. Ion molalities are
    mol/kg solvent. Ionic coefficients and products use unit standard molality.
    Undefined log coefficients and means are NaN and absent activities zero.
    Final species amounts are mol per kg of analytical input, in neutral then
    ion order. Solver status is zero for fixed or degenerate inventories and
    one for a verified equilibrium. Residual diagnostics give maximum absolute
    log mass-action error, normalized charge, carbon and sulfur errors.
    """

    neutral_ids: tuple
    ion_ids: tuple
    electrolyte_ids: tuple
    neutral_mole_fractions: np.ndarray
    neutral_activities: np.ndarray
    neutral_log_coefficients: np.ndarray
    ion_molalities: np.ndarray
    ion_log_coefficients: np.ndarray
    mean_electrolyte_coefficients: np.ndarray
    ion_activity_products: np.ndarray
    neutral_defined: np.ndarray
    ion_defined: np.ndarray
    electrolyte_defined: np.ndarray
    warnings: np.ndarray
    diagnostics: dict | None
    species_moles_per_kg_input: np.ndarray
    solver_status: np.ndarray
    model_identity: tuple = (UPSTREAM_VERSION, UPSTREAM_REVISION)


@lru_cache(maxsize=1)
def _tables():
    with (
        files(__package__).joinpath("data/parameters.npz").open("rb") as stream
    ):
        with np.load(stream) as bundle:
            return {key: bundle[key] for key in bundle.files}


@dataclass(frozen=True)
class PreparedModel:
    neutrals: tuple
    ions: tuple
    pairs: tuple
    estimated_pairs: tuple
    ncomponents: int
    ncation: int
    neutral_indices: np.ndarray
    salt_indices: np.ndarray
    neutral_mw: np.ndarray
    salt_mw: np.ndarray
    stoich: np.ndarray
    pair_counts: np.ndarray
    counts: np.ndarray
    q: np.ndarray
    area: np.ndarray
    volume: np.ndarray
    interaction: np.ndarray
    neutral_counts: np.ndarray
    group_mw: np.ndarray
    group_map: np.ndarray
    main_counts: np.ndarray
    charge: np.ndarray
    bgroup: np.ndarray
    cgroup: np.ndarray
    bpair: tuple
    cpair: tuple
    like: np.ndarray
    three: np.ndarray

    sulfate: bool = False
    carbonate: bool = False
    carbonate_model: str = "pinned"
    water: int = -1
    co2: int = -1
    co2_lambda: np.ndarray | None = None
    virtual_count: int = 0

    def evaluate(self, masses, temperature_K, *, diagnostics=False):
        """Calculate activities for scalar or batched masses in common mass units."""
        return self._evaluate_rows(
            masses, temperature_K, diagnostics, collect_failures=False
        )

    def phase_equilibrium(self, masses, temperature, water_index):
        """Resolve coexisting liquids for one inventory in declaration order."""
        from .phases import phase_equilibrium

        return phase_equilibrium(self, masses, temperature, water_index)

    def _evaluate_rows(
        self, masses, temperature_K, diagnostics, *, collect_failures
    ):
        """Optionally retain per-node numerical failures for offline refinement."""
        mass = np.asarray(masses, dtype=np.float64)
        if mass.ndim == 0 or mass.shape[-1] != self.ncomponents:
            raise ValueError(
                f"expected {self.ncomponents} components on last axis"
            )
        shape = mass.shape[:-1]
        temp = np.broadcast_to(np.asarray(temperature_K, dtype=float), shape)
        n, i, s = len(self.neutrals), len(self.ions), len(self.pairs)

        def output(width, dtype=float):
            return np.empty((temp.size, width), dtype=dtype)

        xout, activity, lng = (output(n) for _ in range(3))
        mol, lngi = output(i), output(i)
        mean, iap = output(s), output(s)
        nm, im, sm = (output(k, bool) for k in (n, i, s))
        warnings = np.empty(temp.size, dtype=int)
        species_moles = output(n + i)
        status = np.zeros(temp.size, dtype=np.int32)
        terms = None
        if diagnostics:
            terms = {key: output(n + i) for key in ("sr", "mr", "lr")}
            if self.sulfate or self.carbonate:
                terms["equilibrium_residuals"] = output(4)
                terms["precipitated_caso4_mol_per_kg_input"] = np.empty(
                    temp.size
                )
                terms["evaluations"] = np.empty(temp.size, dtype=int)
                terms["retries"] = np.empty(temp.size, dtype=int)
            terms.update(
                ionic_strength=np.empty(temp.size),
                molality_conversion=np.empty(temp.size),
            )
        # Bound temporary group-ion and ion-pair tensors independently of population size.
        cost = 16 * 8 * (1 + self.counts.shape[1] * (n + i + 1) + i * i)
        chunk = max(1, min(1024, 4 * 1024**2 // cost))
        tiny = np.sqrt(np.finfo(float).tiny)
        for start in range(0, temp.size, chunk):
            stop = min(temp.size, start + chunk)
            rows = slice(start, stop)
            # flat slicing also works for strided masses and broadcast temperatures
            values = mass.flat[
                start * self.ncomponents : stop * self.ncomponents
            ].reshape(-1, self.ncomponents)
            tk = temp.flat[rows]

            def reject(bad, reason):
                where = np.flatnonzero(bad)
                if where.size:
                    idx = tuple(
                        map(int, np.unravel_index(start + where[0], shape))
                    )
                    raise ActivityDomainError(reason, batch_index=idx)

            reject(~np.isfinite(tk) | (tk <= 0), "invalid temperature")
            reject(
                ~np.isfinite(values).all(axis=1)
                | (values < 0).any(axis=1)
                | (values.max(axis=1) <= 0),
                "invalid dissolved masses",
            )
            fractions = values / values.max(axis=1, keepdims=True)
            fractions /= fractions.sum(axis=1, keepdims=True)
            if self.virtual_count:
                fractions = np.pad(fractions, ((0, 0), (0, self.virtual_count)))
            solvent = fractions[:, self.neutral_indices]
            solvent_mass = solvent.sum(axis=1, keepdims=True)
            reject(solvent_mass[:, 0] <= 0, "zero solvent mass")
            neutral_moles = solvent / self.neutral_mw
            ion_moles = (
                fractions[:, self.salt_indices] / self.salt_mw
            ) @ self.stoich
            if self.sulfate or self.carbonate:
                reject(
                    neutral_moles[:, self.water] <= 0,
                    "zero water in aqueous equilibrium",
                )
                for j in range(len(tk)):
                    try:
                        residuals, calls, retries, solid = solve(
                            self, neutral_moles[j], ion_moles[j], tk[j]
                        )
                    except (
                        ValueError,
                        RuntimeError,
                        FloatingPointError,
                    ) as error:
                        if collect_failures:
                            # Discard partially speciated trial amounts. A benign
                            # pure-water placeholder keeps other rows vectorized.
                            # The builder excludes this row using its status.
                            status[start + j] = -5
                            neutral_moles[j] = 0.0
                            neutral_moles[j, self.water] = (
                                1.0 / self.neutral_mw[self.water]
                            )
                            ion_moles[j] = 0.0
                            if diagnostics:
                                terms["evaluations"][start + j] = 0
                                terms["retries"][start + j] = 0
                                terms["equilibrium_residuals"][start + j] = (
                                    np.nan
                                )
                                terms["precipitated_caso4_mol_per_kg_input"][
                                    start + j
                                ] = np.nan
                            continue
                        index = tuple(
                            map(int, np.unravel_index(start + j, shape))
                        )
                        raise ConvergenceError(
                            str(error), batch_index=index
                        ) from error
                    status[start + j] = int(calls > 0)
                    if diagnostics:
                        terms["equilibrium_residuals"][start + j] = residuals
                        terms["precipitated_caso4_mol_per_kg_input"][
                            start + j
                        ] = solid
                        terms["evaluations"][start + j] = calls
                        terms["retries"][start + j] = retries
                solvent = neutral_moles * self.neutral_mw
                solvent_mass = solvent.sum(axis=1, keepdims=True)
            x = np.column_stack((neutral_moles, ion_moles))
            species_moles[rows] = x
            x /= x.sum(axis=1, keepdims=True)
            mol[rows] = ion_moles / solvent_mass
            if self.carbonate and self.carbonate_model == "gibbs_consistent":
                from .carbonate import activity_terms

                sr, mr, lr, strength, scale, mol[rows], _ = activity_terms(
                    self, neutral_moles, ion_moles, tk
                )
            else:
                sr = np.empty_like(x)
                for temperature in np.unique(tk):
                    selected = tk == temperature
                    sr[selected] = short_range(self, x[selected], temperature)
                mr, lr, strength, scale = electrolyte_terms(
                    self,
                    neutral_moles / neutral_moles.sum(axis=1, keepdims=True),
                    mol[rows],
                    tk,
                )
                if self.carbonate:
                    sr[:, self.co2] = lr[:, self.co2] = 0.0
                    mr[:, self.co2] = 2 * (mol[rows] @ self.co2_lambda)
            coeff = sr + mr + lr
            bad_coeff = ~np.isfinite(coeff).all(axis=1) | ~np.isfinite(scale)
            if collect_failures:
                status[
                    start + np.flatnonzero(bad_coeff & (status[rows] >= 0))
                ] = -1
            else:
                reject(bad_coeff, "unrepresentable activity calculation")
            xout[rows] = x[:, :n]
            nm[rows], im[rows] = solvent > tiny, mol[rows] > tiny
            lng[rows] = np.where(nm[rows], coeff[:, :n], np.nan)
            lngi[rows] = np.where(
                im[rows], coeff[:, n:] - scale[:, None], np.nan
            )
            activity[rows] = np.where(
                nm[rows], x[:, :n] * _exp(coeff[:, :n], 0.4), 0
            )
            for k, (c, a) in enumerate(self.pairs):
                idx = [self.ions.index(c), self.ions.index(a)]
                defined = im[rows][:, idx].all(axis=1)
                sm[rows, k] = defined
                nu = self.pair_counts[k]
                weighted = lngi[rows][:, idx] @ nu
                mean[rows, k] = np.where(
                    defined, _exp(weighted / nu.sum(), 1), np.nan
                )
                with np.errstate(divide="ignore"):
                    product = weighted + np.log(mol[rows][:, idx]) @ nu
                iap[rows, k] = np.where(defined, _exp(product, 1), 0)
            low, high, code = (288, 309, 10) if i else (280, 400, 11)
            warnings[rows] = np.where((tk < low) | (tk > high), code, 16)
            if terms is not None:
                for key, val in (
                    ("sr", sr),
                    ("mr", mr),
                    ("lr", lr),
                    ("ionic_strength", strength),
                    ("molality_conversion", scale),
                ):
                    terms[key][rows] = val

        def restore(arr):
            return arr.reshape(shape + arr.shape[1:])

        if terms is not None:
            terms = {key: restore(val) for key, val in terms.items()}
        arrays = (
            xout,
            activity,
            lng,
            mol,
            lngi,
            mean,
            iap,
            nm,
            im,
            sm,
            warnings,
        )
        return ActivityResult(
            self.neutrals,
            self.ions,
            self.pairs,
            *map(restore, arrays),
            terms,
            restore(species_moles),
            restore(status),
            (UPSTREAM_VERSION, UPSTREAM_REVISION)
            + (
                ("gibbs_consistent",)
                if self.carbonate_model == "gibbs_consistent"
                else ()
            ),
        )


def _exp(value, fraction):
    # ModCalcActCoeff defines lnhuge as 0.49 times the machine log limit.
    limit = fraction * 0.49 * np.log(np.finfo(float).max)
    magnitude = np.abs(value)
    exceedance = np.maximum(magnitude - limit, 0)
    bounded = np.minimum(magnitude, limit) + np.minimum(
        0.01 * np.log1p(exceedance), 0.04 * limit
    )
    return np.exp(np.copysign(bounded, value))


class AIOMFACModel:
    """Prepare a v3.15 analytical mixture for CPU or CUDA activity evaluation."""

    capabilities = frozenset({"direct_cpu", "direct_cuda", "closed_speciation"})

    @staticmethod
    def prepare(
        mixture: MixtureSpec,
        *,
        execution="cpu",
        device=None,
        carbonate_model="pinned",
    ) -> "PreparedModel | CudaModel":
        """Bind analytical and internal species to one requested device."""
        if execution not in ("cpu", "cuda"):
            raise ValueError(f"unsupported AIOMFAC execution {execution!r}")
        if carbonate_model not in ("pinned", "gibbs_consistent"):
            raise ValueError(
                "carbonate_model must be pinned or gibbs_consistent"
            )
        comp = dict(mixture.components)
        original = len(comp)
        ids = set().union(*comp.values())
        sulfate = 248 in ids or 261 in ids and bool(ids & {205, 250})
        carbonate = 250 in ids or 262 in ids and bool(ids & {205, 248})

        if carbonate_model == "gibbs_consistent" and ids & {173, 250, 262}:
            carbonate = True
            sulfate = bool(ids & {248, 261})

        def include(name, groups):
            if groups not in comp.values():
                if name in comp:
                    raise ValueError(f"implicit species name collision: {name}")
                comp[name] = groups

        if sulfate:
            include("implicit H2SO4", {205: 2, 261: 1})
            include("implicit HHSO4", {205: 1, 248: 1})
        if carbonate:
            include("CO2(aq)", {173: 1})
            include("implicit H2CO3", {205: 2, 262: 1})
            include("implicit HHCO3", {205: 1, 250: 1})
            include("implicit HOH", {205: 1, 247: 1})
        prepared = replace(
            _prepare_fixed(comp), carbonate_model=carbonate_model
        )
        if sulfate or carbonate:
            waters = [
                j for j, k in enumerate(prepared.neutrals) if comp[k] == {16: 1}
            ]
            if len(waters) != 1:
                raise ValueError(
                    "aqueous equilibrium requires one water component"
                )
            co2_indices = [
                j
                for j, k in enumerate(prepared.neutrals)
                if comp[k] == {173: 1}
            ]
            if len(co2_indices) > 1:
                raise ValueError(
                    "aqueous equilibrium accepts one dissolved CO2 component"
                )
            co2 = co2_indices[0] if co2_indices else -1
            if carbonate and carbonate_model == "gibbs_consistent":
                # Preserve stoichiometric kg as well as C, H and O. The pinned
                # CO₂ subgroup mass is rounded independently of the ion table.
                tables = _tables()
                co3 = tables["SMWA"][262 - 241] * 0.001
                proton = tables["SMWC"][205 - 201] * 0.001
                mw = prepared.neutral_mw.copy()
                mw[co2] = co3 + 2 * proton - mw[waters[0]]
                mw.setflags(write=False)
                prepared = replace(prepared, neutral_mw=mw)
            lambdas = _tables()["lambdaIN"][np.asarray(prepared.ions) - 201]
            lambdas.setflags(write=False)
            prepared = replace(
                prepared,
                sulfate=sulfate,
                carbonate=carbonate,
                water=waters[0],
                co2=co2,
                co2_lambda=lambdas,
                ncomponents=original,
                virtual_count=len(comp) - original,
            )
        if execution == "cuda":
            try:
                from ._cuda import CudaModel
            except ModuleNotFoundError as error:
                if error.name != "warp":
                    raise
                raise ImportError(
                    'CUDA requires pip install "particula-aiomfac[cuda]"'
                ) from error
            return CudaModel(prepared, device or "cuda:0")
        if device not in (None, "cpu"):
            raise ValueError("device must match CPU execution")
        return prepared


def _prepare_fixed(comp):
    """Arrange terms for a known speciation state (also used in solver qualification)."""
    tables = _tables()
    all_ids = set().union(*comp.values())
    ions = tuple(sorted(k for k in all_ids if k > 200))
    subs = np.array(sorted(all_ids - set(ions)), dtype=int)
    neutrals = tuple(k for k, groups in comp.items() if max(groups) <= 200)
    salts = tuple(k for k in comp if k not in neutrals)
    if not neutrals:
        raise ValueError("at least one solvent component is required")
    if any(k > len(tables["SR_RR"]) for k in ions):
        raise ValueError(f"unsupported ion IDs {ions}")
    rid = np.array(ions, dtype=int)
    cats, ans = rid[rid <= 240], rid[rid > 240]
    ci, ai = cats - 201, ans - 241
    if np.any(ci >= tables["bTABnc"].shape[1]):
        raise ValueError(f"missing MR cation interactions for {cats}")
    z = tables["Ioncharge"][rid - 201]
    stoich = np.zeros((len(salts), len(ions)))
    for k, name in enumerate(salts):
        groups = comp[name]
        if len(groups) != 2 or not set(groups) <= set(ions):
            raise ValueError(f"{name}: expected one cation and one anion")
        stoich[k] = [groups.get(j, 0) for j in ions]
        if stoich[k] @ z != 0 or gcd(*groups.values()) != 1:
            raise ValueError(f"{name}: invalid salt stoichiometry {groups}")
    pairs = tuple((int(c), int(a)) for c in cats for a in ans)
    pair_counts = []
    for c, a in pairs:
        zc, za = int(z[ions.index(c)]), int(-z[ions.index(a)])
        common = gcd(zc, za)
        pair_counts.append([za // common, zc // common])
    pair_counts = np.array(pair_counts, dtype=int).reshape(-1, 2)
    ids = np.r_[subs, rid]
    mg = tables["NKTAB"][subs - 1]
    gmw = tables["GroupMW"][subs - 1] * 0.001
    imw = np.r_[tables["SMWC"][ci], tables["SMWA"][ai]] * 0.001
    if np.any(mg <= 0) or np.any(gmw <= 0) or np.any(imw <= 0):
        raise ValueError(f"unavailable subgroup properties for {ids.tolist()}")
    ng = np.array([[comp[name].get(j, 0) for j in subs] for name in neutrals])
    main = np.unique(mg)
    group_map = mg[:, None] == main
    mcounts = ng @ group_map.astype(int)
    for row in ng:
        if row[np.isin(mg, [52, 68])].sum() > row[mg == 69].sum():
            raise ValueError("subgroup pairing error 13: missing OH group")
    counts = np.zeros((len(neutrals) + len(ions), len(ids)))
    counts[: len(neutrals), : len(subs)] = ng
    counts[len(neutrals) :, len(subs) :] = np.eye(len(ions))
    r, q = (tables[key][ids - 1] for key in ("SR_RR", "SR_QQ"))
    r[ids == 154], q[ids == 154] = 1.381433, 3.0
    if np.any(r <= 0) or np.any(q < 0) or np.any(counts @ q <= 0):
        raise ValueError(f"missing SR subgroup parameters for {ids}")
    interaction = np.zeros((len(ids), len(ids)))
    subarr = tables["ARR"][np.ix_(mg - 1, mg - 1)]
    if np.any(subarr < -888888):
        left, right = np.argwhere(subarr < -888888)[0]
        raise ValueError(f"missing SR interaction {mg[left]}/{mg[right]}")
    interaction[: len(subs), : len(subs)] = subarr
    bc = []
    for prefix in ("b", "c"):
        coeff = np.column_stack(
            [
                tables[prefix + "TABnc"][np.ix_(main - 1, ci)],
                tables[prefix + "TABna"][np.ix_(main - 1, ai)],
            ]
        )
        if 154 in subs:
            for ion, val in (
                (204, (0.30576783, 0.40411133)),
                (261, (-0.21822703, -0.12595744)),
            ):
                if ion in ions:
                    coeff[np.isin(main, [52, 68]), ions.index(ion)] = val[
                        prefix == "c"
                    ]
        if np.any(coeff < -1000):
            g, j = np.argwhere(coeff < -1000)[0]
            raise ValueError(
                f"missing group/ion interaction {main[g]}/{ions[j]}"
            )
        bc.append(coeff)

    def pair_tables(names):
        return tuple(tables[key][np.ix_(ci, ai)] for key in names.split())

    bpair = pair_tables("bTABAC cTABAC omegaTAB")
    for ic, ia in np.argwhere(abs(bpair[0]) < 1e-10):
        if (cats[ic], ans[ia]) != (205, 247):
            raise ValueError(
                f"missing cation/anion interaction {cats[ic]}/{ans[ia]}"
            )
    manifest = json.loads(
        files(__package__).joinpath("data/manifest.json").read_text()
    )
    estimated = tuple(
        pair
        for pair in pairs
        if f"{pair[0]}/{pair[1]}" in manifest["estimated_pairs"]
    )
    model = PreparedModel(
        neutrals=neutrals,
        ions=ions,
        pairs=pairs,
        estimated_pairs=estimated,
        ncomponents=len(comp),
        ncation=len(cats),
        neutral_indices=np.array([tuple(comp).index(k) for k in neutrals], int),
        salt_indices=np.array([tuple(comp).index(k) for k in salts], int),
        neutral_mw=ng @ gmw,
        salt_mw=stoich @ imw,
        stoich=stoich,
        pair_counts=pair_counts,
        counts=counts,
        q=q,
        area=counts @ q,
        volume=counts @ r,
        interaction=interaction,
        neutral_counts=ng,
        group_mw=gmw,
        group_map=group_map,
        main_counts=mcounts,
        charge=z,
        bgroup=bc[0],
        cgroup=bc[1],
        bpair=bpair,
        cpair=pair_tables("Cn1TABAC Cn2TABAC omega2TAB"),
        like=tables["RccTAB"][np.ix_(ci, ci)],
        three=tables["qcca1TAB"][np.ix_(ci, ci, ai)],
    )
    for value in vars(model).values():
        for arr in value if isinstance(value, tuple) else (value,):
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
    return model
