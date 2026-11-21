# %%
"""Test coupled size-dependent deposition across all 15 paired chamber runs."""

import json

import numpy as np
import pandas as pd
import warp as wp
from campaign import HERE, PPB_TO_UG_M3, inputs
from engine import CONDITIONS, F64, KB, brownian_equal_pair
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gpu.dynamics.wall_loss_funcs import (
    rectangle_wall_loss_coefficient_from_transport_wp,
)
from particula.util.constants import MOLECULAR_WEIGHT_AIR
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from wall_fit import OUT

V3 = wp.vec3d
ISOPRENE_UG_M3_PER_PPB = wp.constant(F64(PPB_TO_UG_M3[2]))


@wp.func
def vapor_step(old: F64, source: F64, gain: F64, loss: F64, dt: F64):
    """Integrate uniform production and two competing first-order sinks."""
    rate = gain + loss
    x = rate * dt
    # Taylor form preserves the source term at small rate times step.
    release = F64(1.0) - wp.exp(-x)
    source_release = F64(0.5) * x - x * x / F64(6.0)
    if x > F64(0.0001):
        source_release = F64(1.0) - release / x
    removed = old * release + source * source_release
    return V3(
        old + source - removed, removed * gain / rate, removed * loss / rate
    )


@wp.kernel
def campaign_kernel(  # noqa: C901 - fused CUDA integration and observation loop
    params: wp.array2d(dtype=F64),
    src: wp.array3d(dtype=F64),
    observations: wp.array3d(dtype=F64),
    initial: wp.array2d(dtype=F64),
    doses: wp.array(dtype=F64),
    eddy: int,
    brownian: int,
    sub: int,
    pathway: int,
    errors: wp.array4d(dtype=F64),
):
    """Integrate product maturation, particle deposition and number loss."""
    i, r, d = wp.tid()
    dt = F64(60.0) / F64(sub)
    temps = F64(298.15)
    seed = initial[r, 0]
    count = initial[r, 1]
    soa = F64(0.0)
    inter = V3()
    mature = V3()
    supply_mature = V3()
    for s in range(3):
        arg = dt / (F64(3600.0) * params[i, s])
        mature[s] = F64(1.0) - wp.exp(-arg)
        supply_mature[s] = F64(1.0) - mature[s] / arg
    observed_time_peak = F64(0.0)
    auto_gas = F64(0.0)
    uptake_gas = F64(0.0)
    other_sink = F64(0.0)
    auto_delivered = F64(0.0)
    uptake_delivered = F64(0.0)
    ms = F64(0.0)
    ns = F64(0.0)
    deposited = F64(0.0)
    supplied = F64(0.0)
    for j in range(300):
        old = soa
        nold = count / initial[r, 1]
        supply = V3()
        for s in range(3):
            supply[s] = (
                (
                    params[i, s + 3] * src[r, j, s]
                    + params[i, s + 6] * src[r, j, s + 3]
                )
                * doses[d]
                / F64(sub)
            )
        alpha = (src[r, j, 0] + src[r, j, 3]) * doses[d] / F64(sub)
        cresol = (src[r, j, 1] + src[r, j, 4]) * doses[d] / F64(sub)
        iso_ppb_min = (
            (src[r, j, 2] + src[r, j, 5]) * doses[d] / (ISOPRENE_UG_M3_PER_PPB)
        )
        diverted = F64(0.0)
        if pathway == 5:
            # Split the existing mixed-parent source between its two partners.
            reacted = V3()
            for s in range(3):
                reacted[s] = src[r, j, s] + src[r, j, s + 3]
            for s in range(3):
                first = (s + 1) % 3
                second = (s + 2) % 3
                # Alternate partners are isoprene, isoprene and cresol.
                alt = 2
                if s == 2:
                    alt = 1
                other = reacted[first] + reacted[second]
                if other > F64(0.0):
                    mixed = src[r, j, s + 3] * doses[d] / F64(sub)
                    supply[s] += (
                        mixed
                        * reacted[alt]
                        / other
                        * (params[i, 16 + s] - params[i, 6 + s])
                    )
        if pathway >= 6:
            retain = F64(1.0) / (F64(1.0) + params[i, 12] * iso_ppb_min)
            if pathway != 9:
                diverted += supply[0] * (F64(1.0) - retain)
                supply[0] *= retain
            if pathway != 8:
                diverted += supply[1] * (F64(1.0) - retain)
                supply[1] *= retain
        for _step in range(sub):
            volume = seed / F64(1770.0) + soa / F64(1400.0)
            radius = wp.pow(
                F64(3e-9) * volume / (F64(4.0 * 3.141592653589793) * count),
                F64(1.0 / 3.0),
            )
            kw = params[i, 9] / F64(3600.0)
            if eddy == 1:
                kw = rectangle_wall_loss_coefficient_from_transport_wp(
                    params[i, 9],
                    radius,
                    (seed + soa) / volume,
                    temps,
                    F64(3.0),
                    F64(2.0),
                    F64(3.0),
                    initial[r, 5],
                    initial[r, 6],
                    KB,
                )
            keep = wp.exp(-kw * dt)
            supplied += diverted
            other_sink += diverted
            product = F64(0.0)
            for s in range(3):
                move = inter[s] * mature[s] + supply[s] * supply_mature[s]
                inter[s] += supply[s] - move
                supplied += supply[s]
                product += move
            # Candidate product yields are per mass of reacted parent.
            if pathway == 1:
                prompt = params[i, 10] * alpha
                supplied += prompt
                product += prompt
                auto_delivered += prompt
            if pathway == 2 or pathway == 4:
                source = params[i, 10] * alpha
                k_auto = F64(1.0) / params[i, 11]
                step = vapor_step(
                    auto_gas,
                    source,
                    k_auto,
                    k_auto * params[i, 12] * iso_ppb_min,
                    dt,
                )
                auto_gas = step[0]
                product += step[1]
                auto_delivered += step[1]
                other_sink += step[2]
                supplied += source
            if pathway == 3 or pathway == 4 or pathway == 7:
                source = params[i, 13] * (alpha + cresol)
                area = F64(4.0 * 3.141592653589793) * radius * radius * count
                # cbar = 200 m/s, geometric area in m2/m3.
                k_uptake = params[i, 14] * F64(50.0) * area
                step = vapor_step(
                    uptake_gas,
                    source,
                    k_uptake,
                    params[i, 15] / F64(3600.0),
                    dt,
                )
                uptake_gas = step[0]
                product += step[1]
                uptake_delivered += step[1]
                other_sink += step[2]
                supplied += source
            if brownian == 1:
                pair = brownian_equal_pair(
                    radius,
                    (seed + soa) * F64(1e-9) / count,
                    temps,
                    initial[r, 5],
                    initial[r, 6],
                )
                count = count / (
                    F64(1.0) + F64(0.5) * pair * count * (F64(1.0) - keep) / kw
                )
            count *= keep
            seed *= keep
            survived = wp.sqrt(keep)
            deposited += soa * (F64(1.0) - keep) + product * (
                F64(1.0) - survived
            )
            soa = soa * keep + product * survived
        if errors.shape[3] > 8:
            errors[i, r, d, 8 + 2 * j] = soa
            errors[i, r, d, 9 + 2 * j] = count / initial[r, 1]
        if observations[r, j, 0] >= F64(0.0):
            pred = old + (soa - old) * observations[r, j, 0]
            observed_time_peak = wp.max(observed_time_peak, pred)
            ms += wp.pow(
                (pred - observations[r, j, 1]) / initial[r, 2], F64(2.0)
            )
        if observations[r, j, 2] >= F64(0.0):
            pred = nold + (count / initial[r, 1] - nold) * observations[r, j, 2]
            ns += wp.pow(pred - observations[r, j, 3], F64(2.0))
    errors[i, r, d, 0] = wp.sqrt(ms / initial[r, 3])
    errors[i, r, d, 1] = wp.sqrt(ns / initial[r, 4])
    errors[i, r, d, 2] = wp.abs(
        supplied
        - deposited
        - soa
        - inter[0]
        - inter[1]
        - inter[2]
        - auto_gas
        - uptake_gas
        - other_sink
    )

    if errors.shape[3] >= 5:
        errors[i, r, d, 3] = observed_time_peak
        errors[i, r, d, 4] = count / initial[r, 1]
    if errors.shape[3] > 5:
        errors[i, r, d, 5] = auto_delivered
        errors[i, r, d, 6] = uptake_delivered
        errors[i, r, d, 7] = other_sink
        errors[i, r, d, 608] = soa + deposited


class Campaign:
    """Batch kinetic and injection candidates on a single CUDA device."""

    def __init__(self, source_knot_seconds=1200):
        """Align the measured gas, aerosol and particle counter records."""
        self.runs = inputs(source_knot_seconds)
        conditions = pd.read_csv(HERE / "data/campaign_peaks.csv").set_index(
            "run"
        )
        primary = pd.read_csv(HERE.parent / "data/context.csv")
        extra = pd.read_csv(HERE / "data/wall_counts.csv")
        src = []
        obs = []
        props = []
        self.instruments = []
        for r, _, t, y, m in self.runs:
            if r in (1, 4, 5, 6):
                name = "DMPS" if r == 5 else "WCPC"
                counter = primary[
                    (primary.run == r)
                    & (primary.instrument == name)
                    & (
                        primary.channel.isin(
                            ["Total_number", "total_NumberConc"]
                        )
                    )
                ]
                nt = counter.time_s.to_numpy()
                nv = counter.value.to_numpy()
                n0 = CONDITIONS[r][2]
            else:
                available = extra[extra.run == r]
                name = (
                    "WCPC" if "WCPC" in available.instrument.values else "DMPS"
                )
                sampled = available[available.instrument == name]
                nt = sampled.time_s.to_numpy()
                nv = sampled.count_cm3.to_numpy()
                n0 = nv[np.abs(nt).argmin()]
            self.instruments.append(name)
            keep = (nt >= 0) & (nt <= 18000)
            nt = nt[keep]
            nv = nv[keep] / n0
            entry = np.full((300, 4), -1.0)
            for j, (times, values) in enumerate(((t, y), (nt, nv))):
                ix = np.minimum((times / 60).astype(int), 299)
                if len(np.unique(ix)) != len(ix):
                    raise ValueError((r, "overlapping observations"))
                entry[ix, 2 * j] = times / 60 - ix
                entry[ix, 2 * j + 1] = values
            props.append(
                [
                    conditions.loc[r, "seed_ug_m3"],
                    n0 * 1e6,
                    max(1.0, y.max()),
                    len(y),
                    len(nt),
                    get_dynamic_viscosity(298.15),
                    get_molecule_mean_free_path(
                        MOLECULAR_WEIGHT_AIR, 298.15, 101325.0
                    ),
                ]
            )
            src.append(m)
            obs.append(entry)
        self.initial = np.array(props)
        self.data = [
            wp.array(np.array(a), dtype=F64, device="cuda:0")
            for a in (src, obs, props)
        ]

    def evaluate(
        self,
        p,
        eddy,
        objective,
        doses,
        sub=1,
        *,
        outcomes=False,
        pathway=0,
        traces=False,
    ):
        """Select one bounded dose per run and return score, errors and doses.

        With outcomes, append observed-time peak SOA and final number fraction.
        With traces, append ledger error, peak, final number fraction, gross
        aerosol mass from the alpha and uptake branches, alternative gas loss,
        300 consecutive (SOA, N/N0) minute pairs and gross SOA production.
        Mass diagnostics use micrograms per cubic meter.
        """
        p = np.atleast_2d(p)
        if pathway not in range(10) or (
            pathway and p.shape[1] != (19 if pathway == 5 else 16)
        ):
            raise ValueError(
                "Expected an extended parameter vector for each pathway"
            )
        args = [
            wp.array(p, dtype=F64, device="cuda:0"),
            *self.data,
            wp.array(doses, dtype=F64, device="cuda:0"),
            int(eddy),
            1,
            sub,
            pathway,
        ]
        out = wp.empty(
            (
                len(p),
                len(self.runs),
                len(doses),
                609 if traces else (5 if outcomes else 3),
            ),
            dtype=F64,
            device="cuda:0",
        )
        wp.launch(
            campaign_kernel,
            dim=out.shape[:3],
            inputs=[*args, out],
            device="cuda:0",
        )
        e = out.numpy()
        if not np.isfinite(e).all() or e[:, :, :, 2].max() > 1e-7:
            raise ArithmeticError("Campaign product ledger failed")
        merit = e[:, :, :, 0] if objective == "mass" else e[:, :, :, :2].max(3)
        ix = merit.argmin(2)
        ii = np.arange(len(p))[:, None]
        rr = np.arange(len(self.runs))[None, :]
        result = merit[ii, rr, ix].max(1), e[ii, rr, ix, :2], doses[ix]
        if traces:
            return *result, e[ii, rr, ix, 2:]
        if outcomes:
            return *result, e[ii, rr, ix, 3:]
        return result


def search(campaign, eddy, objective, start, fixed_eddy=None):
    """Search the shared kinetic parameters and refine the best solutions."""
    bounds = np.array(
        [(np.log10(1 / 12), np.log10(4))] * 3
        + [(0, 2)] * 6
        + [(-3, 3) if eddy else tuple(np.log10([0.05, 3.0]))]
    )
    if fixed_eddy is not None:
        bounds[-1, :] = np.log10(fixed_eddy)
        start = start.copy()
        start[-1] = fixed_eddy
    lo, hi = bounds.T

    def physical(x):
        p = np.atleast_2d(x).copy()
        p[:, [0, 1, 2, 9]] = 10 ** p[:, [0, 1, 2, 9]]
        return p

    def loss(x, doses):
        return campaign.evaluate(physical(x), eddy, objective, doses)[0]

    start = start.copy()
    start[[0, 1, 2, 9]] = np.log10(start[[0, 1, 2, 9]])
    coarse = np.linspace(0.85, 1.15, 3)
    fine = np.linspace(0.85, 1.15, 11)
    winners = []
    for seed in (2249, 8623):
        rng = np.random.default_rng(seed)
        sample = lo + qmc.Sobol(10, scramble=True, seed=seed).random_base2(
            9
        ) * (hi - lo)
        # Two pure-isoprene runs report zero mass. Include candidates that
        # satisfy this observation exactly as well as the full yield space.
        sample[::2, 5] = 0.0
        sample[::4, 8] = 0.0
        near = np.clip(
            start + rng.normal(size=(96, 10)) * (hi - lo) * 0.12, lo, hi
        )
        near[:64, 5] = 0.0
        near[:32, 8] = 0.0
        sample = np.vstack([sample, near])
        sample[0] = start
        ranking = np.argsort(loss(sample, coarse))
        pop = np.vstack(
            [
                sample[ranking[:16]],
                sample[rng.choice(len(sample), 48, replace=False)],
            ]
        )
        ans = differential_evolution(
            lambda x: loss(x.T, coarse),
            bounds,
            init=pop,
            rng=rng,
            maxiter=70,
            updating="deferred",
            vectorized=True,
            polish=False,
            tol=1e-5,
        )
        pool = np.vstack([ans.population, sample[ranking[:24]], start])
        values = loss(pool, fine)
        best = pool[values.argmin()]
        value = values.min()
        for scale in np.geomspace(0.03, 0.0001, 24):
            trial = np.clip(
                best + rng.normal(size=(64, 10)) * scale * (hi - lo), lo, hi
            )
            trial[::2, 5] = 0.0
            trial[::4, 8] = 0.0
            vals = loss(trial, fine)
            if vals.min() < value:
                best = trial[vals.argmin()]
                value = vals.min()
        p = physical(best)[0]
        winners.append(dict(seed=seed, score=value, physical=p.tolist()))
        print(
            eddy, objective, seed, round(value, 5), round(p[-1], 4), flush=True
        )
    return winners


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    c = Campaign()
    base = pd.read_csv(HERE / "results/campaign_dose_scores.csv")
    row = base[(base.family == "parent_lags") & (base.stage == "joint")].iloc[0]
    start = np.r_[json.loads(row.lag_h), json.loads(row.yields), row.removal_h]
    records = []
    scores = []
    for eddy in (False, True):
        anchored = start.copy()
        if eddy:
            anchored[-1] = 5.0
        for obj in ("mass", "joint"):
            fits = search(c, eddy, obj, anchored)
            model = "eddy_brownian" if eddy else "constant_brownian"
            records.extend(dict(model=model, objective=obj, **f) for f in fits)
            anchored = np.array(min(fits, key=lambda x: x["score"])["physical"])
            doses = np.linspace(0.85, 1.15, 61)
            _, errors, d = c.evaluate(anchored, eddy, obj, doses, sub=2)
            _, check, _ = c.evaluate(anchored, eddy, obj, doses, sub=4)
            for i, (r, *_) in enumerate(c.runs):
                scores.append(
                    dict(
                        model=model,
                        objective=obj,
                        run=r,
                        wall_parameter=anchored[-1],
                        instrument=c.instruments[i],
                        initial_number_cm3=c.initial[i, 1] * 1e-6,
                        count_samples=int(c.initial[i, 4]),
                        mass_samples=int(c.initial[i, 3]),
                        dose=d[0, i],
                        mass_nrmse=errors[0, i, 0],
                        count_nrmse=errors[0, i, 1],
                        refine_max_score_change=abs(
                            check[0, i] - errors[0, i]
                        ).max(),
                    )
                )
            pd.DataFrame(scores).to_csv(
                OUT / "campaign_scores.csv", index=False
            )
            (OUT / "campaign_restarts.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
