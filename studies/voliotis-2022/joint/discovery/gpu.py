"""CUDA simulation of cumulative parent-tagged chamber reaction networks."""

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reactions import CANDIDATES
from wall_campaign import (
    F64,
    ISOPRENE_UG_M3_PER_PPB,
    KB,
    V3,
    Campaign,
    brownian_equal_pair,
    rectangle_wall_loss_coefficient_from_transport_wp,
)

State = wp.types.vector(length=31, dtype=F64)
PI = wp.constant(F64(np.pi))


@wp.func
def labelled_solid(pool: int):
    """Identify particle inventories counted within the stable bulk mass."""
    return pool >= 11 and pool < 26


@wp.func
def total_soa(x: State):
    """Suspended labile material and chemically stabilized material."""
    return x[6] + x[7] + x[8] + x[9]


@wp.func
def advance_channel(
    x: State, a: int, b: int, dest: int, rate: F64, dt_h: F64, mw: V3
):
    """Transfer product equivalents using positive reaction extents."""
    removed = x[a] * (F64(1.0) - wp.exp(-rate * dt_h))
    partner = F64(0.0)
    if b >= 0:
        ma = mw[a % 3]
        mb = mw[b % 3]
        na = x[a] / ma
        nb = x[b] / mb
        # A+A consumes two equivalents per event. For unlike reactants this
        # bounded first-order-in-time extent tends to mass action as dt -> 0.
        if a == b:
            z = F64(2.0) * rate * dt_h * na
            removed = x[a] * z / (F64(1.0) + z)
        else:
            extent = (
                rate
                * dt_h
                * na
                * nb
                / (F64(1.0) + rate * dt_h * wp.max(na, nb))
            )
            removed = extent * ma
            partner = extent * mb
            x[b] -= partner
    x[a] -= removed
    x[dest] += removed + partner
    return x, removed + partner


@wp.kernel
def network(  # noqa: C901 - fused split integration and tracer bookkeeping
    params: wp.array2d(dtype=F64),
    rates: wp.array2d(dtype=F64),
    source: wp.array3d(dtype=F64),
    obs: wp.array3d(dtype=F64),
    initial: wp.array2d(dtype=F64),
    dose: wp.array2d(dtype=F64),
    channels: wp.array2d(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    n_active: int,
    extended: int,
    split_residue: F64,
    sub: int,
    vapor_share: F64,
    mw: V3,
    vapor_loss_h: F64,
    reverse: int,
    diagnostic: int,
    result: wp.array4d(dtype=F64),
    flux: wp.array4d(dtype=F64),
):
    """Integrate chemistry, deposition and counts for one chamber run."""
    i, r, d = wp.tid()
    dt = F64(60.0) / F64(sub)
    dh = dt / F64(3600.0)
    seed = initial[r, 0]
    count = initial[r, 1]
    x = State()
    mature = V3()
    fresh = V3()
    for s in range(3):
        z = dh / params[i, s]
        mature[s] = F64(1.0) - wp.exp(-z)
        fresh[s] = F64(1.0) - mature[s] / z
    emitted = F64(0.0)
    deposited = F64(0.0)
    ms = F64(0.0)
    ns = F64(0.0)
    peak = F64(0.0)
    late = F64(0.0)
    small = F64(0.0)
    vkeep = wp.exp(-dh * vapor_loss_h)
    for j in range(300):
        old = total_soa(x)
        nold = count / initial[r, 1]
        supply = V3()
        extra = V3()
        iso = (
            (source[r, j, 2] + source[r, j, 5])
            * dose[r, d]
            / ISOPRENE_UG_M3_PER_PPB
        )
        for s in range(3):
            reacted = (
                (source[r, j, s] + source[r, j, s + 3]) * dose[r, d] / F64(sub)
            )
            gross = (
                (
                    params[i, s + 3] * source[r, j, s]
                    + params[i, s + 6] * source[r, j, s + 3]
                )
                * dose[r, d]
                / F64(sub)
            )
            retain = F64(1.0)
            if s < 2:
                retain = F64(1.0) / (F64(1.0) + params[i, 12] * iso)
            supply[s] = gross * retain
            extra[s] = gross * (F64(1.0) - retain) + vapor_share * wp.max(
                F64(0.0), reacted - gross
            )
        for _step in range(sub):
            soa = total_soa(x)
            volume = seed / F64(1770.0) + soa / F64(1400.0)
            radius = wp.pow(
                F64(3e-9) * volume / (F64(4.0 * 3.141592653589793) * count),
                F64(1.0 / 3.0),
            )
            kw = rectangle_wall_loss_coefficient_from_transport_wp(
                params[i, 9],
                radius,
                (seed + soa) / volume,
                F64(298.15),
                F64(3.0),
                F64(2.0),
                F64(3.0),
                initial[r, 5],
                initial[r, 6],
                KB,
            )
            pair = brownian_equal_pair(
                radius,
                (seed + soa) * F64(1e-9) / count,
                F64(298.15),
                initial[r, 5],
                initial[r, 6],
            )
            keep = wp.exp(-kw * dt)
            survived = wp.sqrt(keep)
            deposited += soa * (F64(1.0) - keep)
            area_ratio = F64(4.0) * PI * radius * radius * count / F64(0.001)
            for s in range(3):
                move = x[s] * mature[s] + supply[s] * fresh[s]
                x[s] += supply[s] - move
                x[s + 6] = x[s + 6] * keep + move * survived
                deposited += move * (F64(1.0) - survived)
                before = x[s + 3] + extra[s]
                x[s + 3] = x[s + 3] * vkeep + extra[s] * wp.sqrt(vkeep)
                x[10] += before - x[s + 3]
                emitted += supply[s] + extra[s]
            x[9] *= keep
            if extended:
                # x[9] includes the labelled solid inventories.
                for q in range(11, 26):
                    x[q] *= keep
                for q in range(26, 31):
                    lost = x[q] * (F64(1.0) - vkeep)
                    x[q] -= lost
                    x[10] += lost
            if diagnostic == 2:
                for tag_slot in range(n_active):
                    htag = active[tag_slot]
                    dest = channels[htag, 2]
                    if dest >= 6 and dest <= 9:
                        flux[i, r, d, htag + rates.shape[1]] *= keep
                    elif dest >= 3 and dest <= 5:
                        flux[i, r, d, htag + rates.shape[1]] *= vkeep
            for n in range(n_active):
                h = active[n]
                if reverse == 1:
                    h = active[n_active - 1 - n]
                k = rates[i, h]
                if k > F64(0.0):
                    if channels[h, 3] == 1:
                        k *= area_ratio
                    before_a = x[channels[h, 0]]
                    before_b = F64(0.0)
                    if channels[h, 1] >= 0:
                        before_b = x[channels[h, 1]]
                    x, moved = advance_channel(
                        x,
                        channels[h, 0],
                        channels[h, 1],
                        channels[h, 2],
                        k,
                        dh,
                        mw,
                    )
                    if extended:
                        a = channels[h, 0]
                        dest = channels[h, 2]
                        split = channels[h, 4]
                        if labelled_solid(a):
                            x[9] -= moved
                        if labelled_solid(dest):
                            x[9] += moved
                        if split >= 0:
                            secondary = split_residue * moved
                            x[dest] -= secondary
                            x[split] += secondary
                            if labelled_solid(dest):
                                x[9] -= secondary
                            if labelled_solid(split):
                                x[9] += secondary
                    if diagnostic == 1 or diagnostic == 2:
                        flux[i, r, d, h] += moved
                    if diagnostic == 2:
                        for tag_slot in range(n_active):
                            htag = active[tag_slot]
                            dest = channels[htag, 2]
                            if dest == channels[h, 0] and before_a > F64(0.0):
                                flux[i, r, d, htag + rates.shape[1]] *= (
                                    x[dest] / before_a
                                )
                            elif dest == channels[h, 1] and before_b > F64(0.0):
                                flux[i, r, d, htag + rates.shape[1]] *= (
                                    x[dest] / before_b
                                )
                        flux[i, r, d, h + rates.shape[1]] += moved
            count = (
                count
                / (F64(1.0) + F64(0.5) * pair * count * (F64(1.0) - keep) / kw)
                * keep
            )
            seed *= keep
        if diagnostic > 0:
            for q in range(11 + 20 * extended):
                small = wp.min(small, x[q])
            if extended:
                unlabelled = x[9]
                for q in range(11, 26):
                    unlabelled -= x[q]
                small = wp.min(small, unlabelled)
        if diagnostic == 3 and j % 5 == 4:
            col = (j // 5) * 8
            flux[i, r, d, col] = total_soa(x)
            flux[i, r, d, col + 1] = count / initial[r, 1]
            flux[i, r, d, col + 2] = x[11]
            flux[i, r, d, col + 3] = x[16]
            flux[i, r, d, col + 4] = x[21]
            flux[i, r, d, col + 5] = x[26]
            flux[i, r, d, col + 6] = x[12]
            flux[i, r, d, col + 7] = x[13]
        if diagnostic == 4 or diagnostic == 5:
            flux[i, r, d, 3 * j] = total_soa(x)
            flux[i, r, d, 3 * j + 1] = deposited
            flux[i, r, d, 3 * j + 2] = seed
            if diagnostic == 5:
                flux[i, r, d, 3 * j + 2] = count / initial[r, 1]
        if obs[r, j, 0] >= F64(0.0):
            pred = old + (total_soa(x) - old) * obs[r, j, 0]
            peak = wp.max(peak, pred)
            late = pred
            ms += wp.pow((pred - obs[r, j, 1]) / initial[r, 2], F64(2.0))
        if obs[r, j, 2] >= F64(0.0):
            pred = nold + (count / initial[r, 1] - nold) * obs[r, j, 2]
            ns += wp.pow(pred - obs[r, j, 3], F64(2.0))
    stock = F64(0.0)
    for q in range(11):
        stock += x[q]
    if extended:
        for q in range(26, 31):
            stock += x[q]
    result[i, r, d, 0] = wp.sqrt(ms / initial[r, 3])
    result[i, r, d, 1] = wp.sqrt(ns / initial[r, 4])
    result[i, r, d, 2] = wp.abs(emitted - deposited - stock)
    result[i, r, d, 3] = peak
    result[i, r, d, 4] = late
    result[i, r, d, 5] = count / initial[r, 1]
    result[i, r, d, 6] = small
    result[i, r, d, 7] = x[9]


class Model(Campaign):
    """Evaluate fixed per-run or profiled doses on the same CUDA device."""

    def __init__(
        self,
        reactions=CANDIDATES,
        masses=(184.0, 140.0, 118.0),
        vapor_loss_h=1.0,
    ):
        """Upload kinetic definitions for the requested network."""
        if not reactions or len({r.key for r in reactions}) != len(reactions):
            raise ValueError("Provide distinct labelled reaction channels")
        self.extended = int(
            any(
                max(r.dest, r.a, getattr(r, "split", -1)) > 10
                for r in reactions
            )
        )
        for r in reactions:
            split = getattr(r, "split", -1)
            if not (
                (0 <= r.a < 9 or 11 <= r.a < 31)
                and -1 <= r.b < 9
                and 0 <= r.dest < 31
                and (r.a < 9 or r.b == -1)
                and (split == -1 or (11 <= split < 31 and split != r.dest))
            ):
                raise ValueError(f"Invalid source or product pool in {r.key}")
            if r.dest in (r.a, r.b):
                raise ValueError(
                    f"Product must differ from reactants in {r.key}"
                )
        if len(masses) != 3 or min(masses) <= 0 or vapor_loss_h < 0:
            raise ValueError(
                "Invalid surrogate molar masses or vapor loss rate"
            )
        self.masses = V3(*masses)
        self.vapor_loss_h = vapor_loss_h
        super().__init__()
        self.n_reactions = len(reactions)
        self.channels = wp.array(
            [
                [r.a, r.b, r.dest, int(r.area_scaled), getattr(r, "split", -1)]
                for r in reactions
            ],
            dtype=wp.int32,
            device="cuda:0",
        )

    def run(
        self,
        p,
        rates,
        ids,
        doses,
        objective,
        sub=2,
        diagnostics=False,
        share=0.1,
        reverse=False,
        split_residue=0.5,
    ):
        """Run the supplied rates and profile the dose for each experiment."""
        if not 0 <= split_residue <= 1:
            raise ValueError("Residue mass fraction must be between 0 and 1")
        if self.extended and diagnostics == 2:
            raise ValueError(
                "Split-product networks support gross flux diagnostics"
            )
        rates = np.atleast_2d(rates)
        if rates.shape[1] != self.n_reactions or np.any(rates < 0):
            raise ValueError("Supply one nonnegative coefficient per reaction")
        if not np.isfinite(rates).all() or not set(ids) <= set(
            range(self.n_reactions)
        ):
            raise ValueError("Invalid reaction rates or channel indices")
        p = np.broadcast_to(p, (len(rates), 16)).copy()
        doses = np.asarray(doses)
        if doses.ndim == 1:
            doses = np.tile(doses, (len(self.runs), 1))
        shape = (len(rates), len(self.runs), doses.shape[1])

        def upload(x, dtype=F64):
            return wp.array(
                np.ascontiguousarray(x), dtype=dtype, device="cuda:0"
            )

        out = wp.empty((*shape, 8), dtype=F64, device="cuda:0")
        width = 480 if diagnostics == 3 else self.n_reactions * int(diagnostics)
        if diagnostics in (4, 5):
            # Organic mass, wall mass, and either seed or relative count.
            width = 300 * 3
        flux = wp.zeros(
            (*shape, width) if diagnostics else (1, 1, 1, 1),
            dtype=F64,
            device="cuda:0",
        )
        wp.launch(
            network,
            dim=shape,
            inputs=[
                upload(p),
                upload(rates),
                *self.data,
                upload(doses),
                self.channels,
                upload(ids or [0], wp.int32),
                len(ids),
                self.extended,
                split_residue,
                sub,
                share,
                self.masses,
                self.vapor_loss_h,
                int(reverse),
                int(diagnostics),
                out,
                flux,
            ],
            device="cuda:0",
        )
        e = out.numpy()
        if (
            not np.isfinite(e).all()
            or e[..., 2].max() > 1e-7
            or e[..., 6].min() < -1e-9
        ):
            raise ArithmeticError("Invalid product inventory")
        merit = e[..., 0] if objective == "mass" else e[..., :2].max(-1)
        choice = merit.argmin(2)
        ii = np.arange(shape[0])[:, None]
        rr = np.arange(shape[1])[None, :]
        return (
            merit[ii, rr, choice].max(1),
            e[ii, rr, choice],
            doses[rr, choice],
            flux.numpy()[ii, rr, choice] if diagnostics else None,
        )
