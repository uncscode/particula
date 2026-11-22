"""Independent conservative size-transport replay with prognostic Brownian coagulation."""

import itertools
from functools import lru_cache
import numpy as np
from experiment import forcing, wall_profile
from balance import write
from replay import measurements, GRID, par, TEMP, PRESSURE, KD


@lru_cache(None)
def setup(nbins):
    t, d, old_edges, m, *_ = measurements()
    lo, hi = 2.9, 140.0
    e = np.linspace(np.log(lo), np.log(hi), nbins + 1)
    dx = e[1] - e[0]
    z = np.exp(e)
    c = np.exp((e[1:] + e[:-1]) / 2)
    left = np.maximum(e[:-1, None] / np.log(10), old_edges[None, :-1])
    right = np.minimum(e[1:, None] / np.log(10), old_edges[None, 1:])
    initial = np.maximum(right - left, 0) @ m[:, 0]
    boundary = np.array(
        [np.interp(np.log(lo), np.log(d), r) / np.log(10.0) for r in m.T]
    )
    weights = np.array(
        [
            np.maximum(
                0, np.minimum(e[1:], np.log(b)) - np.maximum(e[:-1], np.log(a))
            )
            / dx
            for a, b in [
                (5, 9.7),
                (9.7, 10 ** old_edges[-1]),
                (15, 10 ** old_edges[-1]),
            ]
        ]
    )
    v = c**3
    K = (
        par.dynamics.get_brownian_kernel_via_system_state(
            c * 0.5e-9, 1350 * np.pi / 6 * (c * 1e-9) ** 3, TEMP, PRESSURE
        )
        * 1e6
    )
    i, j = np.triu_indices(nbins)
    kernel = K[i, j] * np.where(i == j, 0.5, 1.0)
    vol = v[i] + v[j]
    hi = np.searchsorted(v, vol).clip(1, nbins)
    # Overflow products leave the represented domain. Parents always debit their full mass.
    frac = (vol - v[hi - 1]) / (np.r_[v, 2 * v[-1]][hi] - v[hi - 1])
    w = np.where(hi == nbins, 1.0, frac)

    def dg(n):
        p = kernel * n[i] * n[j]
        gain = np.bincount(hi, weights=p * w, minlength=nbins + 1)[
            :nbins
        ] + np.bincount(hi - 1, weights=p * (1 - w), minlength=nbins)
        q = gain - n * (K @ n)
        return q

    probe = np.exp(-(((np.log(c) - np.log(8)) / 0.4) ** 2)) * 100
    q = dg(probe)
    p = kernel * probe[i] * probe[j]
    overflow = hi == nbins
    assert np.isclose(q.sum() + p[overflow].sum(), -p.sum(), rtol=2e-12)
    assert abs(q @ v + p[overflow] @ vol[overflow]) < 1e-11 * (p @ vol)
    return z, c, dx, initial, boundary, weights, dg


def simulate(
    model, gas, oz, px, coag=True, nbins=80, count=181, max_step=30, clear=False
):
    t = measurements()[0][:count]
    z, c, dx, n0, boundary, weights, dg = setup(nbins)
    n = n0.copy()
    g, inputs = forcing(model, gas)
    vel0 = np.interp(z, GRID, g[:, 1]) / z
    vel1 = np.interp(z, GRID, g[:, 3]) / z
    hz = wall_profile("dark", c) + KD
    events = []
    if clear:
        from field import charges

        events = [
            (hh * 3600, np.interp(np.log(c), np.log(diam), f, left=0, right=0))
            for hh, diam, f in charges()
        ]
    result = [weights @ n]
    ts = (t - t[0]) * 3600
    for j, delta in enumerate(np.diff(ts)):
        ns = max(1, int(np.ceil(delta / max_step)))
        maxspeed = max(
            np.max(
                vel0 * inputs[jj, 0]
                + vel1 * (oz * inputs[jj, 1] + px * inputs[jj, 2])
            )
            for jj in (j, j + 1)
        )
        ns = max(ns, int(np.ceil(maxspeed * delta / (0.40 * dx))))
        h = delta / ns
        for k in range(ns):
            w = (k + 0.5) / ns
            ac, o, p = (1 - w) * inputs[j] + w * inputs[j + 1]
            vel = vel0 * ac + vel1 * (oz * o + px * p)
            bd = (1 - w) * boundary[j] + w * boundary[j + 1]

            def rhs(n):
                jump = np.diff(n)
                slope = np.zeros_like(n)
                slope[1:-1] = np.where(
                    jump[:-1] * jump[1:] > 0,
                    np.sign(jump[:-1])
                    * np.minimum(abs(jump[:-1]), abs(jump[1:])),
                    0,
                )
                flux = vel * np.r_[bd, (n + 0.5 * slope) / dx]
                return -np.diff(flux) - hz * n + (dg(n) if coag else 0)

            for switch, fraction in events:
                if ts[j] + k * h < switch <= ts[j] + (k + 1) * h:
                    n = n * (1 - fraction)
            pred = n + h * rhs(n)
            n = 0.5 * (n + pred + h * rhs(pred))
            if min(n) < -1e-8:
                raise ValueError("negative spectrum")
        result.append(weights @ n)
    return np.asarray(result)


def main():
    t, *_, obs, _, _, _ = measurements()
    h = t - t[0]
    selected = []
    checks = []
    candidates = {
        "parent": [(0.0, 0.0)],
        "ozone": [
            (x, 0.0) for x in (0, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016)
        ],
        "peroxide": [
            (0.0, x) for x in (0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
        ],
        "both": list(
            itertools.product(
                [0, 0.0002, 0.0004, 0.0008], [0, 0.01, 0.04, 0.08]
            )
        ),
        "gas_oh": [(0.0, 0.0)],
        "dimer_optimized_pooled": [(1.0, 0.0)],
        "bulk6": [(1.0, 1.0)],
    }
    cache = {}
    for label, s in candidates.items():
        model = (
            label if label.startswith(("dimer", "gas", "bulk")) else "effective"
        )
        group = []
        for gas, (a, b) in itertools.product(
            (
                [0.9, 1.0, 1.1]
                if model.startswith("gas")
                else [1.05, 1.15, 1.25]
            ),
            s,
        ):
            key = (model, gas, a, b)
            if key not in cache:
                y = simulate(*key, count=114)[:, 1]
                cache[key] = np.mean((y - obs[:114, 1]) ** 2) ** 0.5
            group.append((cache[key], key))
        score, key = min(group)
        selected.append((label, key))
        for nb, dt, coal in [(80, 30, True), (140, 15, True), (140, 15, False)]:
            y = simulate(*key, nbins=nb, max_step=dt, coag=coal)
            for idx, band in enumerate(["5_to_9p7", "9p7_plus", "15_plus"]):
                mask = h[:181] >= 3.8
                checks.append(
                    dict(
                        family=label,
                        gas=key[1],
                        a=key[2],
                        b=key[3],
                        bins=nb,
                        step=dt,
                        coag=int(coal),
                        channel=band,
                        train_rmse=np.mean(
                            (y[~mask, idx] - obs[:181, idx][~mask]) ** 2
                        )
                        ** 0.5,
                        peak_rmse=np.mean(
                            (y[mask, idx] - obs[:181, idx][mask]) ** 2
                        )
                        ** 0.5,
                        peak=y[mask, idx].max(),
                    )
                )
    write("sectional_scores.csv", checks)
    write(
        "sectional_search.csv",
        [
            dict(model=m, gas=g, a=a, b=b, training_rmse=s)
            for (m, g, a, b), s in cache.items()
        ],
    )
    later = []
    for name, key in selected:
        y = simulate(*key, nbins=80, count=len(t))
        mask = h >= 6.36
        later.append(
            dict(
                family=name,
                later_rmse=np.mean((y[mask, 1] - obs[mask, 1]) ** 2) ** 0.5,
            )
        )
    write("sectional_transfer.csv", later)


if __name__ == "__main__":
    main()
