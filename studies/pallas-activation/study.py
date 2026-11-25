"""Estimate conditional particle class shares for a fixed salt blend and two explicit organic endmembers."""

import argparse
import json
import time

import numpy as np
import pandas as pd
import warp as wp
from calibration import KNOTS, WIDTH, basis, fit_kappa, fit_pop
from observations import SET, P, cohort, load_bulk, recipes, volume_spectra
from scipy.optimize import linprog, minimize
from scipy.special import entr, expit
from tables import CARBON, DENS

PRIMARY = "two organics"
MODES = [
    PRIMARY,
    "sorbitol only",
    "squalane only",
    "external",
    "size-independent",
    "narrow response",
    "wide response",
    "lower tension",
    "less carbon",
    "more carbon",
    "nonwetting oil",
    "fine classes",
    "more sulfate",
]
SIZES = np.array([50.0, 80.0, 120.0, 180.0])


class Population:
    def __init__(
        self, table, recipe, spec, mode, dens=DENS, carbon_density=CARBON
    ):
        self.carbon_density = carbon_density
        dry_salt = recipe.NaCl_volume + (
            recipe.AS_all_sulfate_volume
            if mode == "more sulfate"
            else recipe.AS_volume
        )
        r = recipe.NaCl_volume / dry_salt
        carbon = recipe.OC_ugC_m3 * {
            "less carbon": 0.75,
            "more carbon": 1.25,
        }.get(mode, 1.0)
        self.target = carbon / dry_salt
        self.r = r
        c = table["comps"]
        keep = (
            np.ones(len(c), bool)
            if mode == "fine classes"
            else np.all(np.isclose(c * 4, np.round(c * 4)), axis=1)
        )
        if mode == "external":
            keep &= c.max(axis=1) == 1
        if mode == "sorbitol only":
            keep &= c[:, 2] == 0
        if mode == "squalane only":
            keep &= c[:, 1] == 0
        self.c = c[keep]
        self.N = len(self.c)
        self.knots = np.log([100.0]) if mode == "size-independent" else KNOTS
        self.volume = spec[1] @ basis(spec[0] + np.log(80.0), self.knots)
        balance = self.c[:, 1:] @ carbon_density - self.target * self.c[:, 0]
        self.A = np.vstack(
            [
                np.kron(np.eye(len(self.knots)), np.ones(self.N)),
                self.feature(self.volume, balance / max(1.0, self.target)),
            ]
        )
        self.b = np.r_[np.ones(len(self.knots)), 0.0]
        pureorg = np.where((self.c[:, 0] == 0) & (self.c.max(axis=1) == 1))[0]
        sugar_c = (self.c[pureorg, 1:] @ carbon_density).mean()
        f = sugar_c / (sugar_c + self.target)
        seed = np.zeros(self.N)
        seed[pureorg] = (1 - f) / len(pureorg)
        seed[self.c[:, 0] == 1] = f
        self.seed = np.tile(seed, len(self.knots))
        i = min(int(r * 16), 15)
        a = r * 16 - i
        self.ld = np.log(
            ((1 - a) * table["d"][i] + a * table["d"][i + 1])[keep]
        )
        if mode == "lower tension":
            self.ld += np.log(60 / 72)
        if mode == "nonwetting oil":
            self.ld[self.c[:, 2] == 1] = np.inf
        self.width = WIDTH * {"narrow response": 0.5, "wide response": 1.4}.get(
            mode, 1.0
        )
        self.mass = self.c * np.r_[1.77 * (1 - r) + 2.165 * r, dens]
        self.mtot = self.mass.sum(axis=1)
        self.enth = self.mtot * entr(self.mass / self.mtot[:, None]).sum(axis=1)

    @staticmethod
    def feature(b, v):
        return np.outer(b, v).ravel()

    def design(self, x, j):
        return (
            basis(x, self.knots)[:, :, None]
            * expit((x[:, None] - self.ld[:, j].T) / self.width)[:, None, :]
        ).reshape(len(x), -1)

    def inventories(self, z):
        return self.volume @ z.reshape(-1, self.N) @ self.c

    def stats(self, z):
        w = self.volume @ z.reshape(-1, self.N)
        v = w @ self.c
        m = w @ self.mass
        total = m.sum()
        H = (w @ self.enth) / total
        chi = np.expm1(H) / np.expm1(entr(m / total).sum())
        frac = self.mass[:, 0] / self.mtot
        h2 = (w @ (self.mtot * (entr(frac) + entr(1 - frac)))) / total
        s = m[0] / total
        chi2 = np.expm1(h2) / np.expm1(entr(s) + entr(1 - s))
        size = basis(np.log(SIZES), self.knots) @ z.reshape(-1, self.N) @ self.c
        result = dict(
            salt=v[0],
            sorbitol=v[1],
            squalane=v[2],
            organic_sorbitol=v[1] / v[1:].sum(),
            carbon_sorbitol=v[1]
            * self.carbon_density[0]
            / (v[1:] @ self.carbon_density),
            chi=chi,
            chi2=chi2,
        )
        for i, r in enumerate(SIZES):
            for k, name in enumerate(["salt", "sorbitol", "squalane"]):
                result[f"{name}{int(r)}"] = size[i, k]
        return result


def constrained(z, M, t, A, b, vec, cap, other=()):
    """Optimize a linear or differentiable feature on the calibration loss sublevel."""
    cons = [
        dict(type="eq", fun=lambda w: A @ w - b, jac=lambda w: A),
        dict(
            type="ineq",
            fun=lambda w: 100 * (cap - np.sum((M @ w - t) ** 2)),
            jac=lambda w: -200 * M.T @ (M @ w - t),
        ),
        *other,
    ]
    f = vec if callable(vec) else lambda w: (vec @ w, vec)
    res = minimize(
        f,
        z,
        jac=True,
        bounds=[(0.0, 1.0)] * len(z),
        constraints=cons,
        method="SLSQP",
        options={"ftol": 5e-10, "maxiter": 350},
    )
    if not res.success and res.status != 8:
        raise RuntimeError(res.message)
    w = res.x
    assert (
        np.max(abs(A @ w - b)) < 2e-7
        and np.sum((M @ w - t) ** 2) < cap + 3e-8
        and w.min() > -1e-7
    )
    return w


def ratio_feature(vec, den, sgn):
    def fun(w):
        a, b = vec @ w, den @ w
        return sgn * a / b, sgn * (vec * b - den * a) / b**2

    return fun


def profile(pop, z, M, t):
    rmse = np.linalg.norm(M @ z - t)
    cap = (rmse + 0.01) ** 2
    bulk = [pop.feature(pop.volume, c) for c in pop.c.T]
    salt_delta = pop.feature(
        np.diff(basis(np.log([50.0, 120.0]), pop.knots), axis=0)[0], pop.c[:, 0]
    )
    res = {}
    for key in ["delta", "organic_sorbitol"]:
        for direction, side in [(1, "lo"), (-1, "hi")]:
            vec = (
                direction * salt_delta
                if key == "delta"
                else ratio_feature(bulk[1], bulk[1] + bulk[2], direction)
            )
            w = constrained(z, M, t, pop.A, pop.b, vec, cap)
            res[key + "_" + side] = float(
                salt_delta @ w if key == "delta" else pop.stats(w)[key]
            )
    # Fix the organic ratio at its optimum to also fix all dry inventories. The entropy
    # numerator is then linear, and its extrema give the true conditional index bounds.
    A = np.vstack([pop.A, bulk[1]])
    b = np.r_[pop.b, bulk[1] @ z]
    entropy = pop.feature(pop.volume, pop.enth)
    for direction, side in [(1, "lo"), (-1, "hi")]:
        w = constrained(z, M, t, A, b, direction * entropy, cap)
        res["chi_" + side] = pop.stats(w)["chi"]
    return res


def solve(p, x, j, y, scale):
    z, pred, oracle = fit_pop(p, x, j, y, scale)
    A = p.design(x, j) * scale[:, None]
    t = y * scale
    err = A @ z - t
    g = 2 * A.T @ err
    lp = linprog(g, A_eq=p.A, b_eq=p.b, bounds=(0.0, 1.0), method="highs")
    assert lp.success
    bound = np.linalg.norm(err) - np.sqrt(
        max(0, err @ err - max(0, g @ z - lp.fun))
    )
    assert bound < 1e-4 and np.max(abs(p.A @ z - p.b)) < 1e-8
    return z, pred, oracle, bound, A, t


def main(pilot):
    start = time.monotonic()
    wp.init()
    dev = wp.get_device("cuda:0")
    tab = np.load(P / "thresholds.npz")
    chem = recipes()
    scans, train, test = cohort()
    spec = volume_spectra(train + test)
    edges, bulk = load_bulk(test)
    dates = sorted(set(spec) & set(chem.index))
    dates = dates[:1] if pilot else dates
    comparators = []
    empirical_omitted = []
    empirical_bulk = []
    fits = []
    curves = []
    profs = []
    shares = []
    counts = []
    omitted = []
    samples = []
    max_grad = max_gap = max_budget = 0.0
    nodes, v = np.polynomial.legendre.leggauss(12)
    bins = (
        edges[1:, None] + edges[:-1, None] + np.diff(edges)[:, None] * nodes
    ) / 2
    for di, date in enumerate(dates):
        scan = scans[scans.date == date].sort_values(["j", "Dp"])
        x = np.log(scan.Dp.to_numpy())
        y = scan.fraction.to_numpy()
        j = scan.j.to_numpy()
        scale = 1 / np.sqrt(5 * np.bincount(j)[j])
        empirical, forward, theta = fit_kappa(x, j, y, scale, True)
        comparators.append(
            (
                date,
                "size-dependent kappa",
                len(y),
                np.linalg.norm((empirical - y) * scale),
                abs(empirical - y) @ (scale**2),
                np.exp(theta[0]),
                theta[1],
            )
        )
        for jj in range(5):
            keep = j != jj
            _, predict, _ = fit_kappa(
                x[keep], j[keep], y[keep], scale[keep] * np.sqrt(5 / 4), True
            )
            empirical_omitted.append(
                (
                    date,
                    SET[jj],
                    np.sqrt(
                        np.mean((predict(x[~keep], j[~keep]) - y[~keep]) ** 2)
                    ),
                )
            )
            if date in test and jj != 2:
                mass, obs = bulk[date, jj]
                probability = forward(
                    bins.ravel(), np.full(bins.size, jj)
                ).reshape(bins.shape) @ (v / 2)
                cc = mass @ probability
                empirical_bulk.append(
                    (
                        date,
                        SET[jj],
                        len(obs),
                        obs.mean(),
                        cc.mean(),
                        np.mean(abs(obs - cc)),
                    )
                )
        for mode in MODES:
            p = Population(tab, chem.loc[date], spec[date], mode)
            z, pred, oracle, gap, M, t = solve(p, x, j, y, scale)
            max_grad = max(max_grad, oracle)
            max_gap = max(max_gap, gap)
            inventory = p.inventories(z)
            max_budget = max(
                max_budget,
                abs(inventory[1:] @ CARBON / (inventory[0] * p.target) - 1),
            )
            fits.append(
                dict(
                    date=date,
                    model=mode,
                    n=len(j),
                    rmse=np.linalg.norm((pred - y) * scale),
                    mae=abs(pred - y) @ (scale**2),
                    **p.stats(z),
                )
            )
            if mode == PRIMARY:
                profs.append(dict(date=date, **profile(p, z, M, t)))
                for knot, row in zip(p.knots, z.reshape(-1, p.N)):
                    shares.extend(
                        (date, np.exp(knot), *c, a) for c, a in zip(p.c, row)
                    )
                curves.extend(zip([date] * len(x), np.exp(x), SET[j], y, pred))
                sizes, ix = np.unique(x, return_inverse=True)
                ns = np.full(len(sizes), 10000 // len(sizes))
                ns[: 10000 % len(sizes)] += 1
                cp = np.maximum(basis(sizes, p.knots) @ z.reshape(-1, p.N), 0.0)
                cp /= cp.sum(axis=1, keepdims=True)
                for seed in range(32):
                    rng = np.random.default_rng([20261125, di, seed])
                    draws = np.array(
                        [rng.multinomial(n, pr) / n for n, pr in zip(ns, cp)]
                    )
                    pp = np.sum(
                        draws[ix]
                        * expit((x[:, None] - p.ld[:, j].T) / p.width),
                        axis=1,
                    )
                    samples.append(
                        (date, seed, np.linalg.norm((pp - pred) * scale))
                    )
                for jj in range(5):
                    keep = j != jj
                    zz, *_ = solve(
                        p,
                        x[keep],
                        j[keep],
                        y[keep],
                        scale[keep] * np.sqrt(5 / 4),
                    )
                    val = p.design(x[~keep], j[~keep]) @ zz
                    omitted.append(
                        (
                            date,
                            SET[jj],
                            np.sqrt(np.mean((val - y[~keep]) ** 2)),
                            np.sqrt(np.mean((pred[~keep] - y[~keep]) ** 2)),
                        )
                    )
            if date in test:
                for jj in (0, 1, 3, 4):
                    mass, obs = bulk[date, jj]
                    pb = (
                        p.design(bins.ravel(), np.full(bins.size, jj)) @ z
                    ).reshape(bins.shape) @ (v / 2)
                    cc = mass @ pb
                    counts.append(
                        (
                            date,
                            mode,
                            SET[jj],
                            len(obs),
                            obs.mean(),
                            cc.mean(),
                            np.mean(abs(obs - cc)),
                        )
                    )
    pre = "pilot-" if pilot else ""

    def save(name, items, columns=None):
        pd.DataFrame(
            items, columns=None if columns is None else columns.split()
        ).to_csv(P / (pre + name), index=False, float_format="%.12g")

    save(
        "comparator.csv", comparators, "date model n rmse mae kappa80 exponent"
    )
    save("comparator-omitted.csv", empirical_omitted, "date SS omitted_rmse")
    save("comparator-bulk.csv", empirical_bulk, "date SS n observed fitted mae")
    save("fits.csv", fits)
    save("profiles.csv", profs)
    save("shares.csv.gz", shares, "date knot salt sorbitol squalane share")
    save("curves.csv", curves, "date D SS observed fitted")
    save("bulk.csv", counts, "date model SS n observed fitted mae")
    save("omitted.csv", omitted, "date SS omitted_rmse fitted_rmse")
    save("samples.csv", samples, "date seed mc_rmse")
    meta = dict(
        device=dev.name,
        dates=len(dates),
        daily_fits=len(fits),
        omitted_refits=len(omitted),
        checked_particles=len(samples) * 10000,
        max_fraction_budget_residual=max_budget,
        max_ad_check=max_grad,
        max_rmse_optimality_gap=max_gap,
        wall_s=time.monotonic() - start,
    )
    (P / (pre + "run.json")).write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    main(ap.parse_args().pilot)
