# %% Exact continuous feasibility regions for shared calibration
"""Linearize the fixed binary mixing law by inverting each humidity channel."""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from inverse import inputs, to_sodium, PACKET, zsr


class JointModel:
    """Unknowns v[d,t], p[h], q[h], with v = p*GF^3 + q at fixed RH."""

    def __init__(self, y, err, center, width, monotonic=True):
        D, T, H = y.shape
        self.D, self.T, self.H = D, T, H
        n = D * T + 2 * H
        rows = []
        rhs = []

        def add(terms, limit):
            row = np.zeros(n)
            for col, val in terms.items():
                row[col] = val
            rows.append(row)
            rhs.append(limit)

        bounds = [(0, 1)] * (D * T) + [(None, None)] * (2 * H)
        for h in range(H):
            p = D * T + h
            q = p + H
            a, b = center[:, h]
            lo = (center[:, h] - width[:, h]) ** 3
            hi = (center[:, h] + width[:, h]) ** 3
            sign = np.sign(b - a)
            # Dry endpoint: v=0 at some F in [G0lo^3,G0hi^3].
            # Reacted endpoint: v=1 in the corresponding product interval.
            for j in range(2):
                left, right = (lo[j], hi[j]) if sign > 0 else (hi[j], lo[j])
                add({p: left, q: 1}, j)
                add({p: -right, q: -1}, -j)
            for d in range(D):
                for t in range(T):
                    v = d * T + t
                    aa = (y[d, t, h] - err[d, t, h]) ** 3
                    bb = (y[d, t, h] + err[d, t, h]) ** 3
                    left, right = (aa, bb) if sign > 0 else (bb, aa)
                    add({p: left, q: 1, v: -1}, 0)
                    add({p: -right, q: -1, v: 1}, 0)
        if monotonic:
            for d in range(D):
                for t in range(1, T):
                    add({d * T + t - 1: 1, d * T + t: -1}, 0)
        self.A = csr_matrix(rows)
        self.b = rhs
        self.bounds = bounds
        self.zero = np.zeros(n)
        self.check = self.solve()

    def solve(self, objective=None, limits=None):
        res = linprog(
            self.zero if objective is None else objective,
            A_ub=self.A,
            b_ub=self.b,
            bounds=self.bounds if limits is None else limits,
            method="highs",
        )
        if res.status not in (0, 2):
            raise RuntimeError(res.message)
        return res

    def extremes(self):
        out = []
        solutions = []
        for i in range(self.D * self.T):
            cost = self.zero.copy()
            cost[i] = 1
            lo = self.solve(cost).x
            hi = self.solve(-cost).x
            out.append([lo[i], hi[i]])
            solutions.extend([lo, hi])
        return np.array(out).reshape(self.D, self.T, 2), np.mean(solutions, axis=0)

    def condition_pair(self, i, j, bounds, steps):
        # Certified outer bounds and attainable inner bounds on f_j/f_i.
        # For every denominator slab solve extrema of v_j within the slab.
        edges = np.linspace(*bounds, steps + 1)
        outer = [np.inf, -np.inf]
        inner = [np.inf, -np.inf]
        c = self.zero.copy()
        c[j] = 1
        for a, b in zip(edges, edges[1:]):
            lim = self.bounds.copy()
            lim[i] = (a, b)
            left = self.solve(c, lim)
            right = self.solve(-c, lim)
            if not left.success:
                continue
            assert right.success
            outer[0] = min(outer[0], to_sodium(left.x[j]) / to_sodium(b))
            outer[1] = max(outer[1], to_sodium(right.x[j]) / to_sodium(a))
            for x in [left.x, right.x]:
                r = to_sodium(x[j]) / to_sodium(x[i])
                inner[0] = min(inner[0], r)
                inner[1] = max(inner[1], r)
        return [outer[0], inner[0], inner[1], outer[1]]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--slabs", type=int, default=16)
    pa.add_argument("--stage", default="joint")
    args = pa.parse_args()
    out = PACKET / args.stage
    out.mkdir(exist_ok=True)
    ds, ts, rh, y, sd, c, w = inputs()
    pair = np.isin(ds, [200, 300])
    allsize = np.ones(len(ds), bool)
    specs = [
        ("all", allsize, np.ones(6, bool)),
        ("pair", pair, np.ones(6, bool)),
        ("low", allsize, rh <= 70),
        ("high", allsize, rh >= 80),
        ("omit75", allsize, rh != 75),
    ]
    records = []
    ratio = []
    witness = []
    ext = []
    metadata = []
    for label, sel, humid in specs:
        yy = y[sel][:, :, humid]
        ss = sd[sel][:, :, humid]
        cc = c[:, humid]
        ww = w[:, humid]

        def model(mult):
            return JointModel(yy, mult * ss + 0.005, cc, ww)

        lo = 0
        hi = 4
        assert model(hi).check.success
        for _ in range(17):
            mid = (lo + hi) * 0.5
            if model(mid).check.success:
                hi = mid
            else:
                lo = mid
        for mult in [1, 2]:
            m = model(mult)
            records.append((label, mult, m.check.success, lo, hi))
            if not m.check.success or label not in ["all", "pair"]:
                continue
            vol, wit = m.extremes()
            n = m.D * m.T
            p = wit[n : n + m.H]
            q = wit[n + m.H :]
            end = np.stack([np.cbrt(-q / p), np.cbrt((1 - q) / p)])
            vv = wit[:n].reshape(m.D, m.T)
            pred = zsr(vv[:, :, None], *end)
            assert np.all(abs(yy - pred) <= mult * ss + 0.005 + 1e-7)
            assert np.all(abs(end - cc) <= ww + 1e-7)
            assert np.all(np.diff(vv, axis=1) >= -1e-8)
            metadata.append(
                dict(
                    domain=label,
                    scatter_multiple=mult,
                    rh_pct=rh[humid].tolist(),
                    fresh=end[0].tolist(),
                    product=end[1].tolist(),
                    worst_abs_gf=float(abs(yy - pred).max()),
                )
            )
            for i, d in enumerate(ds[sel]):
                for j, t in enumerate(ts):
                    ext.append((label, mult, d, t, *vol[i, j], *to_sodium(vol[i, j])))
                    witness.append((label, mult, d, t, vv[i, j], to_sodium(vv[i, j])))
            if label == "pair" or label == "all":
                i, j = np.where(ds[sel] == 200)[0][0], np.where(ds[sel] == 300)[0][0]
                for k, t in enumerate(ts):
                    lim = m.condition_pair(
                        i * m.T + k, j * m.T + k, vol[i, k], args.slabs
                    )
                    ratio.append((label, mult, t, *lim))
    pd.DataFrame(
        records,
        columns=[
            "scope",
            "scatter_multiple",
            "feasible",
            "minimum_scatter_multiple_lo",
            "minimum_scatter_multiple_hi",
        ],
    ).to_csv(out / "consistency.csv", index=False)
    pd.DataFrame(
        ratio,
        columns=[
            "scope",
            "scatter_multiple",
            "time_h",
            "outer_lo",
            "attainable_lo",
            "attainable_hi",
            "outer_hi",
        ],
    ).to_csv(out / "ratios.csv", index=False)
    pd.DataFrame(
        ext,
        columns=[
            "scope",
            "scatter_multiple",
            "diameter_nm",
            "time_h",
            "vmin",
            "vmax",
            "fmin",
            "fmax",
        ],
    ).to_csv(out / "extrema.csv", index=False)
    pd.DataFrame(
        witness,
        columns=["scope", "scatter_multiple", "diameter_nm", "time_h", "v", "f"],
    ).to_csv(out / "example.csv", index=False)
    (out / "endpoints.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.read_csv(out / "consistency.csv").to_string(index=False))
    print(pd.read_csv(out / "ratios.csv").to_string(index=False))


if __name__ == "__main__":
    main()
