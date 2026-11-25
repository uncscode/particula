"""Refit replacing the six-carbon polyol with a three-carbon polyol."""

import numpy as np
import pandas as pd
from observations import P, cohort, load_bulk, recipes, volume_spectra
from study import PRIMARY, Population, solve
from tables import DENS, GROUPS, grid, prepare, thresholds


def main():
    g = {
        ("glycerol" if n == "sorbitol" else n): (
            {150: 2, 151: 1, 153: 3} if n == "sorbitol" else v
        )
        for n, v in GROUPS.items()
    }
    dens = np.array([1.258, DENS[1]])
    carbon = dens * np.array([3.0, 30.0]) * 12.011 / np.array([92.094, 422.82])
    model = prepare("cuda", g)
    c = grid(4)
    r = np.linspace(0, 1, 17)
    data = {
        "comps": c,
        "d": np.array([thresholds(model, c, v, 385, dens)[0] for v in r]),
    }
    chem = recipes()
    scans, train, test = cohort()
    spec = volume_spectra(train + test)
    out = []
    edges, bulk = load_bulk(test)
    bud = []
    u, v = np.polynomial.legendre.leggauss(12)
    bins = (
        edges[1:, None] + edges[:-1, None] + np.diff(edges)[:, None] * u
    ) / 2
    for date in sorted(set(spec) & set(chem.index)):
        p = Population(data, chem.loc[date], spec[date], PRIMARY, dens, carbon)
        scan = scans[scans.date == date].sort_values(["j", "Dp"])
        j = scan.j.to_numpy()
        scale = 1 / np.sqrt(5 * np.bincount(j)[j])
        y = scan.fraction.to_numpy()
        z, pred, *_ = solve(p, np.log(scan.Dp.to_numpy()), j, y, scale)
        info = p.stats(z)
        out.append(
            dict(
                date=date,
                rmse=np.linalg.norm((pred - y) * scale),
                salt50=info["salt50"],
                salt120=info["salt120"],
                chi=info["chi"],
                organic_glycerol=info["organic_sorbitol"],
            )
        )
        actual = p.inventories(z)
        assert abs(actual[1:] @ carbon / (actual[0] * p.target) - 1) < 1e-8
        if date in test:
            for jj in [0, 1, 3, 4]:
                m, obs = bulk[date, jj]
                calc = m @ (
                    (
                        p.design(bins.ravel(), np.full(bins.size, jj)) @ z
                    ).reshape(bins.shape)
                    @ (v / 2)
                )
                bud.append(
                    dict(
                        date=date,
                        setting_index=jj,
                        fitted=calc.mean(),
                        mae=abs(calc - obs).mean(),
                    )
                )
    pd.DataFrame(out).to_csv(P / "alternate.csv", index=False)
    pd.DataFrame(bud).to_csv(P / "alternate-bulk.csv", index=False)


if __name__ == "__main__":
    main()
