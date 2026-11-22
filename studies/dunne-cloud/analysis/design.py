"""Design sensitivity to the achievable gas precision in a future experiment."""

import itertools
import numpy as np
import torch
from fit import FITDIR, log_rates
from fraction_profile import read_best
from reference import AIR, ACID, BAND, J, NH3, OUT, Q, T, WINDOW, dump_csv


def covariates(temp, acid, ammonia):
    temp, acid, ammonia = np.broadcast_arrays(temp, acid, ammonia)
    return torch.tensor(
        np.stack(
            [
                (temp - 248) / 50,
                np.log(acid),
                np.log(
                    ammonia * 1.005 * 101325 / (1.380649e-23 * temp) * 1e-24
                ),
                np.full_like(temp, -690.0),
            ],
            axis=-1,
        )
    )


def range_for(topo, temp, acid, nh3, af, afhi, nf, nfhi):
    lows = []
    highs = []
    dl = covariates(temp, acid * af, nh3 * nf)
    dh = covariates(temp, acid * afhi, nh3 * nfhi)
    for shape in ["linear", "curved"]:
        rs = read_best(
            FITDIR / f"complete_{topo}_{shape}_nominal_neutral_group.json"
        )
        for r in rs:
            if r["cost"] > 1.02 * rs[0]["cost"]:
                continue
            params = torch.tensor(r["parameters"])[None, :].expand(
                len(temp), -1
            )
            lows.append(log_rates(params, dl, topo, shape)[0].exp().numpy())
            highs.append(log_rates(params, dh, topo, shape)[0].exp().numpy())
    return np.min(lows, axis=0), np.max(highs, axis=0)


def main():
    rows = []
    holdout = []
    for split in ["223_low", "248_low"]:
        for topo, shape in itertools.product(
            ["assisted", "full"], ["linear", "curved"]
        ):
            runs = read_best(
                FITDIR / f"{split}_{topo}_{shape}_nominal_neutral_group.json"
            )
            for line in (
                [332, 366] if split == "223_low" else [334, 336, 338, 344]
            ):
                i = line - 2
                for factor in np.geomspace(0.1, 10, 81):
                    x = covariates(np.array([T[i]]), ACID[i], NH3[i] * factor)
                    pred = float(
                        log_rates(
                            torch.tensor(runs[0]["parameters"])[None, :],
                            x,
                            topo,
                            shape,
                        )[0].exp()[0]
                    )
                    holdout.append(
                        dict(
                            split=split,
                            line=line,
                            topology=topo,
                            shape=shape,
                            NH3_factor=factor,
                            ammonia_pptv=NH3[i] * factor,
                            predicted=pred,
                            observed=J[i],
                        )
                    )
    for temp in [223.0, 248.3, 278.5]:
        m = (
            WINDOW
            & (Q == 0)
            & (BAND == round(temp) if temp != 278.5 else BAND == 278)
        )
        acid, nh = np.meshgrid(
            np.geomspace(ACID[m].min(), ACID[m].max(), 25),
            np.geomspace(0.02, 2.0, 26),
        )
        acid = acid.ravel()
        nh = nh.ravel()
        t = np.full_like(acid, temp)
        for prec, a1, a2, n1, n2 in [
            ("archived_gas_envelope", 2 / 3, 1.5, 0.1, 10),
            ("ammonia20pct", 2 / 3, 1.5, 0.8, 1.2),
            ("both10pct", 0.9, 1.1, 0.9, 1.1),
        ]:
            l, h = range_for("full", t, acid, nh, a1, a2, n1, n2)
            u, v = range_for("assisted", t, acid, nh, a1, a2, n1, n2)
            for k in range(len(nh)):
                rows.append(
                    dict(
                        T_K=temp,
                        acid_cm3=acid[k] * 1e6,
                        NH3_pptv=nh[k],
                        precision=prec,
                        full_min=l[k],
                        full_max=h[k],
                        assisted_min=u[k],
                        assisted_max=v[k],
                        separation=max(l[k] / v[k], u[k] / h[k]),
                        both_above_floor=min(l[k], u[k]) >= 0.001,
                        both_below_100=max(h[k], v[k]) <= 100,
                        within_seen_ammonia=NH3[m].min()
                        <= nh[k]
                        <= NH3[m].max(),
                    )
                )
    dump_csv(OUT / "precision_design.csv", rows)
    dump_csv(OUT / "holdout_gas_sweep.csv", holdout)


if __name__ == "__main__":
    main()
