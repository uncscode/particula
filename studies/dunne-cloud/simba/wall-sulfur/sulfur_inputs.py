"""Reconstruct sulfur forcing and balanced gas cluster proposals from archived inputs."""

from functools import lru_cache
import json
import numpy as np
from balance import HERE, write
from replay import measurements, curves, ZipFile, ARCHIVE, TEMP, AIR
from particula.dynamics.chemistry.discovery import (
    balanced_steps,
    enumerate_networks,
)
from dimers import calculation


def interp(t, r):
    return np.interp(t, r[:, 0], r[:, 1])


@lru_cache(None)
def inputs():
    t, _, _, _, acid, _, _, _, old_hp, valid, so = measurements()
    with ZipFile(ARCHIVE) as z:
        oh, acid_author = curves(z, "12c")[:2]
        oh_loss = curves(z, "12b")
        l = curves(z, "12d")[-1]
        so_base = curves(z, "13b")[0]
        lamps = curves(z, "02a")
    # These are first-order OH loss frequencies. Dividing by the second-order
    # coefficient recovers the peroxide used in the authors' gas simulation.
    hp = np.maximum(0.0, interp(t, oh_loss[2]) / 1.8e-12)
    rawt = oh[:, 0]
    dt = np.diff(rawt) * 3600
    # Use measured SO2 in place of the authors' SO2 trajectory at the same OH rate.
    k_so = interp(rawt, oh_loss[5]) / interp(rawt, so_base)
    prod = oh[:, 1] * k_so * interp(rawt, np.column_stack([t, so]))
    sink = interp(rawt, l)

    def propagate(source):
        midloss = (sink[:-1] + sink[1:]) * 0.5
        survival = np.exp(-dt * midloss)
        feed = (
            -np.expm1(-dt * midloss)
            / midloss
            * (source[1:] + source[:-1])
            * 0.5
        )
        y = np.zeros(len(rawt))
        y[0] = acid_author[0, 1]
        for j in range(len(dt)):
            y[j + 1] = survival[j] * y[j] + feed[j]
        return np.interp(t, rawt, y)

    predicted = propagate(prod)
    # A unit continuous dark/background source has no initial acid inventory.
    bg = propagate(np.ones_like(prod)) - propagate(np.zeros_like(prod))
    lamp = interp(t, lamps[0])
    lamp /= max(lamp)
    z = np.column_stack([t, so * lamp / np.median(so)])
    wall_unit = propagate(interp(rawt, z)) - propagate(np.zeros_like(prod))
    # Fit shared constant residual gas production to measured monomer in training.
    train = (t - t[0] < 3.8) & valid
    corrected = 1.15 * acid
    bg_rate = max(
        0.0,
        float(
            bg[train] @ (corrected - predicted)[train] / (bg[train] @ bg[train])
        ),
    )
    # A 385 nm wall return has no identifiable amplitude without that lamp in training.
    results = []
    for name, v in [
        ("OH_SO2", predicted),
        ("OH_SO2_background", predicted + bg_rate * bg),
        ("authors_model", interp(t, acid_author)),
    ]:
        for label, mask in [
            ("early", train),
            ("peak", (t - t[0] >= 3.8) & (t - t[0] <= 6.0001) & valid),
            ("transfer", (t - t[0] >= 6.36) & valid),
        ]:
            results.append(
                dict(
                    model=name,
                    interval=label,
                    acid_rmse_cm3=np.mean((v[mask] - corrected[mask]) ** 2)
                    ** 0.5,
                    acid_bias_cm3=np.mean(v[mask] - corrected[mask]),
                )
            )
    write("gas_acid_scores.csv", results)
    write(
        "gas_inputs.csv",
        [
            dict(
                hour=h,
                acid_observed_cm3=a,
                finite_acid=int(v),
                OH_SO2_acid_cm3=ps,
                background_added_cm3=b * bg_rate,
                peroxide_archived_model_cm3=p,
                peroxide_previous_proxy_cm3=q,
                wall_return_unit_seconds=w,
            )
            for h, a, v, ps, b, p, q, w in zip(
                t - t[0], acid, valid, predicted, bg, hp, old_hp, wall_unit
            )
        ],
    )
    return hp, predicted, predicted + bg_rate * bg, bg_rate


def networks():
    names = ["A", "B", "AB", "A2", "A2B", "A2B2"]
    atoms = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [2, 1], [2, 2]])
    raw = balanced_steps(["gas"] * 6, atoms.tolist())
    # Association only; evaporation is evaluated in the reduced kinetic solver.
    bank = [
        r
        for r in raw
        if sum(v for v in r if v < 0) == -2 and sum(v for v in r if v > 0) == 1
    ]
    keep = {}
    for j in (3, 4, 5):
        target = tuple(
            [-int(atoms[j, 0]), -int(atoms[j, 1])]
            + [int(i == j) for i in range(2, 6)]
        )
        res = enumerate_networks(
            bank,
            [True, True, False, False, False, False],
            [j],
            max_steps=3,
            max_trials=1000,
            max_candidates=200,
            net=target,
        )
        assert res.networks and res.limit is None
        keep[names[j]] = [list(a) for a in res.networks]
    return {
        "names": names,
        "pools": "A=H2SO4, B=NH3",
        "bank": bank,
        "networks": keep,
    }


def make_dimer():
    t, _, _, _, acid, *_ = measurements()
    grid = np.geomspace(1e3, 2.5e8, 36)
    rows = []
    for thermo in ["optimized", "posterior_median"]:
        for mon in grid:
            c = calculation(TEMP, 58.0, mon, 2e-9 * AIR, thermo, 6.4e-20)
            rows.append(
                dict(thermo=thermo, monomer_cm3=mon, dimer_cm3=c["dimer_cm3"])
            )
    write("dimer_lookup.csv", rows)
    (HERE / "cluster_networks.json").write_text(
        json.dumps(networks(), indent=2) + "\n"
    )


if __name__ == "__main__":
    make_dimer()
    print("background", inputs()[-1])
