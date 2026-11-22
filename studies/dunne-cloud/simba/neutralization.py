"""SiMBA sequential ammonia capture and wet salt growth through the same detector."""

import itertools
import json
import numpy as np
import warp as wp
from particula.dynamics.chemistry.discovery import (
    balanced_steps,
    enumerate_networks,
)
from replay import HERE, GRID, TEMP, arrays, advance, measurements, F
from search import write


def enumerate_salt():
    # Columns are S,H,N,O. Water remains at equilibrium with the wet salt.
    atoms = [(1, 2, 0, 4), (0, 3, 1, 0), (1, 5, 1, 4), (1, 8, 2, 4)]
    names = ["H2SO4", "NH3", "NH4HSO4", "(NH4)2SO4"]
    bank = [
        r
        for r in balanced_steps(["particle"] * 4, atoms)
        if r[0] <= 0 and r[1] < 0 and r[-1] >= 0
    ]
    hit = enumerate_networks(
        bank,
        [True, True, False, False],
        [3],
        max_steps=2,
        max_trials=100,
        max_candidates=10,
        net=(-1, -2, 0, 1),
    )
    assert len(hit.networks) == 1 and hit.limit is None
    return dict(
        species=names,
        composition_SHNO=atoms,
        steps=hit.networks[0],
        examined=hit.examined,
    )


def network_grid(original, amp, midpoint, power):
    d = GRID
    frac = d**power / (d**power + midpoint**power)
    stoich = 2 - amp * (1 - frac)
    water = 0.58 * np.exp(
        -4 * 0.079 * 0.01801528 / (1000 * 8.314462618 * TEMP * d * 1e-9)
    )

    def inventory(s):
        unit = (
            98.079 + 17.03052 * s + 18.01528 * (1 + s) * water / (1 - water)
        ) * 1.66053906660e-27
        return 1350 * np.pi / 6 * (d * 1e-9) ** 3 / unit

    acid = inventory(stoich)
    assert np.all(np.diff(acid) > 0) and np.all(np.diff(stoich * acid) >= 0)
    ratio = np.gradient(inventory(2.0), d, edge_order=2) / np.gradient(
        acid, d, edge_order=2
    )
    result = original.copy()
    result[:, 1:4] *= ratio[:, None]
    return result


def replay(network, pars, count):
    (g, f, i, o), hi = arrays(count, 30, "cuda:0")
    geo = wp.array(network_grid(g.numpy(), *network), dtype=F, device="cuda:0")
    cfg = wp.array(np.array(pars, dtype=float), dtype=F, device="cuda:0")
    out = wp.zeros((len(pars), count, 3), dtype=F, device="cuda:0")
    buf = wp.zeros((len(pars), 2), dtype=F, device="cuda:0")
    wp.launch(
        advance,
        dim=(len(pars), len(i) + len(f)),
        inputs=[cfg, geo, f, i, o, hi, 2.9, out, buf],
        device="cuda:0",
    )
    return out.numpy()


def main():
    wp.init()
    (HERE / "ammoniation_network.json").write_text(
        json.dumps(enumerate_salt(), indent=2) + "\n"
    )
    t, *_, obs, hp, valid, so = measurements()
    mask = t[:181] - t[0] < 3.8
    from csv import DictReader

    # Every structure sees the identical physical parameter pairs already screened.
    with (HERE / "search.csv").open() as file:
        physical = sorted(
            set(
                (float(r["gas"]), float(r["wall"]), 0, 0, 0, -1)
                for r in DictReader(file)
            )
        )
    trials = []
    for n in [
        (0.0, 5.0, 2.0),
        *itertools.product([0.5, 1.0], [3.0, 5.0, 10.0, 15.0], [2.0, 4.0]),
    ]:
        p = replay(n, physical, 181)[:, :, 1]
        for pars, curve in zip(physical, p):
            trials.append(
                dict(
                    amplitude=n[0],
                    half_diameter_nm=n[1],
                    power=n[2],
                    acid=pars[0],
                    wall=pars[1],
                    train_rmse=np.mean((curve[mask] - obs[:181, 1][mask]) ** 2)
                    ** 0.5,
                    test_rmse=np.mean((curve[~mask] - obs[:181, 1][~mask]) ** 2)
                    ** 0.5,
                )
            )
    write("ammoniation_search.csv", trials)
    rows = []
    for name, rows0 in [
        ("parent", [r for r in trials if r["amplitude"] == 0]),
        ("positive", [r for r in trials if r["amplitude"] > 0]),
        ("nested", trials),
    ]:
        r = min(rows0, key=lambda z: z["train_rmse"])
        c = replay(
            (r["amplitude"], r["half_diameter_nm"], r["power"]),
            [(r["acid"], r["wall"], 0, 0, 0, -1)],
            len(t),
        )[0]
        late = t - t[0] > 6.0001
        rows.append(
            dict(
                family=name,
                **r,
                late_rmse=np.mean((c[late, 1] - obs[late, 1]) ** 2) ** 0.5,
                early_peak=c[:181, 1].max(),
            )
        )
    write("ammoniation_scores.csv", rows)


if __name__ == "__main__":
    main()
