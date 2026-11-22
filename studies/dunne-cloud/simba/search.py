"""Fit the training interval, freeze candidate parameters, and score time and size transfer."""

import csv
import json
import itertools
import subprocess
import numpy as np
import warp as wp
from cloud_networks import HERE, enumerate_bank
from replay import simulate, measurements


def write(name, rows):
    with (HERE / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def family(s):
    o, p = s[3:5]
    return (
        "both"
        if o and p
        else ("ozone" if o else "peroxide" if p else "physics")
    )


def run():
    wp.init()
    device = "cuda:0"
    assert wp.is_cuda_available()
    library = enumerate_bank()
    (HERE / "networks.json").write_text(json.dumps(library, indent=2) + "\n")
    t, *_, obs, hp, valid, so = measurements()
    hours = t - t[0]
    tr = hours[:181] < 3.8
    hist = {}

    def fit(config):
        configs = sorted(set(config) - hist.keys())
        for off in range(0, len(configs), 96):
            batch = configs[off : off + 96]
            pred, _ = simulate(batch)
            train_rmse = (
                np.mean((pred[:, tr, 1] - obs[:181][tr, 1]) ** 2, axis=1) ** 0.5
            )
            for s, e in zip(batch, train_rmse):
                hist[s] = float(e)

    def allowed(s, f):
        return (f in ("both", "ozone") or s[3] == 0) and (
            f in ("both", "peroxide") or s[4] == 0
        )

    def best(f, ext):
        return min(
            (s for s in hist if s[2] == ext and allowed(s, f)),
            key=lambda s: (hist[s], sum(s[3:5])),
        )

    def config(g, w, ext, o, p):
        return (round(g, 6), round(w, 6), ext, round(o, 10), round(p, 10), -1)

    ozone = [0, 0.00005, 0.0002, 0.0008, 0.0032]
    peroxide = [0, 0.001, 0.01, 0.1, 1.0]
    fit(
        config(*x)
        for x in itertools.product(
            [1.05, 1.15, 1.25],
            [0.5, 0.75, 1.0, 1.5, 2.0],
            [0, 1],
            ozone,
            peroxide,
        )
    )
    print("Coarse search complete", len(hist), flush=True)
    labels = ["physics", "ozone", "peroxide", "both"]
    for scale in [1.0, 0.4]:
        proposals = []
        for ext, f in itertools.product([0, 1], labels):
            b = best(f, ext)
            # Equal refinement in common gas/wall parameters. Nested candidates remain eligible.
            ranges = [
                np.clip(b[0] + np.array([-0.05, 0, 0.05]) * scale, 1.05, 1.25),
                np.clip(b[1] + np.array([-0.125, 0, 0.125]) * scale, 0.5, 2.0),
            ]
            for j, key, cap, small in [
                (3, "ozone", 0.0032, 0.000025),
                (4, "peroxide", 1.0, 0.0005),
            ]:
                if f not in ("both", key):
                    ranges.append([0.0])
                    continue
                v = b[j]
                add = max(v * 0.65, small) * scale
                ranges.append(np.clip([v - add, v, v + add], 0, cap))
            proposals.extend(
                config(g, w, ext, o, p)
                for g, w, o, p in itertools.product(*ranges)
            )
        fit(proposals)
    rows = [
        dict(
            gas=g,
            wall=w,
            extended=ex,
            gamma_ozone=o,
            gamma_peroxide=p,
            train_rmse=v,
        )
        for (g, w, ex, o, p, _), v in hist.items()
    ]
    write("search.csv", rows)
    models = {"original": config(1.15, 1.0, 0, 0, 0)}
    for ex, f in itertools.product([0, 1], labels):
        models[("continued" if ex else "original_kernel") + "_" + f] = best(
            f, ex
        )
    for ex in [0, 1]:
        for fam in ["ozone", "peroxide", "both"]:
            name = ("continued" if ex else "original_kernel") + "_" + fam
            b = models[name]
            parent = best("physics", ex)
            models[name + "_fixed_entry"] = (*b[:5], parent[0])
            models[name + "_ablation"] = (*b[:3], 0.0, 0.0, -1)
    # The final time block and auxiliary spectral channels are first scored here.
    keys = list(models)
    par = [models[k] for k in keys]
    unique = list(dict.fromkeys(par))
    res, bud = simulate(unique, count=len(t))
    fine, budfine = simulate(unique, count=181, step=15)
    truth = obs
    scores = []
    traces = []
    budgets = []

    def score(curve, ref, mask):
        v = curve[mask]
        y = ref[mask]
        return dict(
            rmse=float(np.mean((v - y) ** 2) ** 0.5),
            bias=float(np.mean(v - y)),
            pred_peak=float(v.max()),
            obs_peak=float(y.max()),
        )

    for key, s in models.items():
        a = unique.index(s)
        curve = res[a]
        for block, mask in [
            ("train", hours < 3.8),
            ("peak_holdout", (hours >= 3.8) & (hours <= 6.0001)),
            ("later", hours > 6.0001),
        ]:
            for c, chan in enumerate(["5_to_9p7", "9p7_plus", "15_plus"]):
                scores.append(
                    dict(
                        model=key,
                        block=block,
                        channel=chan,
                        **score(curve[:, c], truth[:, c], mask),
                    )
                )
        budgets.append(
            dict(
                model=key,
                SO2_used_molecules_cm3=sum(bud[a]),
                peroxide_used_molecules_cm3=bud[a, 1],
                ozone_used_molecules_cm3=bud[a, 0],
                fine_early_peak=fine[a, :, 1].max(),
                coarse_early_peak=curve[:181, 1].max(),
                fine_early_holdout_rmse=np.mean(
                    (fine[a, ~tr, 1] - truth[:181][~tr, 1]) ** 2
                )
                ** 0.5,
            )
        )
        if "fixed_entry" in key or "ablation" in key:
            continue
        for idx, h in enumerate(hours):
            traces.append(
                dict(
                    model=key,
                    hour=h,
                    obs_N10=truth[idx, 1],
                    pred_N10=curve[idx, 1],
                    obs_N15=truth[idx, 2],
                    pred_N15=curve[idx, 2],
                    obs_N5to10=truth[idx, 0],
                    pred_N5to10=curve[idx, 0],
                )
            )
    write("scores.csv", scores)
    write("budgets.csv", budgets)
    write("traces.csv", traces)
    write(
        "forcing.csv",
        [
            dict(
                hour=h,
                peroxide_proxy_cm3=hp[i],
                SO2_cm3=so[i],
                acid_measured=int(valid[i]),
            )
            for i, h in enumerate(hours)
        ],
    )
    import particula as parlib

    base = np.genfromtxt(
        HERE.parent / "results/warm_1833.csv",
        names=True,
        dtype=None,
        encoding="utf-8",
        delimiter=",",
    )
    i = (
        (base["model"] == "vdw_wet")
        & (base["gas_scale"] == 1.15)
        & (base["dt_s"] == 30)
        & (base["wall_scale"] == 1)
        & (base["coag"] == 0)
        & (base["boundary_nm"] == 2.9)
    )
    baseline = base["pred_N10_cm3"][i]
    diff = res[unique.index(models["original"]), :181, 1] - baseline
    parity = float(np.sqrt(np.mean(diff**2)) / baseline.max())
    assert parity < 0.01
    assert np.allclose(
        res[unique.index(models["original"]), 0, 1], truth[0, 1], rtol=0.02
    )
    assert np.isfinite(res).all() and np.min(res) >= 0 and np.min(bud) >= 0
    metadata = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        device=str(wp.get_device(device).name),
        particula=parlib.__version__,
        warp=wp.__version__,
        configs=len(hist),
        baseline_normalized_rmse=parity,
        model_params=models,
        columns=[
            "acid calibration",
            "wall scale",
            "continue attractive collisions",
            "ozone efficiency",
            "peroxide efficiency",
            "fixed boundary acid calibration (-1 for coupled)",
        ],
        peroxide_range_cm3=[float(hp.min()), float(hp.max())],
        acid_interpolated=int(sum(~valid)),
        spectra=len(t),
    )
    (HERE / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for r in scores:
        if (
            r["channel"] == "9p7_plus"
            and "ablation" not in r["model"]
            and "fixed_entry" not in r["model"]
        ):
            print(
                r["model"],
                r["block"],
                round(r["rmse"]),
                round(r["pred_peak"]),
                flush=True,
            )


if __name__ == "__main__":
    run()
