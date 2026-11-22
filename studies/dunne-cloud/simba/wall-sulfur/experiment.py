"""Repeat and extend sulfur tests with independently specified native wall curves."""

from functools import lru_cache
import itertools, json
import numpy as np
import warp as wp
from balance import HERE, write, wall
from sulfur_inputs import inputs
from replay import (
    geometry,
    GRID,
    advance,
    arrays,
    measurements,
    F,
    par,
    TEMP,
    PRESSURE,
    AIR,
    speed,
)
from bulk import aqueous, water_fraction
from population import transport

META = json.loads((HERE / "calibration.json").read_text())


def wall_profile(selection, d):
    if selection == "dark":
        return wall(d, META["eddy_dark_s"])
    if selection == "nominal":
        return wall(d, META["eddy_nominal_s"])
    if selection == "tilted":
        shape = np.genfromtxt(
            HERE / "wall_shape.csv",
            names=True,
            dtype=None,
            encoding="utf8",
            delimiter=",",
        )
        row = next(
            r for r in shape if r["leave_band"] == -1 and r["law"] == "tilted"
        )
        return (
            wall(d, META["eddy_nominal_s"])
            * row["scale_at10nm"]
            * (d / 10.0) ** row["tilt"]
        )
    values = np.genfromtxt(
        HERE / "wall_fits.csv",
        names=True,
        dtype=None,
        encoding="utf8",
        delimiter=",",
    )
    scale = next(r["scale"] for r in values if r["window"] == selection)
    return wall(d, META["eddy_nominal_s"]) * scale


@lru_cache(None)
def dimer_growth():
    # (H2SO4)2 · (NH3)2 as capturing representative of the unresolved dimer ensemble.
    diameter = np.geomspace(1, 10, 110)
    mass = (196.158 + 2 * 17.03052) * 1.66053906660e-27
    masses = np.r_[mass, 1500 * np.pi / 6 * (diameter * 1e-9) ** 3]
    particles = par.particles.ParticleData(
        masses=masses[None, :, None],
        density=np.array([1500.0]),
        concentration=np.ones((1, len(masses))),
        charge=np.zeros((1, len(masses))),
        volume=np.ones(1),
        distribution_type="discrete",
    )
    strategy = par.dynamics.CoagulationFactory().get_strategy(
        "van_der_waals",
        dict(
            distribution_type="discrete",
            hamaker_constant=6.4e-20,
            relative_permittivity=1.0,
        ),
    )
    k = strategy.kernel(particles, TEMP, PRESSURE)[0, 1:]
    diff = 8e-6 * (TEMP / 293.15) ** 1.75
    k2 = transport(
        GRID, TEMP, 0.23021904, diff * (116.09428 / 230.21904) ** (1 / 3)
    )
    k2[GRID <= 9.7] = np.interp(GRID[GRID <= 9.7], diameter, k)
    der_inv = geometry()[:, 3] / geometry()[:, 5]
    return k2 * 2e6 * der_inv


@lru_cache(None)
def forcing(model, gas):
    t, _, _, _, acid, of, pf, obs, hp_old, valid, so = measurements()
    hp, acidgas, acid_bg, _ = inputs()
    sulfate = acid * gas
    hp_impact = (
        1e6
        * np.minimum(so * speed(0.064066), hp * speed(0.034015))
        * np.minimum(so / (2e-9 * AIR), 1.0)
    )
    g = geometry().copy()
    if model == "gas_oh" or model == "gas_plus_background":
        # The gas reconstruction already predicts an absolute acid concentration.
        sulfate = (acidgas if model == "gas_oh" else acid_bg) * gas
    if model == "gas_equal_training_mean":
        m = (t - t[0] < 3.8) & valid
        sulfate = acidgas * gas * np.mean(acid[m]) / np.mean(acidgas[m])
    if model.startswith("dimer"):
        data = np.genfromtxt(
            HERE / "dimer_lookup.csv",
            names=True,
            dtype=None,
            encoding="utf8",
            delimiter=",",
        )
        lut = data[
            data["thermo"]
            == ("optimized" if "optimized" in model else "posterior_median")
        ]
        counts = lut["dimer_cm3"]
        m = lut["monomer_cm3"]
        x = m + 2 * counts if "pooled" in model else m
        assert sulfate.min() >= 0 and sulfate.max() < x[-1]
        nc = np.interp(sulfate, np.r_[0.0, x], np.r_[0.0, counts])
        if "pooled" in model:
            sulfate = sulfate - 2 * nc
            assert np.all(sulfate >= 0)
        g[:, 3] = dimer_growth()
        return g, np.column_stack([sulfate, nc, np.zeros_like(nc)])
    if model.startswith("bulk"):
        oa, pa = aqueous(float(model[4:]))
        liters = np.pi / 6 * (GRID * 1e-9) ** 3 * 1e3 * water_fraction(GRID)
        g[:, 3] *= liters / g[:, 5]
        return g, np.column_stack(
            [sulfate, oa * 6.02214076e23, pa * hp / hp_old * 6.02214076e23]
        )
    return g, np.column_stack([sulfate, of, hp_impact])


def simulate(model, selection, gas, settings, count, step):
    dev = "cuda:0"
    (_, f, i, o), hi = arrays(count, step, dev)
    base = f.numpy()
    mid = np.cumsum(base[:, 0]) - 0.5 * base[:, 0]
    t = measurements()[0]
    ts = (t - t[0]) * 3600
    g, v = forcing(model, gas)
    g = g.copy()
    g[:, 4] = wall_profile(selection, GRID)
    for idx in range(3):
        base[:, 1 + idx] = np.interp(mid, ts, v[:, idx])
    cfg = np.array([[1.0, 1.0, 0.0, a, b, -1.0] for a, b in settings])
    grid = wp.array(g, dtype=F, device=dev)
    fc = wp.array(base, dtype=F, device=dev)
    conf = wp.array(cfg, dtype=F, device=dev)
    out = wp.zeros((len(cfg), count, 3), dtype=F, device=dev)
    budget = wp.zeros((len(cfg), 2), dtype=F, device=dev)
    wp.launch(
        advance,
        dim=(len(cfg), len(i) + len(f)),
        inputs=[conf, grid, fc, i, o, hi, 2.9, out, budget],
        device=dev,
    )
    return out.numpy()


def main():
    wp.init()
    t, *_, obs, hp, valid, so = measurements()
    h = t - t[0]
    train = h[:181] < 3.8
    gas_scales = [1.05, 1.10, 1.15, 1.20, 1.25]
    mainwall = "dark"
    scores = []
    params = []
    traces = []
    screen = []
    masks = {
        "train": h < 3.8,
        "peak": (h >= 3.8) & (h <= 6.0001),
        "post_clean_start": h >= 6.36,
        "late_after_11h": h >= 11.0,
    }
    for selection in ["dark", "tilted", "nominal", "no_coag", "double_coag"]:
        bank = ["effective"]
        if selection == "dark":
            bank += [
                "gas_oh",
                "gas_plus_background",
                "bulk2",
                "bulk4",
                "bulk6",
                *[
                    "dimer_" + s + "_" + b
                    for s, b in itertools.product(
                        ["optimized", "median"], ["pooled", "monomer"]
                    )
                ],
            ]
        chosen = {}
        for model in bank:
            samples = []
            settings = [(0, 0)]
            if model == "effective":
                settings = list(
                    itertools.product(
                        [
                            0,
                            0.00005,
                            0.0001,
                            0.0002,
                            0.0003,
                            0.0004,
                            0.0006,
                            0.0008,
                            0.0016,
                            0.0032,
                        ],
                        [
                            0,
                            0.0003,
                            0.001,
                            0.003,
                            0.01,
                            0.02,
                            0.04,
                            0.08,
                            0.16,
                            0.32,
                            0.64,
                            1.0,
                        ],
                    )
                )
            elif model.startswith("bulk"):
                settings = [(1.0, 1.0)]
            elif model.startswith("dimer"):
                settings = [(1.0, 0)]
            scales = (
                [0.9, 0.95, 1.0, 1.05, 1.10]
                if model.startswith("gas")
                else gas_scales
            )
            for gas in scales:
                c = simulate(model, selection, gas, settings, 181, 30)
                for (a, b), pred in zip(settings, c):
                    loss = (
                        np.mean((pred[train, 1] - obs[:181, 1][train]) ** 2)
                        ** 0.5
                    )
                    samples.append((loss, gas, a, b))
                    screen.append(
                        dict(
                            wall=selection,
                            model=model,
                            gas=gas,
                            a=a,
                            b=b,
                            train=loss,
                        )
                    )
            for label, subset in (
                (
                    ("parent", [r for r in samples if r[2] == r[3] == 0]),
                    ("ozone", [r for r in samples if r[3] == 0]),
                    ("peroxide", [r for r in samples if r[2] == 0]),
                    ("both", samples),
                )
                if model == "effective"
                else [(model, samples)]
            ):
                loss, gas, a, b = min(subset)
                chosen[label] = (model, gas, a, b)
        for label, (model, gas, a, b) in chosen.items():
            c = simulate(model, selection, gas, [(a, b)], len(t), 30)[0]
            assert np.all(np.isfinite(c)) and np.all(c >= 0)
            fine = simulate(model, selection, gas, [(a, b)], 181, 15)[0]
            params.append(
                dict(
                    wall=selection,
                    network=label,
                    gas=gas,
                    ozone_or_first=a,
                    peroxide_or_second=b,
                    refined_peak_rmse=np.mean(
                        (fine[~train, 1] - obs[:181, 1][~train]) ** 2
                    )
                    ** 0.5,
                )
            )
            for tag, mask in masks.items():
                for idx, size in enumerate(["5_to_9p7", "9p7_plus", "15_plus"]):
                    y, p = obs[mask, idx], c[mask, idx]
                    scores.append(
                        dict(
                            wall=selection,
                            network=label,
                            period=tag,
                            channel=size,
                            rmse=np.mean((p - y) ** 2) ** 0.5,
                            peak=p.max(),
                            observed_peak=y.max(),
                            bias=np.mean(p - y),
                        )
                    )
            if selection == mainwall:
                traces.extend(
                    dict(network=label, hour=hh, pred_N10=pred, obs_N10=target)
                    for hh, pred, target in zip(h, c[:, 1], obs[:, 1])
                )
    write("search.csv", screen)
    write("parameters.csv", params)
    write("scores.csv", scores)
    write("traces.csv", traces)


if __name__ == "__main__":
    main()
