"""Recreate optical spectra directly from the author archive and fitted chamber settings."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent


def observations(exp):
    stages = np.searchsorted([10, 18], np.arange(exp.nmin), side="right")
    groups = np.floor(
        8 * np.log(exp.d / exp.edges[0]) / np.log(exp.edges[-1] / exp.edges[0])
    )
    return dict(
        y=exp.obs,
        cpi=exp.cpi,
        vapor=exp.vapor,
        valid=exp.valid,
        d=exp.d,
        edges=exp.edges,
        stage=stages,
        band=np.minimum(7, groups).astype(int),
        cut=np.array([np.flatnonzero(v)[-10] for v in exp.valid]),
    )


def score(exp, obs, y, x, p):
    """Separate the recorded optical, band, vapor and imager losses by run/window."""
    import scoring

    scorer = object.__new__(scoring.Scoring)
    scorer.e = exp
    scorer.cut = obs["cut"]
    scorer.mask = np.array(
        [(exp.d >= a) & (exp.d < b) for a, b in scoring.BANDS], float
    )
    widths = np.diff(np.log10(exp.edges))
    scorer.size_weight = 1 / (8 * np.bincount(obs["band"])[obs["band"]])
    scorer.sim = SimpleNamespace(
        distance=lambda prediction: np.abs(
            np.log10(1 + prediction / (widths * 0.1))
            - np.log10(1 + np.nan_to_num(exp.obs) / (widths * 0.1))
        )
    )
    scorer.profile = {}
    for r in range(6):
        for w in scoring.WINDOWS:
            valid = exp.valid[r].copy()
            if w == "prefix":
                valid[scorer.cut[r] :] = False
            if w == "late":
                valid[: scorer.cut[r]] = False
            t = np.flatnonzero(valid)
            labels = obs["stage"][t]
            weights = np.zeros(len(t))
            for k in np.unique(labels):
                weights[labels == k] = 1 / (
                    len(np.unique(labels)) * np.sum(labels == k)
                )
            scorer.profile[r, w] = t, weights
    correction = (
        1 + p[5] + p[6] * (1 - exp.force[:, ::60, 1] / exp.force[:, 0, 1, None])
    )
    vapor = x[..., 16] * correction
    habit = np.zeros(6)
    for r, m, b in [(1, 35, 12), (5, 30, 14)]:
        expected = (
            exp.cpi[r, m].sum()
            * x[r, m, b:16].sum()
            / max(x[r, m, :8].sum(), 1.0e-12)
        )
        habit[r] = max(0, -np.log10(0.01 + expected))
    return scorer.scores(dict(y=y, x=x, vmodel=vapor, habit=habit)), vapor


def calculate(
    exp,
    heat,
    grid,
    material,
    settings,
    case,
    dt,
    obs,
    *,
    archived_fill=False,
    uniform_vapor=False,
):
    from measured_forcing import impose_cold_probe
    from solver import instrument

    c = settings["cases"][case]
    params = np.array(c["microphysics"])[None, :]
    if archived_fill:
        params[0, 62] = 0.0
    if uniform_vapor:
        params[0, 57] = 0.0
    speed, exchange, film, vertical = c["flow_thermal"]
    temperature = np.array(settings["thermal_base"])[None, :]
    temperature[0, [0, 1, 2, 6]] = [speed, exchange, film, vertical]
    matrices = grid.matrix(speed, exchange, dt)[None, :, :]
    field = heat.solve(
        temperature, grid, "volume", dt, retain=True, mixing_matrices=matrices
    )
    impose_cold_probe(heat, field, grid, dt, 1.0, matrices)
    group = case.split("-")[0]
    runs = np.arange(6) if group == "pooled" else np.array([int(group[1:]) - 1])
    force = exp.force.copy()
    force[~np.isin(np.arange(6), runs), :, 12] = 0.0
    exp.ff = exp.arr(force)
    n = 6 * grid.size
    wetwall = wp.zeros(n, dtype=float, device=exp.device)
    ledger = wp.zeros((n, 8), dtype=float, device=exp.device)
    y, x, budget = exp.parcels(
        params,
        [[speed, exchange, 0, settings["salt_localization"]]],
        grid,
        physics="native_full",
        dt=dt,
        nc={1: 80, 2: 40, 5: 20}[dt],
        ni={1: 168, 2: 84, 5: 54}[dt],
        ns={1: 48, 2: 24, 5: 12}[dt],
        mixing_matrices=matrices,
        state_edit=SimpleNamespace(extra_slots=3 * exp.nb + 9, seconds=set()),
        step_inputs=[
            exp.arr(field["forcing"]),
            exp.arr([settings["wall_vapor_efficiency"]]),
            wetwall,
            exp.arr(material[None, :, :]),
            ledger,
        ],
    )
    y, x = instrument(grid, y, x)
    y = y[0].sum(axis=-2)
    x = x[0]
    scores, vapor = score(exp, obs, y, x, params[0])
    return dict(
        runs=runs,
        y=y[runs],
        x=x[runs],
        scores=scores[runs],
        p=params[0],
        vmodel=vapor[runs],
        budget=budget[0, runs].mean(axis=1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Local output directory for regenerated NPZ files",
    )
    parser.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="Local cache for checksum-verified Figshare inputs",
    )
    parser.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="Local cache for computed Particula transport tables and Warp kernels",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Warp device, for example cuda:0 or cpu",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        help="Pooled or individual case names, e.g. pooled-full R5-full. Default is all seven full fits.",
    )
    parser.add_argument(
        "--dt",
        nargs="+",
        type=int,
        choices=[1, 2],
        default=[1, 2],
        help="Time step in seconds",
    )
    parser.add_argument(
        "--a2-table",
        type=Path,
        help="Optional replacement for the supplier's tabulated source specification",
    )
    parser.add_argument(
        "--run5-challenge",
        action="store_true",
        help="Run the matched source and local-vapor perturbations under frozen run-5 settings",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ["JEJU_RAW"] = str(args.raw.resolve())
    if args.a2_table:
        os.environ["JEJU_A2_TABLE"] = str(args.a2_table.resolve())
    wp.config.kernel_cache_dir = str(args.cache / "warp")
    from dust import source_arrays
    from grid import ChamberGrid, ChamberHeat
    from solver import Circulation
    from thermal import archives

    settings = json.loads((HERE / "settings.json").read_text())
    cases = args.case or ["pooled-full", *[f"R{r}-full" for r in range(1, 7)]]
    for name in cases:
        if name not in settings["cases"]:
            parser.error(f"Unknown selected parameter set {name}")
    exp = Circulation(args.cache, device=args.device)
    obs = observations(exp)
    np.savez_compressed(args.out / "observations.npz", **obs)
    heat = ChamberHeat(archives()[0])
    grid = ChamberGrid(6, 1)
    material = source_arrays()
    if args.run5_challenge:
        from sensitivity import perturb

        perturb(exp, heat, grid, material, settings, obs, args.dt, args.out)
        return
    for case in cases:
        for dt in args.dt:
            result = calculate(
                exp, heat, grid, material, settings, case, dt, obs
            )
            np.savez_compressed(args.out / f"{case}-{dt}s.npz", **result)
            print(case, dt, "s", result["scores"][:, 0, 0], flush=True)


if __name__ == "__main__":
    main()
